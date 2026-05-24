"""Local ArcFace embedder trainer (one-off bootstrap).

Ports the metric-learning trainer from sets-warp-backend/admin_train_metric.py
(read-only reference) into sto-warp, adapted for local execution against
on-disk crops only — no HuggingFace round-trip.

Why local:
  - 10,600 synthetic crops (100 × 106 ground BOFF classes) would take ~53 days
    through HF staging at 200 crops/install/UTC day. Local training closes the
    coverage gap in one shot; the resulting .pt is then *manually* uploaded to
    central HF (sets-sto/warp-knowledge/models/) so every other install pulls
    the new gallery on next sync. Central trainer resumes normal operation
    after that.

Inputs:
  - Real crops:      ~/.local/share/warp/training_data/crops/<slot>__<slug>__<hash>.png
    Canonical class label resolved via crops/crop_index.json (key: 'name').
  - Synthetic crops: ~/.local/share/warp/training_data/synthetic_crops/<env>/<class_slug>/<seq>.png
    Canonical class label resolved by reverse-slug lookup against cargo
    boff_abilities (env-scoped).

Outputs (in userdata.models_dir()):
  - icon_embedder.pt           — backbone + projection state_dict
  - embedding_index.npz        — gallery embeddings (full train set, no aug)
  - embedder_label_map.json    — int → canonical name
  - icon_embedder_meta.json    — hyper-params + val_recall@1

Warm-start: existing icon_embedder.pt in models_dir (preserves space coverage
when present). Falls back to ImageNet weights if no prior embedder exists.

Run:
    python -m warp.trainer.embedder_trainer --train
    python -m warp.trainer.embedder_trainer --generate-synthetic --env ground -n 100 --train
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from warp import userdata
from warp.data.cargo import boff_abilities
from warp.debug import log


# ── Hyper-parameters (mirror sets-warp-backend/admin_train_metric.py) ────────

EMBED_DIM      = 256
ARC_MARGIN     = 0.5
ARC_SCALE      = 30.0
PK_P           = 8
PK_K           = 4
BATCH_SIZE     = PK_P * PK_K       # 32
MAX_EPOCHS     = 30
LR             = 3e-4
PATIENCE       = 5
BATCHES_PER_EPOCH_MIN = 60
MIN_SAMPLES    = 5
IMG_SIZE       = 64                # crops stored at this resolution
MODEL_IMG_SIZE = 224               # backbone input

UTC = timezone.utc

# Real-crop filename: <slot>__<class_slug>__<12hex>.png (12+ hex chars)
_CROP_RE = re.compile(r'^(.+)__(.+)__[0-9a-f]{8,}\.png$', re.IGNORECASE)


def _slug(name: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')


# ── Crop loading ─────────────────────────────────────────────────────────────

def _load_real_crops(crops_dir: Path) -> list[tuple[np.ndarray, str]]:
    """Real crops are flat files with name resolved via crop_index.json.
    Filenames alone use slugs and lose case/punctuation; the index has the
    canonical label needed for label-map consistency with the existing
    embedder."""
    index_path = crops_dir / 'crop_index.json'
    if not index_path.is_file():
        log.warning(f'EmbedderTrainer: crop_index.json missing at {index_path} — skipping real crops')
        return []
    try:
        idx = json.loads(index_path.read_text(encoding='utf-8'))
    except Exception as e:
        log.warning(f'EmbedderTrainer: failed to parse crop_index.json: {e}')
        return []

    out: list[tuple[np.ndarray, str]] = []
    skipped_meta = skipped_read = 0
    for fname, meta in idx.items():
        name = (meta or {}).get('name')
        # Keep `__inactive__` / `__empty__` — they are legitimate classes the
        # embedder must learn so empty/greyed-out slots get recognised as such
        # instead of nearest-neighbour-snapping to a random real ability.
        if not isinstance(name, str) or not name:
            skipped_meta += 1
            continue
        p = crops_dir / fname
        if not p.is_file():
            skipped_meta += 1
            continue
        img = cv2.imread(str(p))
        if img is None:
            skipped_read += 1
            continue
        if img.shape[:2] != (IMG_SIZE, IMG_SIZE):
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        out.append((img, name))
    log.info(f'EmbedderTrainer: loaded {len(out)} real crops '
             f'(skipped {skipped_meta} meta, {skipped_read} unreadable)')
    return out


def _slug_to_canonical(env: str | None) -> dict[str, str]:
    """Build `<class_slug> → canonical name` map from cargo boff_abilities.
    `env` is the synthetic_crops subdir name ('ground' / 'space'). If unknown,
    union both buckets."""
    cache = boff_abilities()
    if env in ('ground', 'space'):
        buckets = [cache.get(env) or {}]
    else:
        buckets = [cache.get('ground') or {}, cache.get('space') or {}]
    mapping: dict[str, str] = {}
    for env_dict in buckets:
        for _prof, rank_lists in env_dict.items():
            if not isinstance(rank_lists, (list, tuple)):
                continue
            for rank_dict in rank_lists:
                if not isinstance(rank_dict, dict):
                    continue
                for name in rank_dict.keys():
                    mapping.setdefault(_slug(name), name)
    return mapping


def _load_synthetic_crops(synth_root: Path) -> list[tuple[np.ndarray, str]]:
    """Synthetic crops live under synth_root/<env>/<class_slug>/<seq>.png.
    Folder name is the slug; canonical name resolved via cargo."""
    if not synth_root.is_dir():
        return []
    out: list[tuple[np.ndarray, str]] = []
    for env_dir in sorted(synth_root.iterdir()):
        if not env_dir.is_dir():
            continue
        env = env_dir.name if env_dir.name in ('ground', 'space') else None
        slug_map = _slug_to_canonical(env)
        env_total = 0
        env_unknown = 0
        for class_dir in sorted(env_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            canonical = slug_map.get(class_dir.name)
            if canonical is None:
                env_unknown += 1
                continue
            for png in sorted(class_dir.glob('*.png')):
                img = cv2.imread(str(png))
                if img is None:
                    continue
                if img.shape[:2] != (IMG_SIZE, IMG_SIZE):
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                out.append((img, canonical))
                env_total += 1
        log.info(f'EmbedderTrainer: synthetic[{env_dir.name}] = {env_total} crops '
                 f'({env_unknown} class dirs without cargo match)')
    return out


# ── ArcFace + PK sampler (1:1 with admin_train_metric.py) ────────────────────

def _build_embedder(prev_model_pt: Path | None):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.models as tv_models

    backbone = tv_models.efficientnet_b0(weights=tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = backbone.classifier[1].in_features
    backbone.classifier = nn.Identity()

    class Embedder(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.proj = nn.Linear(in_features, EMBED_DIM)

        def forward(self, x):
            f = self.backbone(x)
            e = self.proj(f)
            return F.normalize(e, dim=1)

    model = Embedder()
    if prev_model_pt and prev_model_pt.exists():
        try:
            state = torch.load(str(prev_model_pt), map_location='cpu')
            missing, unexpected = model.load_state_dict(state, strict=False)
            log.info(f'EmbedderTrainer: warm-started from {prev_model_pt.name} '
                     f'(missing={len(missing)}, unexpected={len(unexpected)})')
        except Exception as e:
            log.warning(f'EmbedderTrainer: warm-start failed ({e}) — using ImageNet weights')
    else:
        log.info('EmbedderTrainer: no prior embedder — starting from ImageNet weights')
    return model


def _build_arcface_head(embed_dim: int, n_classes: int):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class ArcFaceHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.W = nn.Parameter(torch.empty(n_classes, embed_dim))
            nn.init.xavier_normal_(self.W)
            self.margin = ARC_MARGIN
            self.scale = ARC_SCALE

        def forward(self, emb, labels=None):
            W = F.normalize(self.W, dim=1)
            cos = emb @ W.t()
            if labels is None:
                return cos * self.scale
            cos = cos.clamp(-1 + 1e-7, 1 - 1e-7)
            theta = torch.acos(cos)
            target = torch.cos(theta + self.margin)
            one_hot = F.one_hot(labels, num_classes=cos.shape[1]).float()
            logits = cos * (1 - one_hot) + target * one_hot
            return logits * self.scale

    return ArcFaceHead()


class _PKBatchSampler:
    def __init__(self, labels: list[int], P: int, K: int, num_batches: int):
        self.P, self.K, self.num_batches = P, K, num_batches
        self.label_to_idx: dict[int, list[int]] = defaultdict(list)
        for i, lbl in enumerate(labels):
            self.label_to_idx[lbl].append(i)
        self.classes = list(self.label_to_idx.keys())

    def __iter__(self):
        for _ in range(self.num_batches):
            picked = random.sample(self.classes, min(self.P, len(self.classes)))
            batch = []
            for c in picked:
                pool = self.label_to_idx[c]
                if len(pool) >= self.K:
                    batch.extend(random.sample(pool, self.K))
                else:
                    batch.extend(random.choices(pool, k=self.K))
            yield batch

    def __len__(self):
        return self.num_batches


def _build_gallery(model, ds, device, batch_size: int = 64):
    import torch
    model.eval()
    embs, lbls = [], []
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    with torch.no_grad():
        for xb, yb in dl:
            e = model(xb.to(device))
            embs.append(e.cpu().numpy())
            lbls.append(yb.numpy())
    return (np.concatenate(embs).astype('float32'),
            np.concatenate(lbls).astype('int32'))


def _recall_at_1(g_emb, g_lbl, q_emb, q_lbl) -> float:
    if len(q_emb) == 0:
        return 0.0
    sims = q_emb @ g_emb.T
    pred = g_lbl[np.argmax(sims, axis=1)]
    return float((pred == q_lbl).mean())


# ── Core fit loop ────────────────────────────────────────────────────────────

def _fit(crops: list[np.ndarray], labels: list[str],
         models_dir: Path,
         prev_model_pt: Path | None,
         deadline: float | None = None) -> tuple[float, int]:
    import torch
    import torchvision.transforms as T

    n = len(crops)
    if n < MIN_SAMPLES:
        raise RuntimeError(f'Only {n} crops available (need {MIN_SAMPLES}).')

    unique_labels = sorted(set(labels))
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    idx_to_label = {i: l for l, i in label_to_idx.items()}
    n_classes = len(unique_labels)
    y = [label_to_idx[l] for l in labels]
    log.info(f'EmbedderTrainer: {n} crops, {n_classes} classes')

    transform_train = T.Compose([
        T.ToPILImage(),
        T.RandomResizedCrop(MODEL_IMG_SIZE, scale=(0.8, 1.0)),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        T.RandomHorizontalFlip(p=0.3),
        T.RandomAffine(degrees=5, translate=(0.05, 0.05)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    transform_val = T.Compose([
        T.ToPILImage(),
        T.Resize((MODEL_IMG_SIZE, MODEL_IMG_SIZE)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    class CropDataset(torch.utils.data.Dataset):
        def __init__(self, crops, labels, tf):
            self.crops, self.labels, self.tf = crops, labels, tf
        def __len__(self):
            return len(self.crops)
        def __getitem__(self, i):
            return self.tf(cv2.cvtColor(self.crops[i], cv2.COLOR_BGR2RGB)), self.labels[i]

    by_cls: dict[int, list[int]] = defaultdict(list)
    for i, lbl in enumerate(y):
        by_cls[lbl].append(i)
    train_idx, val_idx = [], []
    for lbl, idxs in by_cls.items():
        random.shuffle(idxs)
        if len(idxs) >= 2:
            val_idx.append(idxs[0])
            train_idx.extend(idxs[1:])
        else:
            train_idx.extend(idxs)
    random.shuffle(train_idx)

    train_labels = [y[i] for i in train_idx]
    val_labels = [y[i] for i in val_idx]

    ds_train = CropDataset([crops[i] for i in train_idx], train_labels, transform_train)
    ds_train_eval = CropDataset([crops[i] for i in train_idx], train_labels, transform_val)
    ds_val = CropDataset([crops[i] for i in val_idx], val_labels, transform_val)

    batches_per_epoch = max(BATCHES_PER_EPOCH_MIN, len(ds_train) // BATCH_SIZE)
    pk = _PKBatchSampler(train_labels, PK_P, PK_K, batches_per_epoch)
    dl_train = torch.utils.data.DataLoader(ds_train, batch_sampler=pk, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f'EmbedderTrainer: device={device}, batches/epoch={batches_per_epoch}, '
             f'train={len(train_idx)}, val={len(val_idx)}')

    model = _build_embedder(prev_model_pt).to(device)
    head = _build_arcface_head(EMBED_DIM, n_classes).to(device)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    best_recall = 0.0
    best_state = None
    patience_count = 0

    for epoch in range(MAX_EPOCHS):
        if deadline is not None and time.monotonic() > deadline:
            log.info(f'EmbedderTrainer: time budget exceeded at epoch {epoch+1}')
            break

        model.train(); head.train()
        loss_sum = 0.0
        n_batch = 0
        for xb, yb in dl_train:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            emb = model(xb)
            logits = head(emb, yb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            n_batch += 1
        scheduler.step()
        avg_loss = loss_sum / max(1, n_batch)

        g_emb, g_lbl = _build_gallery(model, ds_train_eval, device)
        q_emb, q_lbl = _build_gallery(model, ds_val, device)
        val_recall = _recall_at_1(g_emb, g_lbl, q_emb, q_lbl)
        log.info(f'EmbedderTrainer: epoch {epoch+1:2d}/{MAX_EPOCHS} '
                 f'loss={avg_loss:.3f} val_recall@1={val_recall:.1%} best={best_recall:.1%}')

        if val_recall > best_recall:
            best_recall = val_recall
            best_state = {
                'model': {k: v.cpu().clone() for k, v in model.state_dict().items()},
                'head': {k: v.cpu().clone() for k, v in head.state_dict().items()},
            }
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                log.info(f'EmbedderTrainer: early stop at epoch {epoch+1}')
                break

    if best_state:
        model.load_state_dict(best_state['model'])
        head.load_state_dict(best_state['head'])

    models_dir.mkdir(parents=True, exist_ok=True)
    model.eval().cpu()

    ds_all = CropDataset(crops, y, transform_val)
    full_emb, full_lbl = _build_gallery(model, ds_all, torch.device('cpu'))

    torch.save(model.state_dict(), str(models_dir / 'icon_embedder.pt'))
    np.savez(str(models_dir / 'embedding_index.npz'),
             embeddings=full_emb, labels=full_lbl)
    with open(models_dir / 'embedder_label_map.json', 'w', encoding='utf-8') as f:
        json.dump({str(i): l for i, l in idx_to_label.items()},
                  f, ensure_ascii=False, indent=2)
    with open(models_dir / 'icon_embedder_meta.json', 'w', encoding='utf-8') as f:
        json.dump({
            'n_classes':    n_classes,
            'embed_dim':    EMBED_DIM,
            'input_size':   MODEL_IMG_SIZE,
            'val_recall@1': best_recall,
            'gallery_size': len(full_emb),
            'arc_margin':   ARC_MARGIN,
            'arc_scale':    ARC_SCALE,
            'pk_p':         PK_P,
            'pk_k':         PK_K,
            'trained_at':   datetime.now(UTC).isoformat() + 'Z',
            'source':       'local-bootstrap',
        }, f, indent=2)

    log.info(f'EmbedderTrainer: saved icon_embedder.pt — {n_classes} classes, '
             f'val_recall@1={best_recall:.1%}, gallery={len(full_emb)}')
    return best_recall, n


# ── Public entry points ──────────────────────────────────────────────────────

def train_local(extra_synthetic_root: Path | None = None,
                seed: int = 42) -> tuple[float, int]:
    """Train embedder from real + synthetic crops on disk.
    Returns (val_recall@1, n_samples)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass

    td = userdata.training_data_dir()
    md = userdata.models_dir()

    real = _load_real_crops(td / 'crops')
    synth_root = extra_synthetic_root or (td / 'synthetic_crops')
    synth = _load_synthetic_crops(synth_root)

    samples = real + synth
    if not samples:
        raise RuntimeError(f'No crops found in {td}/crops or {synth_root}')

    crops = [c for c, _ in samples]
    labels = [l for _, l in samples]
    log.info(f'EmbedderTrainer: total {len(samples)} samples '
             f'(real={len(real)}, synthetic={len(synth)})')

    prev_pt = md / 'icon_embedder.pt'
    return _fit(crops, labels, models_dir=md,
                prev_model_pt=prev_pt if prev_pt.exists() else None)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Local ArcFace embedder trainer for ground BOFF bootstrap.'
    )
    parser.add_argument('--generate-synthetic', action='store_true',
                        help='Run synthetic_crop_generator before training')
    parser.add_argument('--env', choices=['ground', 'space'], default='ground',
                        help='Env for synthetic generation (default: ground)')
    parser.add_argument('-n', '--n-per-class', type=int, default=100,
                        help='Synthetic crops per class (default: 100)')
    parser.add_argument('--train', action='store_true',
                        help='Train embedder from on-disk real + synthetic crops')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if not (args.generate_synthetic or args.train):
        parser.error('nothing to do — pass --generate-synthetic and/or --train')

    if args.generate_synthetic:
        from warp.trainer.synthetic_crop_generator import generate_for_env, _default_output
        out = _default_output(args.env)
        n_written, missing = generate_for_env(args.env, args.n_per_class, out, seed=args.seed)
        print(f'Synthetic: wrote {n_written} crops to {out}')
        if missing:
            print(f'WARNING: {len(missing)} classes skipped (no local wiki PNG)')

    if args.train:
        recall, n = train_local(seed=args.seed)
        print(f'Trained on {n} crops — val_recall@1={recall:.1%}')
        print(f'Outputs in {userdata.models_dir()}')


if __name__ == '__main__':
    main()
