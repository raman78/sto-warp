# sto-warp ML Pipeline — Technical Reference

How machine learning works end-to-end in sto-warp: what the client captures
locally, how that data reaches the community, where the central model is
trained, and how updates flow back to every install.

---

## Overview

sto-warp uses two production classifiers plus one embedder:

| Model | Architecture | Purpose |
|-------|-------------|---------|
| `icon_classifier.pt` | EfficientNet-B0 | Matches item icon crops to item names |
| `screen_classifier.pt` | MobileNetV3-Small | Classifies screenshot type (SPACE_EQ, BOFFS, TRAITS, …) |
| `icon_embedder.pt` | EfficientNet-B0 + ArcFace head | k-NN gallery lookup used as a confidence cross-check |

**The client does not train the production models.** All three files are
downloaded from `sets-sto/warp-knowledge` on HuggingFace (see §4). The
client's contribution is the *data* — confirmed icon crops, confirmed screen
types, and OCR corrections — uploaded to staging and folded into the
community model by the central pipeline (§3).

A separate one-shot admin path for bootstrapping the embedder on synthetic
crops lives at the bottom of this document (§9). End users do not run it.

---

## 1. Local data capture

### Trigger

The user accepts a bounding box in WARP CORE (Enter, autocomplete pick,
**Accept** button, or auto-accept ≥ threshold).

### Input data

```
~/.local/share/warp/training_data/
├── annotations.json          ← confirmed bbox + label records
├── crops/
│   ├── <sha256>.png          ← 64×64 px icon crop per confirmed item
│   └── ...
└── screen_types/
    ├── SPACE_EQ/
    │   └── <filename>.png    ← confirmed full screenshot per type
    └── ...
```

Every confirmation:
1. Crops the bounding box from the screenshot.
2. Saves the crop as `crops/<sha256>.png`.
3. Appends a record to `annotations.json` (`slot`, `name`, `bbox`,
   `crop_sha256`, `confirmed_at`).

Screen type labels are saved separately when you tick / change the screen
type for a file (stored in `screen_types/<TYPE>/<filename>.png`). The
crop is also fed to the in-session matcher immediately, so the next
Auto-Detect on a different screenshot can already match against the
just-confirmed icon — see [`docs/WARP_GUIDE.md` §6](WARP_GUIDE.md#6-confirming-items-and-accepting-results)
for the user-side view.

### No local production training

Earlier sets-warp releases shipped a "Train Model" button that produced a
local icon classifier. That path was removed before sto-warp 1.0.0. The
client today only **captures** training data; production training is
performed centrally (§3). The only training code still callable on a user
machine is the embedder bootstrap (§9), which targets the k-NN gallery, not
the softmax classifier, and is gated behind a CLI flag.

---

## 2. Community upload (local → HuggingFace staging)

### When it happens

WARP CORE runs a background sync timer every 10 minutes. On each tick it
checks for unsynced confirmed crops and POSTs them in batches to the HF
Spaces backend (`sets-sto-warp-backend.hf.space`), which holds the HF
write token as a server-side secret. Since v1.0.5 the client holds no HF
credentials.

File: `warp/trainer/sync.py` — `SyncWorker`

### Rate limiting

Two independent caps, enforced client-side per install_id per UTC day:

| Channel | Cap | Constant | Notes |
|---|---|---|---|
| Crops + screen types (`SyncWorker`) | **1000 / day** | `MAX_DAILY_UPLOADS` in `warp/trainer/sync.py` | Corrections to a previously-uploaded crop do not count against the cap. |
| pHash knowledge contributions (`WARPSyncClient.contribute`) | **200 / day** | `MAX_CONTRIBUTIONS_PER_DAY` in `warp/knowledge/sync_client.py` | Per-icon pHash overrides only — does not gate the main crop upload path. |

The counters are persisted in the per-channel state files under
`~/.config/warp/` so they survive restarts within the same UTC day.

### What is uploaded

**Icon crops** → `sets-sto/sto-icon-dataset` (HF Dataset):
```
staging/<install_id>/crops/<sha256>.png   ← icon crop image (64×64 px)
staging/<install_id>/annotations.jsonl    ← one JSON line per crop:
    {"crop_sha256": "...", "name": "Ablative Shell", "slot": "Science Console", ...}
```

**Screen type screenshots** → same repo:
```
staging/<install_id>/screen_types/<TYPE>/<sha256>.png
```

**Ship Type / Ship Tier text crops** → same staging repo:
```
staging/<install_id>/crops/<sha256>.png       ← text region crop (same path as icon crops)
staging/<install_id>/annotations.jsonl        ← entry per text crop:
    {"crop_sha256": "...", "name": "Fleet Support Cruiser",
     "slot": "Ship Type", "ml_name": "F1eet Support Cruiser", "date": "..."}
```

- `name`: user-confirmed ship type / tier string
- `ml_name`: raw OCR output before user correction (empty when OCR was already correct)
- These entries are filtered out of icon classifier training and processed separately
  to build `ship_type_corrections.json` (see §3 below)

### What is NOT uploaded

- Full screenshots (only the 64×64 icon crop or text region crop)
- Ship name or character name (Ship Name bbox is position-only — never crops, never text)
- The local `.pt` model files
- Anything outside `warp/training_data/`

### Install ID

A random UUID generated at first launch, stored in
`warp/knowledge/install_id.txt`. It is anonymous — not linked to any account.
Its only purpose is democratic voting (1 install = 1 vote per crop hash).

### Deduplication

Each crop is identified by `sha256(crop_bytes)`. If a crop with that hash
already exists in staging for this install_id, it is not uploaded again.
This means the same icon appearing in multiple screenshots is uploaded exactly
once per install.

---

## 3. Central training (HuggingFace staging → trained model)

### Trigger

A GitHub Actions workflow runs `admin_train.py` on a schedule:

```yaml
# .github/workflows/train_central_model.yml  (sets-warp-backend repo)
on:
  schedule:
    - cron: '0 * * * *'   # every hour
  workflow_dispatch:       # manual trigger
```

Training is skipped (fast exit) when:
- No new crops since the last training run (`--skip-if-unchanged`)
- Fewer than 10 new crops arrived (`MIN_NEW_CROPS = 10`)

### Democratic voting

```
For each crop sha256:
    votes = {install_id: label for each user who uploaded that crop}
    winner = majority vote (most common label)
    if tie → first uploader's label wins
```

A single user cannot override the community. If User A labels a crop
"Ablative Shell" and three others label it "Ablative Field Projector",
the community label wins.

### Screen type dataset capping

To prevent dataset bloat for stable UI screens:

```
For each screen type class:
    if n_samples >= 30 and n_samples > 150:
        randomly keep 150 samples
        (avoids storing thousands of near-identical UI screenshots)
```

### Central training process (icon classifier)

File: `sets-warp-backend/admin_train.py` — `train()`

```
1. Download previous icon_classifier.pt from HF warp-knowledge (for fine-tuning)
2. Download all winning crops from staging via snapshot_download per install_id
   (bulk folder download — far fewer HTTP round-trips than per-file)
3. Stratified train/val split
4. Build EfficientNet-B0, replace head for n_classes
5. Load previous backbone weights (classifier.* keys stripped)
   LR = 3e-4 × 0.3 when fine-tuning, 3e-4 from scratch
6. Train with focal loss + cosine annealing + early stopping
7. Save: icon_classifier.pt, label_map.json, icon_classifier_meta.json,
         model_version.json, training_manifest.json
8. Upload all files to sets-sto/warp-knowledge/models/
```

`training_manifest.json` records the set of crop SHAs used in this run —
the next run's skip-if-unchanged check compares against this.

### Central training process (screen classifier)

File: `sets-warp-backend/admin_train.py` — `train_screen_classifier()`

Same flow with MobileNetV3-Small. Only runs if ≥ 7 screen type screenshots
are available. Fine-tunes from previous `screen_classifier.pt` backbone.

### Ship type / tier OCR correction map

File: `sets-warp-backend/admin_train.py` — `collect_text_corrections()`

```
1. Filter staging annotations where slot in {'Ship Type', 'Ship Tier'}
2. For each (ml_name, name) pair where ml_name != '' and ml_name != name:
     votes[ml_name][install_id] = name   ← 1 vote per install_id
3. Democratic vote per ml_name key → majority corrected_name wins
4. Build ship_type_corrections.json:
     {"F1eet Support Cruiser": "Fleet Support Cruiser", ...}
5. Upload to sets-sto/warp-knowledge/models/ship_type_corrections.json
```

Applied in `warp/recognition/text_extractor.py` immediately after OCR reads
the ship type/tier text, before fuzzy ShipDB lookup. Ships with OCR errors
that multiple users have corrected are instantly fixed for all clients.

### model_version.json

Published after each successful central training run:

```json
{
  "version": "<git sha>",
  "trained_at": "2026-03-23T14:00:00Z",
  "n_classes": 2933,
  "val_acc": 0.87,
  "n_samples": 12450,
  "n_users": 18,
  "screen_trained_at": "2026-03-23T14:00:00Z",
  "screen_val_acc": 0.95,
  "screen_n_samples": 340
}
```

---

## 4. Community model delivery (HuggingFace → local)

### On first install (cold-start splash)

The cold-start splash described in
[`SYNC_ARCHITECTURE.md`](SYNC_ARCHITECTURE.md) §3 owns the first-install
model download. Phase `model` of the splash runs the same `ModelUpdater`
that powers the periodic refresh, with `force=True` so the 15 min skip
guard is ignored — the very first launch always pulls every required file:

```
models/icon_classifier.pt          (required)
models/label_map.json              (required)
models/icon_classifier_meta.json
models/model_version.json
models/screen_classifier.pt
models/screen_classifier_labels.json
models/icon_embedder.pt            (optional, ArcFace)
models/embedder_label_map.json     (optional)
models/embedding_index.npz         (optional)
```

After the splash completes, `~/.config/warp/startup_sync_done` is written
and subsequent launches go through the periodic refresh path described
below.

### Ongoing updates (ModelUpdater)

File: `warp/trainer/model_updater.py`

Fired by `SyncCoordinator` as the `model` step of every refresh cycle
(launch + every 60 minutes, see
[`SYNC_ARCHITECTURE.md`](SYNC_ARCHITECTURE.md)). Internally rate-limited.

```
1. Check rate limit: skip if last check was < 15 min ago
   (_CHECK_INTERVAL_HOURS = 0.25 in model_updater.py)
2. GET https://sets-sto-warp-backend.hf.space/model/version
   → returns {available, trained_at, n_classes, val_acc, ...}
3. Compare remote trained_at vs local model_version.json trained_at:
     remote > local  → download and install new model
     remote ≤ local  → skip (local is current)
4. If update needed: download all _MODEL_FILES from HF via hf_hub_download
5. Copy files to warp/models/ atomically
6. Call SETSIconMatcher.reset_ml_session() to reload immediately
7. Save check timestamp to model_version_remote_cache.json
```

**Demotion guard.** The download is only installed if the remote
`trained_at` is **strictly later** than the local one. A tier-down or
class-count regression that would silently downgrade the model is rejected
in the same check — see the `1.0.10` Changelog entry on tier corrections
for the user-visible symptom this prevents.

### screen_classifier fallback

If `screen_classifier.pt` is missing (e.g., first install before bootstrap
completes, or manual deletion), `_ensure_screen_classifier()` downloads it
immediately, bypassing the 15 min rate limit. This runs on every
ModelUpdater check.

---

## 5. Full pipeline diagram

The end-to-end data flow from user confirmation, through HF staging, the
four democratic mergers, central training, and back to every install lives
in its own document: [`DATA_LIFECYCLE.md`](DATA_LIFECYCLE.md).

That document is the canonical reference for the staging-vs-data split,
the Z3 asymmetric thresholds, drain-on-promote, and the audit safety net.
This file (§§1–4 above) covers the ML model story; `DATA_LIFECYCLE.md`
covers the *data* story underneath it.

---

## 6. Priority rules

When WARP matches an icon, the following priority applies:

```
1. Community pHash knowledge override   (warp/knowledge/knowledge_cache.json)
   — exact perceptual hash match, highest confidence, instant
2. Template matching + HSV histogram    (community crop library + cargo icons)
3. ArcFace embedder k-NN                (icon_embedder.pt + embedding_index.npz)
4. Softmax classifier                   (icon_classifier.pt)
5. Session examples                     (crops confirmed during the current
                                         run, used to bridge the gap before
                                         the next central retrain)
```

The classifier and embedder are the *same* files for every install —
`warp/models/icon_classifier.pt`, `warp/models/icon_embedder.pt`. There is
no per-user variant on disk. ModelUpdater replaces them only when the
remote `trained_at` is strictly later than the local one (§4).

---

## 7. Data stored on HuggingFace

| Repo | Path | Contents |
|------|------|----------|
| `sets-sto/sto-icon-dataset` | `staging/<install_id>/crops/` | Icon crop PNGs (64×64) + Ship Type/Tier text crops |
| `sets-sto/sto-icon-dataset` | `staging/<install_id>/annotations.jsonl` | Label records (icon + text; includes `ml_name` for text slots) |
| `sets-sto/sto-icon-dataset` | `staging/<install_id>/screen_types/<TYPE>/` | Screen type PNGs |
| `sets-sto/warp-knowledge` | `models/` | Trained .pt files, label_map.json, model_version.json |
| `sets-sto/warp-knowledge` | `models/ship_type_corrections.json` | OCR correction map: `{raw_ocr: corrected_name}` |
| `sets-sto/warp-knowledge` | `knowledge.json` | pHash → item_name community overrides |
| `sets-sto/warp-knowledge` | `models/training_manifest.json` | SHA set from last training run |

---

## 8. Key files (local)

| File | Purpose |
|------|---------|
| `warp/models/icon_classifier.pt` | Active icon classifier (local or community) |
| `warp/models/screen_classifier.pt` | Active screen type classifier |
| `warp/models/label_map.json` | `{index: item_name}` for icon classifier |
| `warp/models/screen_classifier_labels.json` | `{index: screen_type}` for screen classifier |
| `warp/models/model_version.json` | `trained_at`, `source`, `val_acc` of current model |
| `warp/models/model_version_remote_cache.json` | Timestamp of last remote version check |
| `warp/training_data/annotations.json` | All confirmed annotations |
| `warp/training_data/crops/<sha>.png` | Confirmed icon crops + Ship Type/Tier text region crops |
| `warp/training_data/screen_types/<TYPE>/<file>.png` | Confirmed screen type screenshots |
| `warp/models/ship_type_corrections.json` | OCR correction map downloaded from community (optional) |
| `warp/knowledge/install_id.txt` | Anonymous UUID identifying this installation |

---

## 9. Local bootstrap trainer (one-shot)

The icon embedder (`icon_embedder.pt` + `embedding_index.npz`) is normally
trained centrally on community contributions — the same democratic-voting
pipeline described above for the softmax classifier. When the central
embedder is missing entire class regions (e.g. ground BOFF abilities, whose
icons rarely show up in space-side screenshots), waiting for those crops to
arrive via HF staging is infeasible: 10,600 synthetic crops at
1000 uploads/install/UTC day per `MAX_DAILY_UPLOADS` = ~11 days through
normal sync, plus the wall-clock cost of waiting for those days to pass.

For that one-time gap-closure, sto-warp ships a **local** bootstrap path:

```
1. Generate synthetic crops from cargo wiki PNGs (64×64 BGR, augmented).
2. Train the ArcFace embedder locally on real + synthetic crops.
3. Manually upload the resulting .pt + gallery + label map to central HF.
4. Resume normal central training thereafter.
```

This path is **not** a replacement for central training — it is a single
intervention to seed unknown classes into the gallery so subsequent
community contributions have something to vote against.

### 9.1 Synthetic crop generator

File: `warp/trainer/synthetic_crop_generator.py`

```bash
python -m warp.trainer.synthetic_crop_generator --env ground -n 100
```

For each ability name in `boff_abilities()[env]`, loads the cargo wiki PNG
from `icons_dir()` and emits `n` augmented 64×64 BGR variants under:

```
~/.local/share/warp/training_data/synthetic_crops/<env>/<class_slug>/<seq>.png
```

Augmentations approximate STO's icon-on-UI domain:

| Step | Detail |
|------|--------|
| Background | Random dark BGR gradient + Gaussian noise (mimics navy/grey UI surfaces) |
| Composite | Alpha-aware (icons are BGRA with transparent corners) |
| Scale | Icon height uniform in [44, 60] px before centring |
| Position | ±3 px bbox jitter from canvas centre |
| Colour | HSV: ±15 % brightness, ±20 % saturation, ±5° hue |
| Cooldown | 10 % chance of radial dim sweep overlay |
| Codec | JPEG re-encode at quality 70–95 (gameplay screens are JPEG) |

Synthetic crops live alongside real crops in `training_data/` but are
**never** synced to HF — they are local bootstrap data, not community
contributions.

### 9.2 Local embedder trainer

File: `warp/trainer/embedder_trainer.py`

```bash
# Generate + train in one shot
python -m warp.trainer.embedder_trainer --generate-synthetic --env ground -n 100 --train

# Or train against pre-generated crops
python -m warp.trainer.embedder_trainer --train
```

Architecture mirrors the central trainer (`admin_train_metric.py`):

| Component | Value |
|-----------|-------|
| Backbone | EfficientNet-B0 (ImageNet warm start, then warm-started from existing `icon_embedder.pt` when present) |
| Projection | Linear → 256-d → L2-normalize |
| Head | ArcFace, margin = 0.5, scale = 30.0 |
| Sampler | PK: P = 8 classes × K = 4 samples / batch = 32 |
| Optimizer | AdamW, lr = 3e-4, CosineAnnealingLR |
| Loss | CrossEntropy on ArcFace logits |
| Stop | Early stop on val recall@1, patience = 5, max 30 epochs |

Data sources:
- **Real crops** from `~/.local/share/warp/training_data/crops/` —
  filenames are `<slot>__<slug>__<hash>.png`; canonical labels are pulled
  from `crops/crop_index.json` (key `name`) so the resulting label map
  stays consistent with the existing embedder.
- **Synthetic crops** from `training_data/synthetic_crops/<env>/<slug>/` —
  class slug is reverse-mapped to canonical name via
  `boff_abilities()[env]`.

Outputs in `userdata.models_dir()` (= `~/.cache/warp/models/`):

```
icon_embedder.pt           — backbone + projection state_dict
embedding_index.npz        — full-train gallery (no aug)
embedder_label_map.json    — {index: canonical_name}
icon_embedder_meta.json    — hyper-params + val_recall@1 + source='local-bootstrap'
```

### 9.3 Manual upload to central

After training, copy the four output files to
`sets-sto/warp-knowledge/models/` on HuggingFace (web UI or `huggingface-cli upload`).
The next `ModelUpdater` tick on every install will pull them via the same
flow as in §4. After upload, **central training resumes normal operation**
on community-contributed real crops — no further local intervention.

The upload is a one-time admin action; users do not run this themselves.
