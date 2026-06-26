"""Scrub poisoned crops from the local WARP training data.

Two detection heuristics are applied to every confirmed annotation in
``~/.local/share/warp/training_data/annotations.json``:

1. **Visual sanity for virtual labels** (``__empty__`` / ``__inactive__``):
   real empty / inactive slots in STO are dim, low-saturation gradients.
   A crop labeled virtual but containing a colourful icon (high HSV
   saturation **or** high mean brightness) is almost certainly mis-
   labeled by auto-accept on a low-confidence detection that the user
   never corrected. Such crops poison the session-example pool: they
   match the same screenshot pixel-perfectly on every re-run and beat
   the embedder forever.

2. **Pixel-identical conflicts**: the same crop content (sha256 of the
   PNG bytes) appears in two or more annotations with different labels.
   At most one can be correct — flag all but the most-frequent label.

The script is a dry-run by default. Pass ``--apply`` to actually delete
the suspect crop PNGs and remove their entries from ``annotations.json``
and ``crop_index.json``.

Usage::

    python -m warp.tools.scrub_training_data                 # report only
    python -m warp.tools.scrub_training_data --apply         # actually delete
    python -m warp.tools.scrub_training_data --training-dir /custom/path
    python -m warp.tools.scrub_training_data --sat-max 60    # stricter virtual check
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


DEFAULT_TRAINING_DIR = Path.home() / '.local/share/warp/training_data'
VIRTUAL_LABELS = frozenset({'__empty__', '__inactive__'})

# Bright-pixel ratios — tuned on the known poison cases (XII rarity
# tactical consoles + a Kentari missile launcher mislabeled empty) vs.
# 533 real __empty__/__inactive__ crops from the local pool:
#
#   metric                    POISONS          REAL VIRTUALS (p90 of 533)
#   bright_ratio (V>150)      14.6% – 24.9%    2.7%
#   rich_ratio   (S>100&V>100) 11.2% – 30.0%   6.8%
#
# Real STO virtuals are uniformly dim — even an __inactive__ padlock
# fits in a few % of bright pixels. A virtual-labeled crop with both
# >7% bright AND >7% colour-rich pixels is a colourful real icon.
DEFAULT_BRIGHT_RATIO = 0.07   # fraction of pixels with V > 150
DEFAULT_RICH_RATIO   = 0.07   # fraction of pixels with S > 100 AND V > 100


def _sha_of_file(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def _looks_real_for_virtual_label(crop_bgr: np.ndarray,
                                  bright_ratio_min: float,
                                  rich_ratio_min: float) -> tuple[bool, str]:
    """If a crop is labeled __empty__/__inactive__, decide whether its
    pixels say otherwise. Returns (is_suspect, reason).

    A real virtual is uniformly dim. Suspect when the crop has both a
    meaningful fraction of bright pixels (V > 150) AND saturated
    highlights (S > 100 & V > 100) — that pattern can only come from a
    real, colourful icon mis-labeled as empty/inactive.
    """
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    bright_ratio = float((v > 150).mean())
    rich_ratio   = float(((s > 100) & (v > 100)).mean())
    is_suspect = (bright_ratio > bright_ratio_min
                  and rich_ratio > rich_ratio_min)
    reason = f'bright={bright_ratio:.1%} rich={rich_ratio:.1%}'
    return is_suspect, reason


def _load_annotations(training_dir: Path) -> dict:
    ann_path = training_dir / 'annotations.json'
    if not ann_path.exists():
        raise FileNotFoundError(f'annotations.json not found at {ann_path}')
    return json.loads(ann_path.read_text(encoding='utf-8'))


def _iter_anns(data: dict):
    """Yield (screenshot_label, anns_list, ann) for every annotation.

    Tolerates both schemas:
      - new (post-196e035): ``{sha16: {'filename': ..., 'annotations': [ann, ...]}}``
      - legacy:             ``{filename: [ann, ...]}``

    ``anns_list`` is the mutable list the ann lives in, so callers can
    drop entries in place during ``apply_removals``.
    """
    for key, val in data.items():
        if isinstance(val, dict):
            anns_list = val.get('annotations', [])
            label = val.get('filename', key)
        elif isinstance(val, list):
            anns_list = val
            label = key
        else:
            continue
        for ann in anns_list:
            yield label, anns_list, ann


def _save_annotations(training_dir: Path, data: dict) -> None:
    ann_path = training_dir / 'annotations.json'
    ann_path.write_text(json.dumps(data, indent=2), encoding='utf-8')


def _load_crop_index(training_dir: Path) -> dict:
    idx_path = training_dir / 'crop_index.json'
    if not idx_path.exists():
        return {}
    return json.loads(idx_path.read_text(encoding='utf-8'))


def _save_crop_index(training_dir: Path, idx: dict) -> None:
    idx_path = training_dir / 'crop_index.json'
    idx_path.write_text(json.dumps(idx, indent=2), encoding='utf-8')


def _crop_path_for_ann(training_dir: Path, ann: dict) -> Path | None:
    """Resolve the crop PNG path for one annotation dict.

    Primary source is the explicit ``crop_name`` field; fallback is the
    legacy ``<slot>__<name>__<ann_id>.png`` convention so older
    annotations are still scrubbable.
    """
    crop_name = ann.get('crop_name', '')
    if crop_name:
        p = training_dir / crop_name
        if p.exists():
            return p
    ann_id = ann.get('ann_id', '')
    if not ann_id:
        return None
    slot = ann.get('slot', '').replace(' ', '_').lower()
    name = ann.get('name', '').replace(' ', '_').lower()[:40]
    fname = f'{slot}__{name}__{ann_id}.png'
    p = training_dir / 'crops' / fname
    return p if p.exists() else None


def find_virtual_label_poison(training_dir: Path,
                              data: dict,
                              bright_ratio_min: float,
                              rich_ratio_min: float) -> list[dict]:
    """Find confirmed annotations whose label is __empty__/__inactive__
    but whose crop looks like a real, colourful icon."""
    suspects = []
    for screenshot, _anns_list, ann in _iter_anns(data):
        if ann.get('state') != 'confirmed':
            continue
        name = (ann.get('name') or '').strip()
        if name not in VIRTUAL_LABELS:
            continue
        # Skip entries the user already inspected and confirmed OK.
        if ann.get('poison_reviewed'):
            continue
        crop_path = _crop_path_for_ann(training_dir, ann)
        if crop_path is None:
            continue
        img = cv2.imread(str(crop_path))
        if img is None:
            continue
        is_suspect, reason = _looks_real_for_virtual_label(
            img, bright_ratio_min, rich_ratio_min)
        if not is_suspect:
            continue
        suspects.append({
            'reason':     f'virtual-label/visual: {reason}',
            'screenshot': screenshot,
            'ann_id':     ann.get('ann_id', ''),
            'slot':       ann.get('slot', ''),
            'label':      name,
            'crop_path':  crop_path,
            'ann':        ann,
        })
    return suspects


def find_pixel_conflicts(training_dir: Path, data: dict) -> list[dict]:
    """Find groups of pixel-identical crops with different labels.
    All but the majority label are flagged as suspects."""
    by_hash: dict[str, list[dict]] = defaultdict(list)
    for screenshot, _anns_list, ann in _iter_anns(data):
        if ann.get('state') != 'confirmed':
            continue
        crop_path = _crop_path_for_ann(training_dir, ann)
        if crop_path is None:
            continue
        sha = _sha_of_file(crop_path)
        by_hash[sha].append({
            'screenshot': screenshot,
            'ann':        ann,
            'crop_path':  crop_path,
        })

    suspects = []
    for sha, entries in by_hash.items():
        labels = {e['ann'].get('name', '') for e in entries}
        if len(labels) < 2:
            continue
        # Pick the majority label as the "kept" one; flag the rest.
        counts: dict[str, int] = defaultdict(int)
        for e in entries:
            counts[e['ann'].get('name', '')] += 1
        majority = max(counts.items(), key=lambda kv: kv[1])[0]
        for e in entries:
            lbl = e['ann'].get('name', '')
            if lbl == majority:
                continue
            suspects.append({
                'reason':     f'pixel-conflict: {sha[:10]} also labeled {majority!r} '
                              f'({counts[majority]}× vs {counts[lbl]}×)',
                'screenshot': e['screenshot'],
                'ann_id':     e['ann'].get('ann_id', ''),
                'slot':       e['ann'].get('slot', ''),
                'label':      lbl,
                'crop_path':  e['crop_path'],
                'ann':        e['ann'],
            })
    return suspects


def review_suspects(suspects: list[dict],
                    training_dir: Path | None = None,
                    data: dict | None = None) -> list[dict]:
    """Interactive per-crop review using PySide6 (pipx venv ships headless
    OpenCV, so cv2.imshow is unavailable). Shows each crop scaled 6× with
    label + reason; user presses D=delete, K=keep, Q/Esc=quit. Returns the
    subset marked for deletion.

    When `training_dir` and `data` are provided, kept (K) entries are
    persistently marked with `poison_reviewed=True` in annotations.json
    so the runtime poison guard and future scrub runs ignore them."""
    if not suspects:
        return []
    kept_ann_ids: list[str] = []

    from PySide6.QtCore import Qt
    from PySide6.QtGui import QImage, QPixmap, QKeyEvent
    from PySide6.QtWidgets import QApplication, QDialog, QLabel, QVBoxLayout

    app = QApplication.instance() or QApplication([])

    class ReviewDialog(QDialog):
        def __init__(self):
            super().__init__()
            self.setWindowTitle('WARP scrub — D=DELETE  K=keep  Q=quit')
            self.decision: str | None = None
            self.info_label = QLabel()
            self.info_label.setStyleSheet(
                'color: white; background: #222; padding: 8px; font-family: monospace;')
            self.info_label.setTextFormat(Qt.TextFormat.RichText)
            self.img_label = QLabel()
            self.img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.img_label.setStyleSheet('background: #111;')
            lay = QVBoxLayout(self)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(0)
            lay.addWidget(self.info_label)
            lay.addWidget(self.img_label, 1)

        def set_crop(self, bgr: np.ndarray, info_html: str) -> None:
            self.info_label.setText(info_html)
            h, w = bgr.shape[:2]
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            big = cv2.resize(rgb, (w * 6, h * 6), interpolation=cv2.INTER_NEAREST)
            bh, bw = big.shape[:2]
            qimg = QImage(big.data, bw, bh, bw * 3, QImage.Format.Format_RGB888).copy()
            self.img_label.setPixmap(QPixmap.fromImage(qimg))
            self.resize(max(bw, 600), bh + 100)

        def keyPressEvent(self, ev: QKeyEvent) -> None:
            k = ev.key()
            if k == Qt.Key.Key_D:
                self.decision = 'd'; self.accept()
            elif k == Qt.Key.Key_K:
                self.decision = 'k'; self.accept()
            elif k in (Qt.Key.Key_Q, Qt.Key.Key_Escape):
                self.decision = 'q'; self.accept()
            else:
                super().keyPressEvent(ev)

    dlg = ReviewDialog()
    to_delete: list[dict] = []
    for i, s in enumerate(suspects, 1):
        cp = s['crop_path']
        img = cv2.imread(str(cp))
        if img is None:
            print(f'[{i}/{len(suspects)}] {cp.name} — cannot read, skipping')
            continue
        info_html = (
            f'<b>[{i}/{len(suspects)}]</b> label=<b>{s["label"]}</b> '
            f'slot=<b>{s["slot"]}</b><br>'
            f'<span style="color:#bbb">{s["reason"]}</span><br>'
            f'<span style="color:#7f7">D = DELETE</span> &nbsp; '
            f'<span style="color:#fff">K = keep</span> &nbsp; '
            f'<span style="color:#f88">Q / Esc = quit</span>'
        )
        dlg.set_crop(img, info_html)
        print(f'[{i}/{len(suspects)}] {cp.name}')
        print(f'  label={s["label"]!r}  slot={s["slot"]!r}')
        print(f'  reason:     {s["reason"]}')
        print(f'  screenshot: {s["screenshot"]}')
        dlg.decision = None
        dlg.exec()
        d = dlg.decision
        if d == 'd':
            to_delete.append(s); print('  → marked for DELETE\n')
        elif d == 'k':
            if s.get('ann_id'):
                kept_ann_ids.append(s['ann_id'])
            print('  → keep (marked poison_reviewed)\n')
        else:
            print('  → quit review (decisions so far retained)\n')
            break

    if kept_ann_ids and training_dir is not None and data is not None:
        kept_set = set(kept_ann_ids)
        touched = 0
        for _screenshot, _anns_list, ann in _iter_anns(data):
            if ann.get('ann_id') in kept_set and not ann.get('poison_reviewed'):
                ann['poison_reviewed'] = True
                touched += 1
        if touched:
            _save_annotations(training_dir, data)
            print(f'Persisted poison_reviewed=True on {touched} annotation(s).')

    return to_delete


def apply_removals(training_dir: Path,
                   data: dict,
                   suspects: list[dict]) -> int:
    """Delete crop PNGs and remove annotation entries. Returns count."""
    crop_index = _load_crop_index(training_dir)
    to_remove_ids = {s['ann_id'] for s in suspects if s['ann_id']}
    removed = 0
    for s in suspects:
        cp = s['crop_path']
        if cp and cp.exists():
            cp.unlink()
            removed += 1
        # Drop from crop_index by filename match (key contains ann_id)
        ann_id = s['ann_id']
        if ann_id:
            stale = [k for k in crop_index if ann_id in k]
            for k in stale:
                del crop_index[k]
    # Drop annotation entries (handles both schemas)
    for key, val in list(data.items()):
        if isinstance(val, dict):
            val['annotations'] = [a for a in val.get('annotations', [])
                                  if a.get('ann_id') not in to_remove_ids]
        elif isinstance(val, list):
            data[key] = [a for a in val
                         if a.get('ann_id') not in to_remove_ids]
    _save_annotations(training_dir, data)
    _save_crop_index(training_dir, crop_index)
    return removed


def _print_suspects(title: str, suspects: list[dict]) -> None:
    print(f'\n── {title} — {len(suspects)} suspect(s) ──')
    if not suspects:
        return
    for s in suspects:
        print(f'  [{s["slot"]}] label={s["label"]!r}')
        print(f'    screenshot: {s["screenshot"]}')
        print(f'    crop:       {s["crop_path"].name}')
        print(f'    reason:     {s["reason"]}')


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description='Scrub poisoned crops from WARP training data.')
    p.add_argument('--training-dir', type=Path, default=DEFAULT_TRAINING_DIR,
                   help=f'path to training_data (default: {DEFAULT_TRAINING_DIR})')
    p.add_argument('--apply', action='store_true',
                   help='actually delete suspect crops + annotations (default: dry-run)')
    p.add_argument('--review', action='store_true',
                   help='show each suspect crop (6× scaled) and ask d/k per crop. '
                        'After review, prompts once before deleting.')
    p.add_argument('--bright-ratio', type=float, default=DEFAULT_BRIGHT_RATIO,
                   help=f'min fraction of bright pixels (V>150) to flag a '
                        f'virtual label (default: {DEFAULT_BRIGHT_RATIO})')
    p.add_argument('--rich-ratio', type=float, default=DEFAULT_RICH_RATIO,
                   help=f'min fraction of colour-rich pixels (S>100&V>100) '
                        f'to flag a virtual label (default: {DEFAULT_RICH_RATIO})')
    p.add_argument('--skip-virtual', action='store_true',
                   help='skip the virtual-label visual sanity check')
    p.add_argument('--skip-conflicts', action='store_true',
                   help='skip the pixel-identical conflict check')
    args = p.parse_args(argv)

    training_dir = args.training_dir
    if not training_dir.exists():
        print(f'ERROR: {training_dir} does not exist'); return 2
    data = _load_annotations(training_dir)

    suspects: list[dict] = []
    if not args.skip_virtual:
        s1 = find_virtual_label_poison(
            training_dir, data, args.bright_ratio, args.rich_ratio)
        _print_suspects('Virtual-label visual sanity', s1)
        suspects += s1
    if not args.skip_conflicts:
        s2 = find_pixel_conflicts(training_dir, data)
        _print_suspects('Pixel-identical conflicts', s2)
        suspects += s2

    # De-duplicate by ann_id (same entry can hit both heuristics)
    seen = set()
    unique = []
    for s in suspects:
        key = s['ann_id'] or (str(s['crop_path']), s['label'])
        if key in seen:
            continue
        seen.add(key)
        unique.append(s)
    suspects = unique

    print(f'\nTotal unique suspects: {len(suspects)}')
    if not suspects:
        return 0

    if args.review:
        print('\nEntering interactive review — d=DELETE  k=keep  q=quit')
        suspects = review_suspects(suspects, training_dir=training_dir, data=data)
        print(f'\nReview done — {len(suspects)} crop(s) marked for deletion.')
        if not suspects:
            return 0
        ans = input(f'Delete {len(suspects)} crop(s) now? [y/N] ').strip().lower()
        if ans != 'y':
            print('Aborted — no changes made.')
            return 0
    elif not args.apply:
        print('Dry-run — re-run with --apply (or --review) to delete.')
        return 0

    removed = apply_removals(training_dir, data, suspects)
    print(f'Removed {removed} crop file(s); annotations.json + crop_index.json updated.')
    print('Restart WARP CORE so the session pool reloads without the poison.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
