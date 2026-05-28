"""Admin tool: build & curate STO icon equivalence classes.

Some STO items share *identical* icon art (Mk variants, faction reskins,
re-released items, vanity duplicates). Two crops of two different items
that happen to use the same art will keep flipping between names in the
trainer's community-conflict pipeline forever — neither side is wrong,
both sides claim the same pixels.

The fix is an *equivalence class*: a curated list of item names that
share the same icon art. When the trainer sees a community proposal
whose name belongs to the same class as the on-disk name, it silently
keeps the user's choice (no orange conflict row).

This file is **admin-only**: the maintainer runs it locally over the
fetched wiki icon mirror, reviews candidates visually, then pushes the
curated JSON to the HF knowledge repo. End users never touch this tool;
they only consume the resulting `icon_equivalence.json` downloaded by
`sync_client`.

Subcommands
-----------
    python -m warp.tools.icon_equivalence build
        Scan ~/.config/warp/icons/*.png, compute a perceptual hash for
        each, group images whose pHashes are within Hamming distance
        ``--threshold`` (default 0 → identical hash only; the wiki
        crops are clean PNGs so identical art produces identical pHash
        bits, and higher thresholds quickly pick up false positives
        from shared template framing on kit modules etc.), and write
        candidate groups to
        ~/.config/warp/icon_equivalence_candidates.json.

    python -m warp.tools.icon_equivalence review
        PySide6 dialog: walk through candidate groups, check the icons
        that really are the same art, and write the curated result to
        ~/.config/warp/icon_equivalence.json.

    python -m warp.tools.icon_equivalence push
        Upload the curated icon_equivalence.json to the HF knowledge
        repo ``sets-sto/warp-knowledge`` using the HF_TOKEN env var.

pHash
-----
We cannot use ``cv2.img_hash`` because the headless OpenCV wheel
shipped with the project lacks the contrib modules. The standard
32×32-resize → 2D DCT → top-left 8×8 → threshold-by-median construction
takes ~30 lines and yields a 64-bit hash where Hamming distance is a
robust proxy for visual similarity (0 = identical hash, 1–4 = same art,
>10 = different).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from warp.data import cargo
from warp.debug import log
from warp import userdata


CANDIDATES_FILE = 'icon_equivalence_candidates.json'
CURATED_FILE = 'icon_equivalence.json'
# Co-located with the community crops dataset so we have a single HF repo
# to manage. The file lives at the repo root (siblings of `data/crops/`).
HF_REPO = 'sets-sto/sto-icon-dataset'
HF_REPO_TYPE = 'dataset'


# ── pHash ─────────────────────────────────────────────────────────────────

def phash64(img_path: Path) -> int | None:
    """Return a 64-bit perceptual hash for the image, or None on read error.

    Algorithm: resize to 32×32 grayscale → 2D DCT → keep the top-left
    8×8 low-frequency block → threshold each cell against the median of
    the 63 non-DC coefficients → pack 64 bits in row-major order.

    Insensitive to small resizes, JPEG compression and mild colour /
    contrast changes; sensitive to actual shape differences.
    """
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(img.astype(np.float32))
    block = dct[:8, :8].flatten()
    # Median over the 63 AC coefficients (skip the DC term at index 0).
    med = float(np.median(block[1:]))
    bits = 0
    for i, v in enumerate(block):
        if float(v) > med:
            bits |= (1 << i)
    return bits


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


# ── name decoding ─────────────────────────────────────────────────────────

def filename_to_name(p: Path) -> str:
    """Turn ``2399+Starfleet+Phaser+Sniper+Rifle.png`` into the readable
    item name. Wiki crops are stored URL-encoded (``+`` = space,
    ``%22`` = quote, …)."""
    return urllib.parse.unquote_plus(p.stem)


# ── build ─────────────────────────────────────────────────────────────────

def cmd_build(args: argparse.Namespace) -> int:
    icons_dir = cargo.icons_dir()
    if not icons_dir.is_dir():
        log.error(f'icon_equivalence: icons dir not found: {icons_dir}')
        return 1

    paths = sorted(icons_dir.glob('*.png'))
    log.info(f'icon_equivalence: scanning {len(paths)} icons in {icons_dir}')
    if not paths:
        log.error('icon_equivalence: no PNGs to scan')
        return 1

    hashes: list[int] = []
    keep: list[Path] = []
    for p in paths:
        h = phash64(p)
        if h is None:
            log.warning(f'icon_equivalence: skip unreadable {p.name}')
            continue
        hashes.append(h)
        keep.append(p)

    n = len(keep)
    log.info(f'icon_equivalence: hashed {n} icons; grouping with '
             f'Hamming ≤ {args.threshold}')

    arr = np.array(hashes, dtype=np.uint64)

    # Union-find.
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    # O(n²) pairwise — n ≈ 4k is small enough in numpy.
    threshold = int(args.threshold)
    for i in range(n - 1):
        xor = arr[i] ^ arr[i + 1:]
        bytes_view = xor.view(np.uint8).reshape(-1, 8)
        bits = np.unpackbits(bytes_view, axis=1).sum(axis=1)
        close = np.where(bits <= threshold)[0]
        for off in close:
            union(i, i + 1 + int(off))

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    candidates: list[dict] = []
    for members in groups.values():
        if len(members) < 2:
            continue
        members_sorted = sorted(members, key=lambda k: keep[k].name.lower())
        candidates.append({
            'names': [filename_to_name(keep[k]) for k in members_sorted],
            'files': [keep[k].name for k in members_sorted],
            'phashes': [f'{hashes[k]:016x}' for k in members_sorted],
        })

    candidates.sort(key=lambda g: (-len(g['names']), g['names'][0].lower()))

    out_path = userdata.config_dir() / CANDIDATES_FILE
    out_path.write_text(json.dumps({
        'version': 1,
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'threshold': threshold,
        'icons_dir': str(icons_dir),
        'total_icons': n,
        'group_count': len(candidates),
        'groups': candidates,
    }, indent=2, ensure_ascii=False))

    log.info(f'icon_equivalence: {len(candidates)} candidate groups '
             f'(covering {sum(len(g["names"]) for g in candidates)} icons) '
             f'→ {out_path}')
    return 0


# ── review ───────────────────────────────────────────────────────────────

def cmd_review(args: argparse.Namespace) -> int:
    cand_path = userdata.config_dir() / CANDIDATES_FILE
    if not cand_path.is_file():
        log.error(f'icon_equivalence: candidates file missing — run '
                  f'`build` first ({cand_path})')
        return 1

    data = json.loads(cand_path.read_text())
    groups = data.get('groups') or []
    if not groups:
        log.error('icon_equivalence: no candidate groups to review')
        return 1

    icons_dir = Path(data.get('icons_dir') or cargo.icons_dir())

    curated_path = userdata.config_dir() / CURATED_FILE
    existing: list[list[str]] = []
    if curated_path.is_file():
        try:
            existing = json.loads(curated_path.read_text()).get('classes') or []
        except Exception as e:
            log.warning(f'icon_equivalence: could not parse existing '
                        f'{curated_path}: {e}')

    # Run the Qt dialog.
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QPixmap
    from PySide6.QtWidgets import (
        QApplication, QCheckBox, QDialog, QFrame, QGridLayout, QHBoxLayout,
        QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget,
    )

    app = QApplication.instance() or QApplication(sys.argv)

    class ReviewDialog(QDialog):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle('Icon Equivalence — Admin Review')
            self.resize(900, 700)
            self._idx = 0
            # Indices into groups[self._idx]['names'] still available for
            # the current group. None ⇒ first visit, show full group.
            # After "Keep & repeat" we drop already-saved positions so the
            # admin can mine a second sub-class from the same candidate.
            self._remaining: list[int] | None = None
            # Per-icon checked state for the current pass, keyed by the
            # original index inside groups[self._idx]. Lives across the
            # re-renders triggered by checkbox toggles; reset on _next /
            # _prev / sub-pass change.
            self._state: dict[int, bool] = {}
            self._state_key: tuple = ()
            self._kept: list[list[str]] = list(existing)

            self._header = QLabel()
            self._header.setStyleSheet('font-weight: bold; font-size: 13px;')

            self._inner = QWidget()
            self._inner_layout = QVBoxLayout(self._inner)
            self._inner_layout.setContentsMargins(8, 8, 8, 8)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(self._inner)

            btn_keep_next = QPushButton('Keep checked && next')
            btn_keep_next.clicked.connect(self._on_keep_next)
            btn_keep_rep = QPushButton('Keep checked && repeat')
            btn_keep_rep.setToolTip(
                'Save the checked items as a class, then re-show the '
                'unchecked items from this same group so you can extract '
                'a second sub-class.')
            btn_keep_rep.clicked.connect(self._on_keep_repeat)
            btn_skip = QPushButton('Skip')
            btn_skip.clicked.connect(self._next)
            btn_back = QPushButton('← Back')
            btn_back.clicked.connect(self._prev)
            btn_save = QPushButton('Save && Exit')
            btn_save.clicked.connect(self._save_and_close)

            actions = QHBoxLayout()
            actions.addWidget(btn_back)
            actions.addWidget(btn_skip)
            actions.addStretch()
            actions.addWidget(btn_keep_rep)
            actions.addWidget(btn_keep_next)
            actions.addWidget(btn_save)

            root = QVBoxLayout(self)
            root.addWidget(self._header)
            root.addWidget(scroll, 1)
            root.addLayout(actions)

            self._render()

        def _clear_inner(self) -> None:
            while self._inner_layout.count():
                item = self._inner_layout.takeAt(0)
                w = item.widget()
                if w is not None:
                    w.deleteLater()

        def _ensure_state(self, indices: list[int]) -> None:
            """First time we render a (group, sub-pass) combination, mark
            every visible item as checked. Re-renders triggered by a
            checkbox toggle reuse the existing state."""
            key = (self._idx, tuple(indices))
            if self._state_key != key:
                self._state = {i: True for i in indices}
                self._state_key = key

        def _make_cell(self, orig_i: int, name: str, fname: str,
                       *, selected: bool) -> QWidget:
            cell = QWidget()
            cl = QVBoxLayout(cell)
            cl.setContentsMargins(4, 4, 4, 4)

            thumb = QLabel()
            pix = QPixmap(str(icons_dir / fname))
            if not pix.isNull():
                thumb.setPixmap(pix.scaled(
                    128, 128, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            thumb.setAlignment(Qt.AlignCenter)

            cb = QCheckBox(name)
            cb.setChecked(selected)
            cb.setToolTip(fname)
            # Closure over orig_i — toggle updates state and re-renders so
            # the cell jumps to the other panel for side-by-side compare.
            cb.toggled.connect(lambda checked, i=orig_i: self._on_toggle(i, checked))

            cl.addWidget(thumb)
            cl.addWidget(cb)
            return cell

        def _section_header(self, text: str, *, accent: str) -> QLabel:
            lbl = QLabel(text)
            lbl.setStyleSheet(
                f'font-weight: bold; padding: 6px 4px; '
                f'border-bottom: 2px solid {accent}; color: {accent};')
            return lbl

        def _build_grid(self, indices: list[int], g: dict,
                        cols: int = 4) -> QWidget:
            wrap = QWidget()
            grid = QGridLayout(wrap)
            grid.setContentsMargins(0, 4, 0, 8)
            for pos, orig_i in enumerate(indices):
                row, col = divmod(pos, cols)
                cell = self._make_cell(
                    orig_i,
                    g['names'][orig_i],
                    g['files'][orig_i],
                    selected=self._state.get(orig_i, True),
                )
                grid.addWidget(cell, row, col)
            return wrap

        def _on_toggle(self, orig_i: int, checked: bool) -> None:
            self._state[orig_i] = checked
            self._render()

        def _render(self) -> None:
            self._clear_inner()
            if self._idx >= len(groups):
                self._header.setText(
                    f'All {len(groups)} candidate groups reviewed — '
                    f'{len(self._kept)} classes kept. Click Save & Exit.'
                )
                return
            g = groups[self._idx]
            indices = (list(range(len(g['names'])))
                       if self._remaining is None
                       else list(self._remaining))
            self._ensure_state(indices)

            selected = [i for i in indices if self._state.get(i, True)]
            rejected = [i for i in indices if not self._state.get(i, True)]

            sub_note = ('' if self._remaining is None
                        else f'  (sub-pass, {len(indices)} remaining)')
            self._header.setText(
                f'Group {self._idx + 1} / {len(groups)} — '
                f'{len(indices)} icons{sub_note} — '
                f'{len(selected)} selected, {len(rejected)} rejected'
            )

            self._inner_layout.addWidget(self._section_header(
                f'Will be saved as one class  ({len(selected)})',
                accent='#3a8a3a'))
            if selected:
                self._inner_layout.addWidget(self._build_grid(selected, g))
            else:
                empty = QLabel('  (uncheck the rejects below to populate)')
                empty.setStyleSheet('color: #888; padding: 8px 4px;')
                self._inner_layout.addWidget(empty)

            sep = QFrame()
            sep.setFrameShape(QFrame.HLine)
            sep.setFrameShadow(QFrame.Sunken)
            self._inner_layout.addWidget(sep)

            self._inner_layout.addWidget(self._section_header(
                f'Not in this class  ({len(rejected)})',
                accent='#a04040'))
            if rejected:
                self._inner_layout.addWidget(self._build_grid(rejected, g))
            else:
                empty = QLabel('  (uncheck items above to move them here)')
                empty.setStyleSheet('color: #888; padding: 8px 4px;')
                self._inner_layout.addWidget(empty)

            self._inner_layout.addStretch(1)

        def _collect_checked(self) -> list[int]:
            """Save currently-checked items as a class (if ≥2) and return
            their original indices into ``groups[self._idx]``."""
            if self._idx >= len(groups):
                return []
            g = groups[self._idx]
            visible = (list(range(len(g['names'])))
                       if self._remaining is None
                       else list(self._remaining))
            chosen_orig = [i for i in visible if self._state.get(i, True)]
            if len(chosen_orig) < 2:
                log.info('icon_equivalence: <2 items checked — nothing saved')
                return chosen_orig
            chosen_names = [g['names'][i] for i in chosen_orig]
            key = tuple(sorted(chosen_names))
            if not any(tuple(sorted(c)) == key for c in self._kept):
                self._kept.append(chosen_names)
                log.info(f'icon_equivalence: kept class {chosen_names}')
            else:
                log.info('icon_equivalence: skipped duplicate of '
                         'existing class')
            return chosen_orig

        def _on_keep_next(self) -> None:
            self._collect_checked()
            self._next()

        def _on_keep_repeat(self) -> None:
            chosen = self._collect_checked()
            if len(chosen) < 2:
                # Nothing was actually saved — pointless to repeat,
                # behave like Skip.
                self._next()
                return
            currently_shown = (list(self._remaining)
                               if self._remaining is not None
                               else list(range(len(groups[self._idx]['names']))))
            chosen_set = set(chosen)
            leftover = [i for i in currently_shown if i not in chosen_set]
            if len(leftover) < 2:
                # No way to form another class from what's left.
                self._next()
                return
            self._remaining = leftover
            self._render()

        def _next(self) -> None:
            self._idx += 1
            self._remaining = None
            self._render()

        def _prev(self) -> None:
            if self._idx > 0:
                self._idx -= 1
                self._remaining = None
                self._render()

        def _save_and_close(self) -> None:
            payload = {
                'version': 1,
                'generated_at': datetime.now(timezone.utc).isoformat(),
                'source_threshold': data.get('threshold'),
                'class_count': len(self._kept),
                'classes': [sorted(c) for c in self._kept],
            }
            curated_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False))
            log.info(f'icon_equivalence: wrote {len(self._kept)} curated '
                     f'classes → {curated_path}')
            self.accept()

    dlg = ReviewDialog()
    dlg.exec()
    return 0


# ── push ──────────────────────────────────────────────────────────────────

def cmd_push(args: argparse.Namespace) -> int:
    curated_path = userdata.config_dir() / CURATED_FILE
    if not curated_path.is_file():
        log.error(f'icon_equivalence: curated file missing — run '
                  f'`review` first ({curated_path})')
        return 1

    token = os.environ.get('HF_TOKEN') or os.environ.get(
        'HUGGINGFACE_TOKEN')
    if not token:
        log.error('icon_equivalence: HF_TOKEN env var not set')
        return 1

    try:
        from huggingface_hub import HfApi
    except Exception as e:
        log.error(f'icon_equivalence: huggingface_hub import failed: {e}')
        return 1

    api = HfApi(token=token)
    commit = api.upload_file(
        path_or_fileobj=str(curated_path),
        path_in_repo=CURATED_FILE,
        repo_id=HF_REPO,
        repo_type=HF_REPO_TYPE,
        commit_message=f'icon_equivalence: refresh from admin '
                        f'({datetime.now(timezone.utc).isoformat()})',
    )
    log.info(f'icon_equivalence: pushed {curated_path.name} to '
             f'{HF_REPO} ({HF_REPO_TYPE}) — {commit}')
    return 0


# ── CLI ───────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description='Admin tool to build/review/publish STO icon '
                    'equivalence classes.')
    sub = parser.add_subparsers(dest='cmd', required=True)

    p_build = sub.add_parser('build', help='Scan icons and group by pHash.')
    p_build.add_argument('--threshold', type=int, default=0,
                         help='Max Hamming distance to consider "same art" '
                              '(default 0 = identical pHash only; 2–4 catches '
                              'near-duplicates but pulls in false positives '
                              'from shared template framing).')
    p_build.set_defaults(func=cmd_build)

    p_review = sub.add_parser('review',
                              help='Open the PySide6 review dialog.')
    p_review.set_defaults(func=cmd_review)

    p_push = sub.add_parser('push',
                            help='Upload curated JSON to HF (needs HF_TOKEN).')
    p_push.set_defaults(func=cmd_push)

    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == '__main__':
    sys.exit(main())
