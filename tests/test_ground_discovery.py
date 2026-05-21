"""Side test: discover ground content on TRAITS/BOFFS screens.

We currently treat every TRAITS / BOFFS screen as space-only. The same
screen layouts also host ground content (Personal Ground Traits, Active
Ground Reputation, Away Team BOFF panel). Before wiring a "discover both
environments" mode into the main pipeline, this script measures the
upside on the local corpus:

    TRAITS  → run trait_grid.detect_traits with build_type='SPACE_TRAITS'
              (current behaviour) vs build_type=None (no environment
              filter). Report extra ground slots that surface and verify
              the space baseline does not shrink.

    BOFFS   → run boff_marker.detect_panel once (current behaviour). Then
              mask out the matched panel region and run again to see
              whether a second profession-marker panel (the away-team
              one) shows up.

No production code is modified. Run standalone:

    python tests/test_ground_discovery.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from warp.data import cargo
from warp.recognition import boff_marker as _bm
from warp.recognition import trait_grid as _tg
from warp.recognition.icon_matcher import SETSIconMatcher


CORPUS = Path.home() / '.local/share/warp/training_data/screen_types'
TRAITS_DIR = CORPUS / 'TRAITS'
BOFFS_DIR  = CORPUS / 'BOFFS'

GROUND_TRAIT_SLOTS = {
    'Personal Ground Traits', 'Ground Reputation', 'Active Ground Rep',
}
SPACE_TRAIT_SLOTS = {
    'Personal Space Traits', 'Starship Traits', 'Space Reputation',
    'Active Space Rep',
}


def _list_images(d: Path) -> list[Path]:
    if not d.exists():
        return []
    return sorted(p for p in d.iterdir()
                  if p.suffix.lower() in ('.png', '.jpg', '.jpeg'))


def _bbox_count(d: dict) -> int:
    return sum(len(v) for v in d.values())


def _panel_bbox(res: dict) -> tuple[int, int, int, int] | None:
    """Tight enclosing rect over all seat markers + their slot projections."""
    if not res:
        return None
    xs0, ys0, xs1, ys1 = [], [], [], []
    for (mx, my, mw, mh, *_) in res.get('col_a', []) + res.get('col_b', []):
        xs0.append(mx); ys0.append(my)
        xs1.append(mx + mw); ys1.append(my + mh)
    for (_seat, _slot, x, y, w, h, _c) in res.get('slots', []):
        xs0.append(x); ys0.append(y)
        xs1.append(x + w); ys1.append(y + h)
    if not xs0:
        return None
    return (min(xs0), min(ys0), max(xs1), max(ys1))


# ── Traits ─────────────────────────────────────────────────────────────

def run_traits(matcher, app_cache) -> None:
    files = _list_images(TRAITS_DIR)
    print(f'\n=== TRAITS corpus: {len(files)} images ===\n')

    space_only_loss   = 0
    extra_ground_imgs = 0
    extra_ground_bxs  = 0
    per_image_extra: list[tuple[str, dict, dict]] = []

    for p in files:
        img = cv2.imread(str(p))
        if img is None:
            print(f'  {p.name:48s}  <unreadable>')
            continue

        base = _tg.detect_traits(img, matcher, app_cache,
                                 build_type='SPACE_TRAITS')
        free = _tg.detect_traits(img, matcher, app_cache,
                                 build_type=None)

        base_space = {k: v for k, v in base.items() if k in SPACE_TRAIT_SLOTS}
        free_space = {k: v for k, v in free.items() if k in SPACE_TRAIT_SLOTS}
        free_ground = {k: v for k, v in free.items() if k in GROUND_TRAIT_SLOTS}
        free_other  = {k: v for k, v in free.items()
                       if k not in SPACE_TRAIT_SLOTS
                       and k not in GROUND_TRAIT_SLOTS}

        # Regression check: free mode must keep what space-filter mode found.
        regressed = False
        for slot, boxes in base_space.items():
            if len(free_space.get(slot, [])) < len(boxes):
                regressed = True
                break
        if regressed:
            space_only_loss += 1

        if free_ground:
            extra_ground_imgs += 1
            extra_ground_bxs += _bbox_count(free_ground)
            per_image_extra.append((p.name, free_ground, free_other))

        flag = ' REGRESSION' if regressed else ''
        print(f'  {p.name[:48]:48s}  '
              f'space={_bbox_count(base_space):2d}'
              f' (free_space={_bbox_count(free_space):2d})'
              f'  ground={_bbox_count(free_ground):2d}'
              f'  other={_bbox_count(free_other):2d}'
              f'{flag}')

    print(f'\n  images with ground content discovered : '
          f'{extra_ground_imgs}/{len(files)}')
    print(f'  extra ground bboxes (total)           : {extra_ground_bxs}')
    print(f'  images with space-baseline regression : {space_only_loss}')
    if per_image_extra:
        print('\n  ground sections per image:')
        for name, gnd, oth in per_image_extra:
            parts = [f'{k}={len(v)}' for k, v in gnd.items()]
            if oth:
                parts += [f'?{k}={len(v)}' for k, v in oth.items()]
            print(f'    {name[:48]:48s}  {", ".join(parts)}')


# ── Boffs ──────────────────────────────────────────────────────────────

def run_boffs() -> None:
    files = _list_images(BOFFS_DIR)
    print(f'\n=== BOFFS corpus: {len(files)} images ===\n')

    second_panel_imgs = 0
    no_panel_imgs     = 0

    for p in files:
        img = cv2.imread(str(p))
        if img is None:
            print(f'  {p.name:48s}  <unreadable>')
            continue

        try:
            res1 = _bm.detect_panel(img)
        except Exception as e:
            print(f'  {p.name[:48]:48s}  detect_panel failed: {e!r}')
            continue

        if not res1:
            no_panel_imgs += 1
            print(f'  {p.name[:48]:48s}  no panel anchored')
            continue

        n1 = len(res1['col_a']) + len(res1['col_b'])
        bbox1 = _panel_bbox(res1)

        # Mask out the matched panel with a generous pad, then re-run.
        img2 = img.copy()
        if bbox1 is not None:
            x0, y0, x1, y1 = bbox1
            pad_y = int((y1 - y0) * 0.25)
            pad_x = int((x1 - x0) * 0.08)
            H, W = img.shape[:2]
            x0 = max(0, x0 - pad_x); y0 = max(0, y0 - pad_y)
            x1 = min(W, x1 + pad_x); y1 = min(H, y1 + pad_y)
            img2[y0:y1, x0:x1] = 0

        try:
            res2 = _bm.detect_panel(img2)
        except Exception as e:
            res2 = None
            print(f'  {p.name[:48]:48s}  pass-2 raised: {e!r}')

        if res2:
            n2 = len(res2['col_a']) + len(res2['col_b'])
            second_panel_imgs += 1
            bbox2 = _panel_bbox(res2)
            same = bbox2 is not None and bbox1 is not None and (
                abs(bbox2[0] - bbox1[0]) < 20 and abs(bbox2[1] - bbox1[1]) < 20
            )
            flag = '  (DUP? overlaps first)' if same else ''
            print(f'  {p.name[:48]:48s}  panel1 seats={n1:2d}  '
                  f'panel2 seats={n2:2d} score={res2["score"]:.2f}{flag}')
        else:
            print(f'  {p.name[:48]:48s}  panel1 seats={n1:2d}  '
                  f'panel2 = none')

    print(f'\n  images with a 2nd panel detected      : '
          f'{second_panel_imgs}/{len(files)}')
    print(f'  images where no panel anchors at all  : {no_panel_imgs}')


def main() -> None:
    if not CORPUS.exists():
        print(f'corpus not found: {CORPUS}')
        sys.exit(1)

    # Cache + matcher needed only by trait_grid (boff_marker is pure-cv).
    app_cache = cargo.cache_view()
    matcher = SETSIconMatcher(app_view := cargo.app_view())  # noqa: F841

    run_traits(matcher, app_cache)
    run_boffs()


if __name__ == '__main__':
    main()
