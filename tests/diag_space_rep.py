"""Diagnostic for missing Space Reputation row in 20250719145506_1.jpg.

Reproduces trait_grid's CC pipeline against the image and reports:
  - all CCs that survive _detect_icon_ccs
  - which y-band the Space Reputation row sits in
  - what _find_trait_rows produces
  - what _lock_grids_multi clusters into panels
  - whether the rep row falls inside any panel's resweep band
  - which filter (w/h/AR/threshold) rejects rep icons, if any
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from warp.recognition.trait_grid import (
    _detect_icon_ccs, _find_trait_rows, _cluster_row_groups,
    _lock_grids_multi, _resweep_rows_in_band, _cluster_resweep_groups,
    ICON_H_FRAC_LO, ICON_H_FRAC_HI, ICON_AR_LO, ICON_AR_HI,
)

IMG_PATH = Path('/home/raman/STO_screens/screeny/20250719145506_1.jpg')
OUT = Path(__file__).parent / '_diag_out'
OUT.mkdir(exist_ok=True)


def main():
    img = cv2.imread(str(IMG_PATH))
    if img is None:
        print(f'FAIL: cannot read {IMG_PATH}')
        return 1
    H, W = img.shape[:2]
    print(f'image: {W}x{H}')

    # --- raw CC sweep (no filtering) ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    raw = [(int(stats[i, 0]), int(stats[i, 1]),
            int(stats[i, 2]), int(stats[i, 3])) for i in range(1, n)]
    print(f'raw CCs: {len(raw)} (after threshold=30)')

    # --- filtered CCs (what trait_grid actually uses) ---
    ccs = _detect_icon_ccs(img)
    print(f'filtered CCs (icon-sized): {len(ccs)}')

    # --- per-row Y bins (rough manual segmentation) ---
    # Print histogram of CC center-Y to see where rows live
    cys = sorted([y + h / 2 for _, y, _, h in ccs])
    print(f'  CC center-Y range: {cys[0]:.0f} .. {cys[-1]:.0f}')

    # --- find_trait_rows + cluster_row_groups + lock_grids_multi ---
    rows = _find_trait_rows(ccs)
    print(f'\n_find_trait_rows: {len(rows)} chains')
    for i, r in enumerate(rows):
        xs = sorted(b[0] + b[2] / 2 for b in r)
        ys = [b[1] + b[3] / 2 for b in r]
        ws = [b[2] for b in r]
        hs = [b[3] for b in r]
        print(f'  row {i}: n={len(r)}  cy={np.median(ys):.0f}  '
              f'x=[{xs[0]:.0f}..{xs[-1]:.0f}]  '
              f'iw~{int(np.median(ws))}  ih~{int(np.median(hs))}')

    groups = _cluster_row_groups(rows)
    print(f'\n_cluster_row_groups: {len(groups)} groups')
    for i, g in enumerate(groups):
        ys = [b[1] + b[3] / 2 for r in g for b in r]
        print(f'  group {i}: {len(g)} rows, cy={int(min(ys))}..{int(max(ys))}')

    panels = _lock_grids_multi(groups, max_panels=4)
    print(f'\n_lock_grids_multi: {len(panels)} panels')
    for i, p in enumerate(panels):
        print(f'  panel {i}: cols={[round(c,1) for c in p["cols"]]} '
              f'iw={p["icon_w"]:.1f} ih={p["icon_h"]:.1f} '
              f'y={p["y_top"]}..{p["y_bot"]}')

    # --- Where do we EXPECT Space Reputation to be? ---
    # Look at the image bottom 35%, find bright runs along Y.
    print('\n=== Hunt for Space Reputation row ===')
    bottom_y = int(H * 0.65)
    print(f'Searching y ≥ {bottom_y} for rep-row candidates')
    bottom_ccs = [(x, y, w, h) for x, y, w, h in raw if y >= bottom_y]
    print(f'  raw CCs in bottom band: {len(bottom_ccs)}')
    # Classify why each was filtered
    h_lo = int(H * ICON_H_FRAC_LO)
    h_hi_frac = 0.65 if H < 250 else ICON_H_FRAC_HI
    h_hi = int(H * h_hi_frac)
    print(f'  filter window: h in [{h_lo}, {h_hi}], AR in [{ICON_AR_LO}, {ICON_AR_HI}+4px]')

    survivors_in_band = []
    rejected = {'h_too_small': 0, 'h_too_large': 0, 'ar_bad': 0}
    for x, y, w, h in bottom_ccs:
        if h < h_lo:
            rejected['h_too_small'] += 1; continue
        if h > h_hi:
            rejected['h_too_large'] += 1; continue
        if w < int(h * ICON_AR_LO) or w > int(h * ICON_AR_HI) + 4:
            rejected['ar_bad'] += 1; continue
        survivors_in_band.append((x, y, w, h))
    print(f'  rejected in band: {rejected}')
    print(f'  survivors in band: {len(survivors_in_band)}')
    # Cluster survivors by Y to find rep row candidates
    if survivors_in_band:
        by_y = sorted(survivors_in_band, key=lambda b: b[1] + b[3] / 2)
        rows_in_band = [[by_y[0]]]
        for b in by_y[1:]:
            prev_cy = np.median([c[1] + c[3] / 2 for c in rows_in_band[-1]])
            if abs((b[1] + b[3] / 2) - prev_cy) <= 12:
                rows_in_band[-1].append(b)
            else:
                rows_in_band.append([b])
        for i, r in enumerate(rows_in_band):
            xs = sorted(b[0] + b[2] / 2 for b in r)
            ys = [b[1] + b[3] / 2 for b in r]
            ws = [b[2] for b in r]
            hs = [b[3] for b in r]
            print(f'  bottom row {i}: n={len(r)} cy={int(np.median(ys))} '
                  f'x=[{int(xs[0])}..{int(xs[-1])}] '
                  f'iw={int(np.median(ws))} ih={int(np.median(hs))}')

    # --- Visualize ---
    vis = img.copy()
    for x, y, w, h in ccs:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 1)
    for p in panels:
        cv2.rectangle(vis, (int(p['cols'][0] - p['icon_w']),
                            p['y_top']),
                      (int(p['cols'][-1] + p['icon_w']),
                       p['y_bot']),
                      (0, 200, 255), 2)
    out_path = OUT / 'space_rep_diag.png'
    cv2.imwrite(str(out_path), vis)
    print(f'\nwrote: {out_path}')
    # Bottom-band only, larger
    if H > 250:
        crop = img[bottom_y:, :].copy()
        for x, y, w, h in bottom_ccs:
            cv2.rectangle(crop, (x, y - bottom_y),
                          (x + w, y + h - bottom_y), (0, 255, 0), 1)
        for x, y, w, h in survivors_in_band:
            cv2.rectangle(crop, (x, y - bottom_y),
                          (x + w, y + h - bottom_y), (0, 0, 255), 2)
        out2 = OUT / 'space_rep_bottom_band.png'
        cv2.imwrite(str(out2), crop)
        print(f'wrote: {out2}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
