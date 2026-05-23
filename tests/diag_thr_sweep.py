"""Compare _detect_icon_ccs at thr=30 vs thr=50 across the whole image."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from warp.recognition.trait_grid import (
    _find_trait_rows, _cluster_row_groups, _lock_grids_multi,
    ICON_H_FRAC_LO, ICON_H_FRAC_HI, ICON_AR_LO, ICON_AR_HI,
)

IMG_PATH = Path('/home/raman/STO_screens/screeny/20250719145506_1.jpg')


def ccs_at(img, thr):
    H = img.shape[0]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    h_lo = int(H * ICON_H_FRAC_LO)
    h_hi_frac = 0.65 if H < 250 else ICON_H_FRAC_HI
    h_hi = int(H * h_hi_frac)
    ccs = []
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        if h < h_lo or h > h_hi:
            continue
        if w < int(h * ICON_AR_LO) or w > int(h * ICON_AR_HI) + 4:
            continue
        ccs.append((int(x), int(y), int(w), int(h)))
    return ccs


img = cv2.imread(str(IMG_PATH))
for thr in (30, 40, 50, 60):
    ccs = ccs_at(img, thr)
    rows = _find_trait_rows(ccs)
    groups = _cluster_row_groups(rows)
    panels = _lock_grids_multi(groups)
    print(f'\nthr={thr}: ccs={len(ccs)} rows={len(rows)} groups={len(groups)} panels={len(panels)}')
    for i, p in enumerate(panels):
        print(f'  panel{i}: iw={p["icon_w"]:.0f} ih={p["icon_h"]:.0f} '
              f'y={p["y_top"]}..{p["y_bot"]} cols={[round(c,1) for c in p["cols"]]}')
    # List rows by cy
    for i, r in enumerate(rows):
        ys = [b[1] + b[3] / 2 for b in r]
        ws = [b[2] for b in r]
        cy = sum(ys) / len(ys)
        iw = sum(ws) / len(ws)
        x0 = min(b[0] for b in r)
        x1 = max(b[0] + b[2] for b in r)
        print(f'    row{i} n={len(r)} cy={cy:.0f} iw={iw:.0f} x=[{x0}..{x1}]')
