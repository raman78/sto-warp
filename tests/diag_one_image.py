"""Compare full detect_traits output at thr=30 vs thr=50 for one image."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

IMG = Path(sys.argv[1])
img = cv2.imread(str(IMG))
print(f'image: {IMG.name} {img.shape[1]}x{img.shape[0]}\n')

import warp.recognition.trait_grid as tg


def run_with_threshold(thr):
    src = (f'    _, mask = cv2.threshold(gray, {thr}, '
           f'255, cv2.THRESH_BINARY)')
    # Hot-swap _detect_icon_ccs body for the test
    orig = tg._detect_icon_ccs

    def detect_icon_ccs(img):
        H = img.shape[0]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        h_lo = int(H * tg.ICON_H_FRAC_LO)
        h_hi_frac = 0.65 if H < 250 else tg.ICON_H_FRAC_HI
        h_hi = int(H * h_hi_frac)
        ccs = []
        for i in range(1, n):
            x, y, w, h, _ = stats[i]
            if h < h_lo or h > h_hi:
                continue
            if w < int(h * tg.ICON_AR_LO) or w > int(h * tg.ICON_AR_HI) + 4:
                continue
            ccs.append((int(x), int(y), int(w), int(h)))
        return ccs

    tg._detect_icon_ccs = detect_icon_ccs
    ccs = tg._detect_icon_ccs(img)
    rows = tg._find_trait_rows(ccs)
    groups = tg._cluster_row_groups(rows)
    panels = tg._lock_grids_multi(groups)
    tg._detect_icon_ccs = orig
    return ccs, rows, groups, panels


for thr in (30, 50):
    ccs, rows, groups, panels = run_with_threshold(thr)
    print(f'=== thr={thr} ===')
    print(f'  CCs={len(ccs)}  rows={len(rows)}  groups={len(groups)}  panels={len(panels)}')
    for i, p in enumerate(panels):
        print(f'  panel{i}: iw={p["icon_w"]:.0f} ih={p["icon_h"]:.0f} '
              f'cols={[round(c,1) for c in p["cols"]]} y={p["y_top"]}..{p["y_bot"]}')
    for i, r in enumerate(rows):
        ys = [b[1] + b[3] / 2 for b in r]
        ws = [b[2] for b in r]
        cy = sum(ys) / len(ys)
        iw = sum(ws) / len(ws)
        x0 = min(b[0] for b in r)
        x1 = max(b[0] + b[2] for b in r)
        print(f'    row{i}: n={len(r)} cy={cy:.0f} iw={iw:.0f} x=[{x0}..{x1}]')
    print()
