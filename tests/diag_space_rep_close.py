"""Hypothesis check: does morphological CLOSE on the threshold mask
recover the Space Reputation row by merging icon fragments?"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from warp.recognition.trait_grid import (
    ICON_H_FRAC_LO, ICON_H_FRAC_HI, ICON_AR_LO, ICON_AR_HI,
)

IMG_PATH = Path('/home/raman/STO_screens/screeny/20250719145506_1.jpg')
OUT = Path(__file__).parent / '_diag_out'
OUT.mkdir(exist_ok=True)


def cc_pass(img, kernel_size, mode='close'):
    H = img.shape[0]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    if kernel_size > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT,
                                       (kernel_size, kernel_size))
        op = cv2.MORPH_CLOSE if mode == 'close' else cv2.MORPH_DILATE
        mask = cv2.morphologyEx(mask, op, k)
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
    return mask, ccs


def main():
    img = cv2.imread(str(IMG_PATH))
    H = img.shape[0]
    bottom_y = int(H * 0.65)

    for k in (0, 2, 3, 4, 5):
        mask, ccs = cc_pass(img, k)
        bottom = [c for c in ccs if c[1] >= bottom_y]
        rep_band = [c for c in bottom if c[1] + c[3] / 2 > bottom_y + 50]
        print(f'kernel={k}: total_icons={len(ccs):3d}  '
              f'bottom={len(bottom):2d}  rep_band(cy>{bottom_y+50})={len(rep_band):2d}')
        for x, y, w, h in rep_band:
            print(f'    cc x={x} y={y} w={w} h={h}')

        vis = img.copy()
        for x, y, w, h in ccs:
            color = (0, 0, 255) if y >= bottom_y else (0, 255, 0)
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 1)
        cv2.imwrite(str(OUT / f'rep_close_k{k}.png'), vis)
        cv2.imwrite(str(OUT / f'rep_mask_k{k}.png'), mask)


if __name__ == '__main__':
    main()
