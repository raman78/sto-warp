"""List ALL raw CCs in the Space Reputation Y-band, by size."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

IMG_PATH = Path('/home/raman/STO_screens/screeny/20250719145506_1.jpg')
OUT = Path(__file__).parent / '_diag_out'

img = cv2.imread(str(IMG_PATH))
H, W = img.shape[:2]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Locate rep row Y-band: look at saved bottom_band image — rep icons at
# image-coord y ≈ 380-420 (relative cy=405 from earlier diag).
y_lo, y_hi = 370, 425

for thr in (10, 20, 30, 50, 80):
    _, mask = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    in_band = []
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        cy = y + h / 2
        if y_lo <= cy <= y_hi and w >= 6 and h >= 6:
            in_band.append((x, y, w, h))
    in_band.sort(key=lambda b: b[0])
    print(f'\nthr={thr}: {len(in_band)} CCs in rep row band (cy in [{y_lo},{y_hi}])')
    for x, y, w, h in in_band:
        ar = w / h
        print(f'  x={x:3d} y={y:3d} w={w:3d} h={h:3d}  ar={ar:.2f}')
    cv2.imwrite(str(OUT / f'rep_thr{thr}_mask.png'), mask)
