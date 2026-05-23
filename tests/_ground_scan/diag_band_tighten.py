"""Try tighter GND HSV bands against the 6 GROUND_YES images.

For each image: crop region around the first detected ground marker,
build masks under CURRENT band + 3 candidate tighter bands, save an
8x zoom side-by-side panel. Also dump per-band CC bbox sizes for the
top-area blob so we can compare marker size shrinkage.
"""
import sys, cv2, numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

from warp.recognition.boff_marker import detect_panel

# (label, h_lo, h_hi, s_lo, s_hi, v_lo, v_hi)
BANDS = [
    ('CURR', 15, 30, 120, 230, 25, 95),
    ('A',    18, 28, 140, 230, 30, 95),
    ('B',    18, 28, 150, 230, 35, 95),
    ('C',    19, 27, 160, 230, 35, 95),
]

IMAGES = [
    ('GOOD_overview1',  '/home/raman/STO_screens/screeny/overview1.png'),
    ('GOOD_01-19',      '/home/raman/STO_screens/screeny/Screenshot_2026-01-19_145546.png'),
    ('BAD__01-16',      '/home/raman/STO_screens/screeny/screenshot_2026-01-16-21-45-54.jpg'),
    ('BAD__01-23',      '/home/raman/STO_screens/screeny/screenshot_2026-01-23-21-27-06.jpg'),
    ('GOOD_03-27',      '/home/raman/STO_screens/screeny2/Screenshot_2026-03-27_071621-6384970cb81a6316.png'),
    ('GOOD_01-31',      '/home/raman/STO_screens/screeny2/image_2026-01-31_120257300-4e2d65dbf0c82a6b.png'),
]

OUT = HERE / 'band_tighten'
OUT.mkdir(exist_ok=True)
SCALE = 8
PAD = 8

def build_mask(hsv, band):
    _, h_lo, h_hi, s_lo, s_hi, v_lo, v_hi = band
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    return ((H >= h_lo) & (H <= h_hi) & (S >= s_lo) & (S <= s_hi)
            & (V >= v_lo) & (V <= v_hi)).astype(np.uint8) * 255

def biggest_cc_bbox(mask):
    """Return (w, h, area) of the largest CC, or None."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None
    best = None
    for i in range(1, n):
        x, y, w, h, area = (int(stats[i, j]) for j in range(5))
        if 10 <= w <= 80 and 8 <= h <= 80 and area >= 40:
            if best is None or area > best[2]:
                best = (w, h, area)
    return best

print(f'{"image":18s} | ' + ' | '.join(f'{b[0]:>14s}' for b in BANDS))
print('-' * (18 + len(BANDS) * 17))

for tag, p in IMAGES:
    img = cv2.imread(p)
    res = detect_panel(img)
    if not res:
        continue
    g_marks = [m for m in res['seats'] if m[5] == 'G']
    if not g_marks:
        continue
    side, mx, my, mw, mh, code, spec = g_marks[0]

    x0 = max(0, mx - PAD); y0 = max(0, my - PAD)
    x1 = min(img.shape[1], mx + mw + PAD); y1 = min(img.shape[0], my + mh + PAD)
    crop = img[y0:y1, x0:x1].copy()
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    cols = [crop]  # original first
    sizes = []
    for band in BANDS:
        m = build_mask(hsv, band)
        overlay = crop.copy()
        overlay[m > 0] = (overlay[m > 0] * 0.4 + np.array([0, 0, 255]) * 0.6).astype(np.uint8)
        cols.append(overlay)
        cc = biggest_cc_bbox(m)
        sizes.append(f'{cc[0]}x{cc[1]}' if cc else '----')

    panel = np.concatenate(cols, axis=1)
    h_p, w_p = panel.shape[:2]
    big = cv2.resize(panel, (w_p * SCALE, h_p * SCALE), interpolation=cv2.INTER_NEAREST)

    col_w = w_p * SCALE // (1 + len(BANDS))
    cv2.putText(big, 'orig', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    for i, band in enumerate(BANDS):
        cv2.putText(big, band[0], (col_w * (i+1) + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    cv2.imwrite(str(OUT / f'{tag}.png'), big)

    print(f'{tag:18s} | ' + ' | '.join(f'{s:>14s}' for s in sizes))

print(f'\nwrote overlays to {OUT}/')
