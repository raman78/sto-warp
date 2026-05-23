"""Overlay the brown HSV mask on top of the original tab area, 8× zoom.
For each of 6 images, crop around the first detected ground marker and
side-by-side: original | mask | overlay (mask=red over original).

This shows EXACTLY which pixels the band accepts. On GOOD images it
should cling tightly to the brown tab. On BAD images it should
visibly bleed into background/edge pixels.
"""
import sys, cv2, numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

from warp.recognition.boff_marker import MAIN_BANDS, detect_panel

GND = [b for b in MAIN_BANDS if b[7] == 'G'][0]
_, h_lo, h_hi, s_lo, s_hi, v_lo, v_hi, _ = GND

IMAGES = [
    ('GOOD_overview1',       '/home/raman/STO_screens/screeny/overview1.png'),
    ('GOOD_2026-01-19',      '/home/raman/STO_screens/screeny/Screenshot_2026-01-19_145546.png'),
    ('BAD__2026-01-16',      '/home/raman/STO_screens/screeny/screenshot_2026-01-16-21-45-54.jpg'),
    ('BAD__2026-01-23',      '/home/raman/STO_screens/screeny/screenshot_2026-01-23-21-27-06.jpg'),
    ('GOOD_2026-03-27',      '/home/raman/STO_screens/screeny2/Screenshot_2026-03-27_071621-6384970cb81a6316.png'),
    ('GOOD_2026-01-31',      '/home/raman/STO_screens/screeny2/image_2026-01-31_120257300-4e2d65dbf0c82a6b.png'),
]

OUT = HERE / 'mask_overlay'
OUT.mkdir(exist_ok=True)

SCALE = 8
PAD = 8  # extra context around the tab

for tag, p in IMAGES:
    img = cv2.imread(p)
    res = detect_panel(img)
    if not res:
        continue
    # First ground marker
    g_marks = [m for m in res['seats'] if m[5] == 'G']
    if not g_marks:
        continue
    side, mx, my, mw, mh, code, spec = g_marks[0]

    # Crop region with padding
    x0 = max(0, mx - PAD); y0 = max(0, my - PAD)
    x1 = min(img.shape[1], mx + mw + PAD); y1 = min(img.shape[0], my + mh + PAD)
    crop = img[y0:y1, x0:x1].copy()

    # Build the mask on the crop
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    mask = ((H >= h_lo) & (H <= h_hi) & (S >= s_lo) & (S <= s_hi)
            & (V >= v_lo) & (V <= v_hi)).astype(np.uint8) * 255

    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    overlay = crop.copy()
    overlay[mask > 0] = (overlay[mask > 0] * 0.4 + np.array([0, 0, 255]) * 0.6).astype(np.uint8)

    panel = np.concatenate([crop, mask_bgr, overlay], axis=1)
    h_p, w_p = panel.shape[:2]
    big = cv2.resize(panel, (w_p * SCALE, h_p * SCALE),
                     interpolation=cv2.INTER_NEAREST)
    # Annotate columns
    cv2.putText(big, 'orig',    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    cv2.putText(big, 'mask',    (w_p*SCALE//3 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    cv2.putText(big, 'overlay', (2*w_p*SCALE//3 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    cv2.imwrite(str(OUT / f'{tag}.png'), big)

    # Stats on accepted pixels
    if mask.sum() > 0:
        accepted = (mask > 0)
        Hs = H[accepted]; Ss = S[accepted]; Vs = V[accepted]
        print(f'{tag:24s}  marker={mw}x{mh}  px={accepted.sum():4d}  '
              f'H[{Hs.min():3d}-{Hs.max():3d}]  '
              f'S[{Ss.min():3d}-{Ss.max():3d}]  '
              f'V[{Vs.min():3d}-{Vs.max():3d}]')

print(f'\nwrote 8x zoom overlays to {OUT}/')
