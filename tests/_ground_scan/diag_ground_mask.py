"""For each GROUND_YES image, dump the brown-HSV connected components
in the panel zone so we can see whether the tab CC bleeds into a
neighbouring brown area on the two oversized-slot images.
"""
import sys, cv2, numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

from warp.recognition.boff_marker import MAIN_BANDS

GND = [b for b in MAIN_BANDS if b[7] == 'G'][0]
_, h_lo, h_hi, s_lo, s_hi, v_lo, v_hi, _ = GND

IMAGES = [
    ('GOOD', '/home/raman/STO_screens/screeny/overview1.png'),
    ('GOOD', '/home/raman/STO_screens/screeny/Screenshot_2026-01-19_145546.png'),
    ('BAD',  '/home/raman/STO_screens/screeny/screenshot_2026-01-16-21-45-54.jpg'),
    ('BAD',  '/home/raman/STO_screens/screeny/screenshot_2026-01-23-21-27-06.jpg'),
    ('GOOD', '/home/raman/STO_screens/screeny2/Screenshot_2026-03-27_071621-6384970cb81a6316.png'),
    ('GOOD', '/home/raman/STO_screens/screeny2/image_2026-01-31_120257300-4e2d65dbf0c82a6b.png'),
]

OUT = HERE / 'mask_dump'
OUT.mkdir(exist_ok=True)

for tag, p in IMAGES:
    img = cv2.imread(p)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    mask = ((H >= h_lo) & (H <= h_hi) & (S >= s_lo) & (S <= s_hi)
            & (V >= v_lo) & (V <= v_hi)).astype(np.uint8) * 255
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    # Filter to plausible tab-size blobs (matches detect_markers ranges)
    cands = []
    for i in range(1, n):
        x, y, w, h, area = (int(stats[i, j]) for j in range(5))
        if 15 <= w <= 60 and 12 <= h <= 60 and area >= 80:
            cands.append((area, x, y, w, h))
    cands.sort(reverse=True)
    print(f'\n[{tag}] {Path(p).name}')
    for area, x, y, w, h in cands[:10]:
        ar = w / max(h, 1)
        print(f'  bbox=({x:4d},{y:4d}) {w:3d}x{h:3d}  area={area:5d}  ar={ar:.2f}')

    out = img.copy()
    for area, x, y, w, h in cands[:10]:
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 255), 1)
    stem = f'{tag}__{Path(p).stem}'
    cv2.imwrite(str(OUT / f'{stem}.png'), out)
