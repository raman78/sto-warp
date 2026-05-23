"""For each GROUND_YES image, draw BOTH the seat marker bboxes (the
brown tab for ground / coloured bar for space) AND the projected
ability slot bboxes. Saves to ./markers_and_slots/.

This is the missing viz layer compared to scan_ground_pipeline.py,
which only drew slots.
"""
import sys, cv2
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

from warp.recognition.boff_marker import detect_panel

IMAGES = [
    '/home/raman/STO_screens/screeny/overview1.png',
    '/home/raman/STO_screens/screeny/Screenshot_2026-01-19_145546.png',
    '/home/raman/STO_screens/screeny/screenshot_2026-01-16-21-45-54.jpg',
    '/home/raman/STO_screens/screeny/screenshot_2026-01-23-21-27-06.jpg',
    '/home/raman/STO_screens/screeny2/Screenshot_2026-03-27_071621-6384970cb81a6316.png',
    '/home/raman/STO_screens/screeny2/image_2026-01-31_120257300-4e2d65dbf0c82a6b.png',
]

OUT = HERE / 'markers_and_slots'
OUT.mkdir(exist_ok=True)

# Marker outline = solid, slot outline = dashed-look (thin)
MARKER_COLORS = {
    'T': (0, 0, 255), 'E': (0, 215, 255), 'S': (255, 200, 0),
    'U': (220, 220, 220), 'G': (40, 120, 255),
}
SLOT_COLOR = (180, 255, 180)

for p in IMAGES:
    img = cv2.imread(p)
    res = detect_panel(img)
    if not res:
        print(f'-- no panel: {p}')
        continue
    out = img.copy()

    # 1) Seat marker boxes (thick, coloured by code)
    for (_side, mx, my, mw, mh, code, _spec) in res['seats']:
        c = MARKER_COLORS.get(code, (0, 255, 0))
        cv2.rectangle(out, (mx, my), (mx + mw, my + mh), c, 2)
        cv2.putText(out, code, (mx, max(10, my - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)

    # 2) Ability slot boxes (thin, light-green)
    for (_si, _ki, x, y, w, h, _code) in res['slots']:
        cv2.rectangle(out, (x, y), (x + w, y + h), SLOT_COLOR, 1)

    name = Path(p).stem
    cv2.imwrite(str(OUT / (name + '.png')), out)
    n_g = sum(1 for s in res['seats'] if s[5] == 'G')
    n_s = sum(1 for s in res['seats'] if s[5] != 'G')
    print(f'  {Path(p).name}  space={n_s}  ground={n_g}  slots={len(res["slots"])}')

print(f'\nwrote viz to {OUT}/')
