"""Dump marker w/h and projected slot w/h for each ground panel
on the 6 GROUND_YES images, so we can compare why two of them have
oversized slots while the others look normal.
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

for p in IMAGES:
    img = cv2.imread(p)
    if img is None:
        print(f'!! cannot read {p}')
        continue
    res = detect_panel(img)
    if not res:
        print(f'-- no panel  {p}')
        continue
    g_seats = [s for s in res['seats'] if s[5] == 'G']
    s_seats = [s for s in res['seats'] if s[5] != 'G']
    g_slots = [s for s in res['slots'] if s[6] == 'G']
    s_slots = [s for s in res['slots'] if s[6] != 'G']
    print(f'\n=== {Path(p).name}  img={img.shape[1]}x{img.shape[0]} ===')
    if s_seats:
        sw = [m[3] for m in s_seats]
        sh = [m[4] for m in s_seats]
        print(f'  SPACE seats: n={len(s_seats)}  marker w={sw}  h={sh}')
        if s_slots:
            print(f'    SPACE slot ex: w={s_slots[0][5]}  h={s_slots[0][6]}')
    if g_seats:
        gw = [m[3] for m in g_seats]
        gh = [m[4] for m in g_seats]
        print(f'  GROUND seats: n={len(g_seats)}  marker w={gw}  h={gh}')
        if g_slots:
            print(f'    GROUND slot ex: w={g_slots[0][5]}  h={g_slots[0][6]}'
                  f'  (ab_h derived from median marker_h / 0.63)')
            from statistics import median
            med_h = median(gh)
            print(f'    median marker_h={med_h:.0f} → ab_h={round(med_h / 0.63):d}')
