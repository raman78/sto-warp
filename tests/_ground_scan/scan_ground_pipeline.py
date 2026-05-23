"""Full-pipeline ground-BOFF scan over /home/raman/STO_screens.

Runs ScreenTypeClassifier → LayoutDetector.detect for every screenshot
under STO_screens. Splits results into two folders relative to this
script:
  ground_yes/ — pipeline assigned >=1 ground seat (key contains '[G')
  ground_no/  — image was BOFF-classed but no G seat returned

Each viz draws every detected seat bbox + the seat_id label.
"""
import os, sys, shutil, cv2
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent  # sto-warp/
sys.path.insert(0, str(REPO))

from warp.recognition.layout_detector import LayoutDetector
from warp.recognition.screen_classifier import ScreenTypeClassifier

ROOT = '/home/raman/STO_screens'
OUT_YES = HERE / 'ground_yes'
OUT_NO = HERE / 'ground_no'
for d in (OUT_YES, OUT_NO):
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)

MODELS = REPO / 'warp' / 'models'
clf = ScreenTypeClassifier(MODELS)
ld = LayoutDetector()

CLF_TO_BUILD = {
    'BOFFS': 'BOFFS',
    'SPACE_MIXED': 'SPACE_MIXED',
    'GROUND_MIXED': 'GROUND_MIXED',
}

paths = []
for dp, _, files in os.walk(ROOT):
    for f in files:
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            paths.append(os.path.join(dp, f))
paths.sort()
print(f'scanning {len(paths)} files')

yes_list, no_list, skip_list = [], [], []
errors = 0
for p in paths:
    try:
        img = cv2.imread(p)
        if img is None:
            continue
        stype, conf = clf.classify(img)
        if stype not in CLF_TO_BUILD:
            skip_list.append((p, stype, conf))
            continue
        build = CLF_TO_BUILD[stype]
        layout = ld.detect(img, build)
    except Exception as e:
        errors += 1
        print(f'ERR {p}: {e}')
        continue

    g_seats = [k for k in (layout or {}) if '[G' in k]
    other_seats = [k for k in (layout or {}) if k.startswith('Boff Seat ')
                   and '[G' not in k]

    out = img.copy()
    for sid, bxs in (layout or {}).items():
        is_g = '[G' in sid
        color = (40, 120, 255) if is_g else ((0, 255, 0) if 'Boff Seat' in sid else (200, 200, 200))
        for box in bxs:
            x, y, w, h = box[0], box[1], box[2], box[3]
            cv2.rectangle(out, (x, y), (x + w, y + h), color, 2 if is_g else 1)
        if bxs:
            x0 = min(b[0] for b in bxs)
            y0 = min(b[1] for b in bxs)
            cv2.putText(out, sid, (x0, max(12, y0 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    rel = os.path.relpath(p, ROOT).replace('/', '__')
    stem = os.path.splitext(rel)[0]
    if g_seats:
        cv2.imwrite(str(OUT_YES / (stem + '.png')), out)
        yes_list.append((p, stype, conf, len(g_seats), len(other_seats)))
    else:
        cv2.imwrite(str(OUT_NO / (stem + '.png')), out)
        no_list.append((p, stype, conf, len(other_seats)))

with open(HERE / 'ground_yes.txt', 'w') as f:
    for p, st, cf, g, o in yes_list:
        f.write(f'G={g:2d} other={o:2d}  clf={st}({cf:.2f})  {p}\n')
with open(HERE / 'ground_no.txt', 'w') as f:
    for p, st, cf, o in no_list:
        f.write(f'other={o:2d}  clf={st}({cf:.2f})  {p}\n')
with open(HERE / 'ground_skipped.txt', 'w') as f:
    for p, st, cf in skip_list:
        f.write(f'clf={st}({cf:.2f})  {p}\n')

print(f'\nGROUND_YES: {len(yes_list)}')
for p, st, cf, g, o in yes_list:
    print(f'  G={g:2d} other={o:2d}  clf={st}({cf:.2f})  {os.path.relpath(p, ROOT)}')
print(f'\nGROUND_NO (boff-classed, no G seat): {len(no_list)}')
print(f'SKIPPED (not boff-classed): {len(skip_list)}')
print(f'errors: {errors}')
