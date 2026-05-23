"""Sweep _detect_icon_ccs threshold across the full screenshot corpus.

For each image, run trait_grid panel detection at:
  - thr=30 (current default)
  - thr=50 (proposed fixed bump)
  - Otsu (adaptive)

Report per image: n_panels, total bboxes, panel iw stats. Highlight
regressions (cases where thr=30 worked but thr=50/Otsu doesn't).
"""
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

CORPUS = Path('/home/raman/STO_screens/screeny')


def ccs_with(img, mode):
    H = img.shape[0]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if mode == 'otsu':
        _, mask = cv2.threshold(gray, 0, 255,
                                cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    else:
        _, mask = cv2.threshold(gray, int(mode), 255, cv2.THRESH_BINARY)
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


def analyze(img, mode):
    ccs = ccs_with(img, mode)
    rows = _find_trait_rows(ccs)
    groups = _cluster_row_groups(rows)
    panels = _lock_grids_multi(groups)
    real = [p for p in panels if p['icon_w'] >= 20]
    garbage = [p for p in panels if p['icon_w'] < 20]
    total_bb = sum(len(r) for r in rows)
    return {
        'panels': len(panels),
        'real_panels': len(real),
        'garbage_panels': len(garbage),
        'total_rows': len(rows),
        'total_bbox': total_bb,
        'panel_iw': [round(p['icon_w'], 0) for p in panels],
    }


def main():
    images = sorted(CORPUS.glob('*'))
    images = [p for p in images if p.suffix.lower() in ('.png', '.jpg', '.jpeg')]
    print(f'corpus: {len(images)} images\n')

    headers = ['thr=30', 'thr=50', 'otsu']
    summary = {h: {'panels_real': 0, 'panels_garbage': 0,
                   'total_bbox': 0, 'images_with_real_panel': 0}
               for h in headers}

    regressions = []
    improvements = []
    print(f'{"image":<40} {"thr=30":<22} {"thr=50":<22} {"otsu":<22}')
    print('-' * 110)
    for ip in images:
        img = cv2.imread(str(ip))
        if img is None:
            continue
        r30 = analyze(img, 30)
        r50 = analyze(img, 50)
        ros = analyze(img, 'otsu')

        def fmt(r):
            return f'P={r["real_panels"]}+{r["garbage_panels"]}g bb={r["total_bbox"]}'

        line = f'{ip.name:<40} {fmt(r30):<22} {fmt(r50):<22} {fmt(ros):<22}'
        # Regression: thr=30 had real panels, new mode has fewer
        if r30['real_panels'] > r50['real_panels']:
            line += '  ⚠50regress'
            regressions.append((ip.name, '50', r30, r50))
        if r30['real_panels'] > ros['real_panels']:
            line += '  ⚠otsu regress'
            regressions.append((ip.name, 'otsu', r30, ros))
        if r50['total_bbox'] > r30['total_bbox'] + 2:
            line += '  ✓50gain'
            improvements.append((ip.name, '50', r30, r50))
        print(line)

        for h, r in zip(headers, (r30, r50, ros)):
            summary[h]['panels_real'] += r['real_panels']
            summary[h]['panels_garbage'] += r['garbage_panels']
            summary[h]['total_bbox'] += r['total_bbox']
            if r['real_panels'] > 0:
                summary[h]['images_with_real_panel'] += 1

    print('\n=== Summary ===')
    for h in headers:
        s = summary[h]
        print(f'  {h}: images_with_real_panel={s["images_with_real_panel"]:3d}  '
              f'real_panels={s["panels_real"]:4d}  '
              f'garbage_panels={s["panels_garbage"]:4d}  '
              f'total_bbox={s["total_bbox"]:5d}')

    if regressions:
        print(f'\n=== Regressions ({len(regressions)}) ===')
        for name, mode, r_old, r_new in regressions[:20]:
            print(f'  {name} @ {mode}: real {r_old["real_panels"]}→{r_new["real_panels"]}  '
                  f'bbox {r_old["total_bbox"]}→{r_new["total_bbox"]}')

    if improvements:
        print(f'\n=== Improvements (+2 bbox vs thr=30 at thr=50) ({len(improvements)}) ===')
        for name, mode, r_old, r_new in improvements[:20]:
            print(f'  {name}: bbox {r_old["total_bbox"]}→{r_new["total_bbox"]}')


if __name__ == '__main__':
    main()
