"""Why did space detection regress on image-4eb0d22d10c1f525.png
after per-family refine of icon dims?

Dump:
  - all markers found by detect_markers
  - median W/H computed from ALL markers (old behavior)
  - median W/H computed from SPACE markers only (new behavior)
  - best_panel result for space markers under both dim hints
"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

import cv2
from warp.recognition.boff_marker import (
    estimate_icon_dims, detect_markers, _refine_dims_from_markers,
    best_panel, _SPACE_CODES, _GROUND_CODES,
)

P = '/home/raman/STO_screens/screeny2/image-4eb0d22d10c1f525.png'
img = cv2.imread(P)
print(f'image {img.shape[1]}x{img.shape[0]}')

icon_w0, icon_h0 = estimate_icon_dims(img)
print(f'\nestimate_icon_dims: w={icon_w0} h={icon_h0}')

markers = detect_markers(img, icon_w0, icon_h0)
print(f'\ndetect_markers: {len(markers)} total')
by_code = {}
for m in markers:
    by_code.setdefault(m[4], []).append(m)
for c in sorted(by_code):
    ms = by_code[c]
    sizes = ', '.join(f'{m[2]}x{m[3]}' for m in ms)
    print(f'  [{c}] n={len(ms):2d}  sizes: {sizes}')

space = [m for m in markers if m[4] in _SPACE_CODES]
ground = [m for m in markers if m[4] in _GROUND_CODES]
print(f'\nspace markers: {len(space)}    ground markers: {len(ground)}')

# Old (combined) refine
all_w, all_h = _refine_dims_from_markers(markers, icon_w0, icon_h0)
print(f'\nOLD refine (all markers):    w={all_w} h={all_h}')

# New (per-family) refine
sp_w, sp_h = _refine_dims_from_markers(space, icon_w0, icon_h0)
gd_w, gd_h = _refine_dims_from_markers(ground, icon_w0, icon_h0) if ground else (icon_w0, icon_h0)
print(f'NEW refine (space only):     w={sp_w} h={sp_h}')
print(f'NEW refine (ground only):    w={gd_w} h={gd_h}')

# best_panel for space with both dims
print(f'\nbest_panel(space, OLD={all_w}x{all_h}, img):')
p_old = best_panel(space, all_w, all_h, img=img)
print('  ', 'None' if p_old is None
      else f'a={len(p_old[0])} b={len(p_old[1])} score={p_old[2]:.3f}')

print(f'best_panel(space, NEW={sp_w}x{sp_h}, img):')
p_new = best_panel(space, sp_w, sp_h, img=img)
print('  ', 'None' if p_new is None
      else f'a={len(p_new[0])} b={len(p_new[1])} score={p_new[2]:.3f}')
