"""
BOFF profession-marker detector — production module.

Locates a BOFF panel in a screenshot by detecting the small profession-
coloured badges that sit on the LEFT of each seat name bar (which itself
sits BELOW the seat's 4 ability icons). From the badge grid we recover:

  - the canonical 3+2 (or 2+2 / 2+1) seat layout,
  - per-seat ability slot bboxes via bible-driven projection.

This module is detection-only. It MUST NOT read `annotations.json` or
any user ground truth (CORE RULE: WARP detection must derive its output
from pixels).

Public API:
    detect_panel(img) -> Optional[PanelResult]

PanelResult is a TypedDict-like dict:
    {
      'col_a':   list[(x, y, w, h, code, spec_code | None)],  # left column
      'col_b':   list[(x, y, w, h, code, spec_code | None)],  # right column
      'score':   float,                            # RANSAC score
      'seats':   list[(side, mx, my, mw, mh, seat_code, spec_code | None)],
      'slots':   list[(seat_idx, slot_idx, x, y, w, h, seat_code)],
    }

Algorithm prototype + 2026-04-30 baseline (36 GT screens):
  - Panel anchor:  100% (36/36)
  - Seat hit:      100% (177/177)
  - Slot IoU≥.30:  99.9%
  - Slot IoU≥.50:  99.9%
  - Slot IoU≥.70:  96.0%

Algorithm reference: docs/BOFF_DETECTION.md.
Diagnostic prototype:  tests/diag_boff_markers.py.
"""
from __future__ import annotations

import statistics as st
from typing import Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Bands
# ---------------------------------------------------------------------------

# Seat-type bands (main zone of the marker, 75-95% of bar width).
# Sampled across 25+ GT seats. Used as DETECTION seeds.
MAIN_BANDS: list[tuple[str, int, int, int, int, int, int, str]] = [
    # name, H_lo, H_hi, S_lo, S_hi, V_lo, V_hi, code
    ('TAC',   0,   6,   125, 255,  85, 255, 'T'),  # red
    ('TAC', 174, 180,   125, 255,  85, 255, 'T'),  # red wrap
    ('ENG',  18,  33,   100, 220, 140, 255, 'E'),  # saturated gold (V_lo: 160->140, H_hi: 30->33 for brighter renderings)
    ('SCI', 102, 114,   160, 255,  90, 255, 'S'),  # blue (V_lo: 140->90 for dark UI)
    ('UNI',  18,  33,    25,  95, 145, 255, 'U'),  # pale cream (V_lo: 195->145, H_hi: 30->33 for darker-tinted bars in compressed UI)
    ('UNI',  10,  35,     0,  25, 220, 255, 'U'),  # near-white variant — selection-highlighted Universal bars (S near 0, V blown to 255)
    ('GND',  18,  28,   150, 230,  35,  95, 'G'),  # ground BOFF — dark saturated brown (sampled H≈23 S≈206 V≈67 from /home/raman/STO_przydatne/ground.png); tightened from H[15,30] S[120,230] V[25,95] to reject background bleed (darker, less saturated edge pixels) on JPG screenshots with non-uniform surroundings
]

# Spec-stripe bands (narrow right edge, 5-25% of bar width). NOT used
# for detection — applied post-hoc to label each detected seat with
# its specialization.
STRIPE_BANDS: list[tuple[str, int, int, int, int, int, int, str]] = [
    # name, H_lo, H_hi, S_lo, S_hi, V_lo, V_hi, spec_code
    # Spec codes are human-readable abbreviations so seat keys like
    # `Boff Seat L[U+Plt]_96` are intuitive in logs and review UIs
    # (the single-letter codes O/P/Y/C/L did NOT match first letters
    # and confused everyone).
    ('CMD',  12,  19,    80, 200, 130, 200, 'Cmd'),  # Command — orange
    ('INT', 120, 135,   130, 200, 150, 220, 'Int'),  # Intelligence — purple
    ('TMP',  25,  35,   140, 255, 215, 255, 'Tem'),  # Temporal — bright gold
    ('PIL',  86, 100,    80, 140, 215, 255, 'Plt'),  # Pilot — light cyan
    ('MW',   32,  44,   170, 255, 195, 255, 'MW'),   # Miracle Worker — lime (S_lo 220→170 for muted/desaturated lime on darker UI)
]


# ---------------------------------------------------------------------------
# Bible — measured on Stations.png ("idealny rozkład").
# All values are panel-internal pixels at the reference scale where one
# full marker bar is 29 pixels wide. At any other UI scale we rescale
# everything by `k = detected_marker_w / _BIBLE_MARKER_W`.
# ---------------------------------------------------------------------------

_BIBLE_MARKER_W   = 29
_BIBLE_SLOT_W     = 29
_BIBLE_SLOT_H     = 37
_BIBLE_GAP_FIRST  = 3      # marker_right_edge → slot1_left_edge (X)
_BIBLE_GAP_SLOT   = 2      # slot_right_edge   → next_slot_left   (X)
_BIBLE_STRIDE_X   = _BIBLE_SLOT_W + _BIBLE_GAP_SLOT   # = 31


# ---------------------------------------------------------------------------
# Code → human label maps (for callers that want pretty names)
# ---------------------------------------------------------------------------

SEAT_CODE_LABEL = {
    'T': 'Tactical',
    'E': 'Engineering',
    'S': 'Science',
    'U': 'Universal',
    'G': 'Ground',
}
SPEC_CODE_LABEL = {
    'Cmd': 'Command',
    'Int': 'Intelligence',
    'Tem': 'Temporal',
    'Plt': 'Pilot',
    'MW':  'Miracle Worker',
}


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def estimate_icon_dims(img: np.ndarray) -> tuple[int, int]:
    """Initial wide range for the marker size — refined later from
    detected marker sizes."""
    h, _w = img.shape[:2]
    icon_h = max(int(round(h * 0.045)), 24)
    icon_w = max(int(round(icon_h * 0.78)), 22)
    return icon_w, icon_h


def _refine_dims_from_markers(markers, icon_w, icon_h):
    if len(markers) < 4:
        return icon_w, icon_h
    hs = sorted(m[3] for m in markers)
    ws = sorted(m[2] for m in markers)
    med_h = hs[len(hs) // 2]
    med_w = ws[len(ws) // 2]
    if 16 <= med_h <= 90 and 4 <= med_w <= 90:
        return max(med_w, 16), max(med_h, 20)
    return icon_w, icon_h


def _colour_mask(hsv, h_lo, h_hi, s_lo, s_hi, v_lo, v_hi):
    H = hsv[:, :, 0]; S = hsv[:, :, 1]; V = hsv[:, :, 2]
    return ((H >= h_lo) & (H <= h_hi)
            & (S >= s_lo) & (S <= s_hi)
            & (V >= v_lo) & (V <= v_hi)).astype(np.uint8) * 255


def _merge_close_bboxes(boxes, gap_x, overlap_y_frac=0.6):
    """Merge bboxes whose x-gap is ≤ gap_x and whose y-overlap is at
    least overlap_y_frac of the shorter bbox height."""
    if not boxes:
        return boxes
    boxes = sorted(boxes, key=lambda b: b[0])
    merged = [list(boxes[0])]
    for b in boxes[1:]:
        m = merged[-1]
        gx = b[0] - (m[0] + m[2])
        oy = max(0, min(m[1] + m[3], b[1] + b[3]) - max(m[1], b[1]))
        short = max(1, min(m[3], b[3]))
        if gx <= gap_x and (oy / short) >= overlap_y_frac:
            x0 = min(m[0], b[0])
            y0 = min(m[1], b[1])
            x1 = max(m[0] + m[2], b[0] + b[2])
            y1 = max(m[1] + m[3], b[1] + b[3])
            m[0] = x0; m[1] = y0
            m[2] = x1 - x0; m[3] = y1 - y0
            m[4] = m[4] + b[4]
        else:
            merged.append(list(b))
    return [tuple(m) for m in merged]


# ---------------------------------------------------------------------------
# Marker detection
# ---------------------------------------------------------------------------

def detect_markers(img: np.ndarray, icon_w: int, icon_h: int):
    """Detect seat-type markers. Returns list of (x, y, w, h, code).

    Per band:
      1. HSV mask + small horizontal CLOSE.
      2. Connected components on BOTH raw and CLOSE-d masks (dual-mask CC).
      3. _merge_close_bboxes() to glue same-colour fragments.
      4. Size + aspect + fill-density + uniformity + Canny-edge filters.
      5. Cross-band IoU dedupe + size-outlier cull.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Fixed size floor (decoupled from estimate_icon_dims, which mis-scales
    # on PicCollage composites and cropped panels). Floor is intentionally
    # permissive (13x10) — Strategy 0 already gates false positives via
    # v_std_FULL ≤ 45, fill_ratio, edge_frac, and HSV-uniformity checks.
    # Real markers range 13x10 (compressed Stations panel) to 60x60 (Collage).
    abs_min_w, abs_min_h = 13, 10
    abs_max_w, abs_max_h = 90, 70
    min_w = abs_min_w
    h_im = img.shape[0]
    img_rel_max = int(h_im * 0.085)
    max_w = abs_max_w
    min_h = abs_min_h
    max_h = abs_max_h
    ar_min, ar_max = 0.30, 1.8
    fill_min = 0.70

    kx = max(3, int(round(icon_w * 0.22)))
    ky = max(2, icon_h // 12)

    uni_v_max = 28
    uni_h_max = 6
    edge_max = 0.07
    edge_inset = 2
    edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 80, 160)

    # Merge bands sharing the same name (e.g. red TAC wraps 0/180).
    by_name: dict[str, dict] = {}
    for name, h_lo, h_hi, s_lo, s_hi, v_lo, v_hi, code in MAIN_BANDS:
        by_name.setdefault(
            name, {'code': code, 'ranges': []},
        )['ranges'].append((h_lo, h_hi, s_lo, s_hi, v_lo, v_hi))

    out: list[tuple[int, int, int, int, str]] = []
    seen_rects: list[tuple[int, int, int, int]] = []
    for _name, info in by_name.items():
        code = info['code']
        m = None
        for h_lo, h_hi, s_lo, s_hi, v_lo, v_hi in info['ranges']:
            part = _colour_mask(hsv, h_lo, h_hi, s_lo, s_hi, v_lo, v_hi)
            m = part if m is None else cv2.bitwise_or(m, part)

        m_closed = cv2.morphologyEx(
            m, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky)),
        )
        gap_x = max(2, int(round(icon_w * 0.35)))
        merged: list[tuple] = []
        for src in (m, m_closed):
            n, _, stats, _ = cv2.connectedComponentsWithStats(
                src, connectivity=8)
            raw = [tuple(int(v) for v in stats[i, [0, 1, 2, 3, 4]])
                   for i in range(1, n)]
            merged.extend(_merge_close_bboxes(
                raw, gap_x=gap_x, overlap_y_frac=0.55))

        for x, y, w, h, area in merged:
            # Extend total width through any trailing spec stripe (MW
            # lime, INT purple, etc.) so the size filter sees the full
            # bar, not just the main-band core. Small Craft seats with
            # an MW stripe split the bar into 28-29 px main + 4 px
            # stripe + 3-4 px gap — main alone fails min_w. The main-
            # band width (`w`) and bbox stay unchanged; only `full_w`
            # is used for the size/aspect gate.
            full_w, has_spec, stripe_w = full_bar_extent(
                hsv, (x, y, w, h, code))
            gate_w = full_w if has_spec else w
            if gate_w < min_w or gate_w > max_w:
                continue
            if h < min_h or h > max_h:
                continue
            ar = gate_w / max(h, 1)
            if ar < ar_min or ar > ar_max:
                continue
            sel = m[y:y + h, x:x + w] > 0
            if sel.sum() < 20:
                continue
            crop_v = hsv[y:y + h, x:x + w, 2][sel]
            crop_h = hsv[y:y + h, x:x + w, 0][sel]
            v_std = float(np.std(crop_v))
            h_std = min(
                float(np.std(crop_h)),
                float(np.std((crop_h.astype(np.int32) + 90) % 180)),
            )
            if h_std > uni_h_max:
                continue
            # Very-strong hue uniformity (h_std ≤ 2.0) bypasses the
            # brightness checks. When all masked pixels share an almost
            # identical hue, the bar IS a coloured marker even if the
            # rendering has a selection highlight or a row-separator
            # tear-up that pushes v_std above the nominal limits. Random
            # noise blobs never reach this hue stability.
            hue_locked = h_std <= 2.0
            v_max_eff = 35 if hue_locked else uni_v_max
            if v_std > v_max_eff:
                continue
            # Full-bbox uniformity check. False positives like slot icons
            # have a dark glyph in the centre that the mask-only v_std misses.
            # Real markers MAY have dark regions that should NOT be part of
            # the colour/brightness estimate:
            #   1) Always exclude the inner 35%x35% lower-left (rank badge).
            #   2) If the lower-left half is much darker than the rest of the
            #      bbox (mean V diff > 60), the badge is bigger — widen
            #      exclusion to the lower-left 50%x50% quadrant.
            #   3) If the right-edge ~20% strip is much darker than the
            #      rest, the spec-stripe slot is empty (dark UI showing
            #      through) — exclude it too.
            v_full = hsv[y:y + h, x:x + w, 2].astype(np.float32)
            excl = np.ones((h, w), dtype=bool)
            base_excl_w = int(w * 0.35); base_excl_h = int(h * 0.35)
            excl[h - base_excl_h:, :base_excl_w] = False
            half_h0 = int(h * 0.5); half_w1 = int(w * 0.5)
            ll_half = v_full[half_h0:, :half_w1]
            rest_mask = np.ones((h, w), dtype=bool)
            rest_mask[half_h0:, :half_w1] = False
            ll_mean = float(ll_half.mean()) if ll_half.size > 0 else 0.0
            rest_mean = float(v_full[rest_mask].mean()) \
                if rest_mask.sum() > 0 else 0.0
            if rest_mean - ll_mean > 60:
                excl[half_h0:, :half_w1] = False
            right_w = max(1, int(w * 0.2))
            right_strip = v_full[:, w - right_w:]
            left_only = np.ones((h, w), dtype=bool)
            left_only[:, w - right_w:] = False
            right_mean = float(right_strip.mean()) \
                if right_strip.size > 0 else 0.0
            left_rest_mean = float(v_full[left_only].mean()) \
                if left_only.sum() > 0 else 0.0
            if left_rest_mean - right_mean > 60:
                excl[:, w - right_w:] = False
            v_std_full = float(np.std(v_full[excl])) if excl.sum() > 0 else 0.0
            v_full_max_eff = 80 if hue_locked else 45
            if v_std_full > v_full_max_eff:
                continue
            # Strong-uniformity bypass for fill/edge: a flat colour bar
            # (low v_std, low h_std) IS a marker. Spec stripes (e.g. MW
            # lime on Engineering) break the main-zone CC short and add
            # a transition that bumps Canny edge_frac just past the
            # baseline — relax both thresholds when uniformity is high.
            # hue_locked (h_std ≤ 2) is a stricter form of uniformity and
            # also implies strong_uniform.
            strong_uniform = (v_std <= 20 and h_std <= 4) or hue_locked
            fill_thr = 0.60 if strong_uniform else fill_min
            edge_thr = 0.12 if strong_uniform else edge_max
            # When a spec stripe is present inside the bbox, the main-
            # colour mask cannot cover the stripe columns and the stripe
            # transition bumps edge_frac. Relax both thresholds by a fixed
            # amount that matches the typical 3-5 px stripe footprint.
            if has_spec:
                fill_thr -= 0.10
                edge_thr += 0.05
            if area < (w * h) * fill_thr:
                continue
            ix0 = x + edge_inset; iy0 = y + edge_inset
            ix1 = x + w - edge_inset; iy1 = y + h - edge_inset
            if ix1 > ix0 and iy1 > iy0:
                edge_crop = edges[iy0:iy1, ix0:ix1]
                edge_frac = float(edge_crop.sum() / 255.0) / (
                    (ix1 - ix0) * (iy1 - iy0))
                if edge_frac > edge_thr:
                    continue
            dup = False
            for (px, py, pw, ph) in seen_rects:
                if x < px + pw and px < x + w and y < py + ph and py < y + h:
                    ix = min(x + w, px + pw) - max(x, px)
                    iy = min(y + h, py + ph) - max(y, py)
                    iou = (ix * iy) / max(w * h + pw * ph - ix * iy, 1)
                    if iou > 0.4:
                        dup = True
                        break
            if dup:
                continue
            seen_rects.append((x, y, w, h))
            out.append((int(x), int(y), int(w), int(h), code))

    if len(out) >= 4:
        med_w = sorted(m[2] for m in out)[len(out) // 2]
        med_h = sorted(m[3] for m in out)[len(out) // 2]
        out = [m for m in out
               if m[2] >= 0.65 * med_w and m[3] >= 0.65 * med_h]
    return out


# ---------------------------------------------------------------------------
# Spec-stripe / full bar
# ---------------------------------------------------------------------------

_STRIPE_COL_FILL = 0.60   # column must be ≥60% in-band to count
_STRIPE_MIN_RUN  = 2      # need ≥2 consecutive in-band columns


def _stripe_col_scan(hsv, marker):
    """Column-by-column scan for a thin vertical spec stripe just to the
    right of the marker's base bar. Returns (code, run_len, run_end_abs_x)
    or (None, 0, -1).

    Per STRIPE_BAND we compute the per-column fraction of pixels matching
    the band, then keep the longest run of cols with frac ≥ _STRIPE_COL_FILL.
    The winning band is the one with the longest qualifying run. This
    rejects diffuse colour noise (e.g. green name-plate background bleeding
    into the search box) by requiring vertical continuity — a real stripe
    is ~2-5 px wide and fills almost the whole bar height.
    """
    x, y, w, h, _ = marker
    H_im, W_im = hsv.shape[:2]
    look = max(8, int(round(w * 0.45)))
    # Scan covers the last ~20% of the base bar plus a lookahead window.
    # The in-bbox region is needed because some markers (overview.png
    # Xarnok ENG+MW) have the spec stripe physically INSIDE the base
    # bbox (the connectedComponents bbox already includes the stripe).
    # 20% is wide enough to catch a 4-5 px stripe at the bbox edge, narrow
    # enough that solid base colour doesn't fill ≥0.6 of any stripe band
    # (base hues are chosen to be distinguishable from stripe hues).
    sx0 = max(0, x + max(1, int(round(w * 0.80))))
    sx1 = min(W_im, x + w + look)
    sy0 = max(0, y + int(h * 0.15))
    sy1 = min(H_im, y + h - int(h * 0.15))
    if sx1 <= sx0 or sy1 <= sy0:
        return (None, 0, -1)
    crop = hsv[sy0:sy1, sx0:sx1]
    HCH = crop[:, :, 0]; SCH = crop[:, :, 1]; VCH = crop[:, :, 2]
    best_code = None; best_run = 0; best_end = -1
    for _, h_lo, h_hi, s_lo, s_hi, v_lo, v_hi, code in STRIPE_BANDS:
        m = ((HCH >= h_lo) & (HCH <= h_hi)
             & (SCH >= s_lo) & (SCH <= s_hi)
             & (VCH >= v_lo) & (VCH <= v_hi))
        col_frac = m.mean(axis=0)
        cur = 0; run = 0; end = -1
        for ci, f in enumerate(col_frac):
            if f >= _STRIPE_COL_FILL:
                cur += 1
                if cur > run:
                    run = cur; end = ci
            else:
                cur = 0
        if run > best_run:
            best_run = run; best_code = code; best_end = end
    if best_run < _STRIPE_MIN_RUN:
        return (None, 0, -1)
    return (best_code, best_run, sx0 + best_end)


def classify_stripe(hsv, marker, _icon_w_unused=None):
    """Identify the specialization stripe on the right edge of a marker.
    Returns (spec_code, run_len_in_cols) or (None, 0)."""
    code, run, _end = _stripe_col_scan(hsv, marker)
    return (code, run)


def full_bar_extent(hsv, marker):
    """Find the rightmost x where the colour bar (main zone + spec
    stripe) still has a strong column-fill. Returns
    (full_w_including_stripe, has_spec, stripe_width).

    `has_spec` is True whenever `_stripe_col_scan` confirms a stripe,
    even if the stripe lies INSIDE the detection bbox (full_w == w).
    Some markers (overview.png Xarnok ENG+MW) ship with the stripe
    already inside the connectedComponents bbox; downstream gating in
    detect_markers needs to know the bar is multi-coloured so the
    fill_ratio / edge_frac thresholds can be relaxed accordingly.
    """
    x, _y, w, _h, _code = marker
    code, _run, end_abs = _stripe_col_scan(hsv, marker)
    if code is None:
        return (w, False, 0)
    full_w = max(w, end_abs - x + 1)
    has_spec = True
    stripe_w = max(0, full_w - w)
    return (full_w, has_spec, stripe_w)


def annotate_specs(img, markers):
    """Return [(x, y, w, h, seat_code, spec_code_or_None, score)]."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    out = []
    for x, y, w, h, code in markers:
        spec, score = classify_stripe(hsv, (x, y, w, h, code))
        out.append((x, y, w, h, code, spec, round(score, 3)))
    return out


# ---------------------------------------------------------------------------
# Panel selection — RANSAC-style 3+2 grid search
# ---------------------------------------------------------------------------

def _slot_evidence(gray, edges, marker, scale_k):
    """Score in [0, 1]: does the region right of `marker` look like an
    STO ability slot? Used to disambiguate real BOFF panels from noise
    markers (avatar skin, item icons, etc.) when many candidates compete.

    Geometry from bible: slot starts `3*k` px right of marker, is `29*k`
    wide and `37*k` tall (k = marker_w / 29). The slot is centred
    vertically on the marker.

    Heuristic:
      - edge density in slot region: 0.03 < e < 0.45  (frame + glyph)
      - mean grey ≤ 200                              (not blown-out skin)
      - the score scales with edge density up to 0.15 → 1.0.
    """
    x, y, w, h, _ = marker
    gap = max(1, int(round(3 * scale_k)))
    sw = max(8, int(round(29 * scale_k)))
    sh = max(10, int(round(37 * scale_k)))
    cy = y + h // 2
    sx = x + w + gap
    sy = cy - sh // 2
    H, W = gray.shape[:2]
    if sx < 0 or sy < 0 or sx + sw > W or sy + sh > H:
        return 0.0
    e = edges[sy:sy + sh, sx:sx + sw]
    edge_frac = float(e.sum() / 255.0) / max(1, sw * sh)
    if edge_frac < 0.03 or edge_frac > 0.45:
        return 0.0
    if float(gray[sy:sy + sh, sx:sx + sw].mean()) > 200:
        return 0.0
    return min(1.0, edge_frac / 0.15)


def best_panel(markers, icon_w, icon_h, img=None):
    """Pick the best 2-column anchor grid. Returns (col_a, col_b, score)
    or None if no plausible panel was found.

    When `img` is provided, candidate panels also receive a slot-evidence
    bonus: each selected marker contributes `[0,1]` based on whether an
    STO ability slot is visible to its right at the bible-predicted
    position. This breaks ties on collages where multiple loose markers
    compete with the real grid.
    """
    if len(markers) < 4:
        return None

    cx_arr = [m[0] + m[2] / 2 for m in markers]
    cy_arr = [m[1] + m[3] / 2 for m in markers]
    n = len(markers)

    pitch_y_candidates = [
        max(icon_h * 1.6, 24),
        max(icon_h * 2.0, 30),
        max(icon_h * 2.4, 36),
        max(icon_h * 3.0, 50),
        max(icon_h * 3.6, 70),
    ]

    if img is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 160)
    else:
        gray = edges = None

    x_tol = max(icon_w * 0.7, 8)

    best = None
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if abs(cy_arr[i] - cy_arr[j]) > max(icon_h * 0.7, 12):
                continue
            xa, xb = cx_arr[i], cx_arr[j]
            # Local anchor width: large markers anchor wider columns even
            # when the global icon_w estimate is small (e.g. a collage
            # where most markers are tiny but one cluster is 60-80 px).
            # x_tol stays based on global icon_w to keep column-membership
            # strict (otherwise noise markers drift in on wide anchors).
            local_w = max(icon_w, markers[i][2], markers[j][2])
            if xb <= xa + 3 * local_w:
                continue
            if xb - xa > 9 * local_w:
                continue

            for pitch_y in pitch_y_candidates:
                y_tol = pitch_y * 0.30
                anchor_y = (cy_arr[i] + cy_arr[j]) / 2
                col_a, col_b = [], []
                for k in range(n):
                    cx, cy = cx_arr[k], cy_arr[k]
                    r = round((cy - anchor_y) / pitch_y)
                    if abs(r) > 3:
                        continue
                    expected_y = anchor_y + r * pitch_y
                    if abs(cy - expected_y) > y_tol:
                        continue
                    if abs(cx - xa) <= x_tol:
                        col_a.append((r, markers[k]))
                    elif abs(cx - xb) <= x_tol:
                        col_b.append((r, markers[k]))

                def _keep_one(col):
                    seen: dict[int, tuple] = {}
                    for r, m in col:
                        if r not in seen:
                            seen[r] = m
                    return [m for _, m in sorted(seen.items())]
                col_a = _keep_one(col_a)
                col_b = _keep_one(col_b)

                if len(col_a) < 2 or len(col_b) < 1:
                    continue
                if len(col_a) > 3 or len(col_b) > 3:
                    continue

                aligned = 0
                yas = [m[1] + m[3] / 2 for m in col_a]
                ybs = [m[1] + m[3] / 2 for m in col_b]
                for yb in ybs:
                    if any(abs(yb - ya) < y_tol for ya in yas):
                        aligned += 1
                if aligned < 1:
                    continue

                n_total = len(col_a) + len(col_b)
                canon_table = {5: 1.5, 4: 1.0, 3: 0.4, 2: 0.0, 6: 0.6}
                canon = canon_table.get(n_total, 0.0)

                codes = {m[4] for m in col_a} | {m[4] for m in col_b}
                div = 0.6 if len(codes) >= 2 else -0.8
                layout = 0.3 if len(col_a) >= len(col_b) else 0.0

                pitch_score = 0.0
                if len(col_a) >= 2:
                    ys = sorted(yas)
                    diffs = [ys[k + 1] - ys[k] for k in range(len(ys) - 1)]
                    if len(diffs) >= 2:
                        m = st.mean(diffs)
                        d = st.stdev(diffs)
                        pitch_score = 0.4 * (1.0 - min(d / max(m, 1), 1.0))

                slot_score = 0.0
                if gray is not None:
                    sel = col_a + col_b
                    med_w = sorted(m[2] for m in sel)[len(sel) // 2]
                    sk = med_w / _BIBLE_MARKER_W
                    ev = [_slot_evidence(gray, edges, m, sk) for m in sel]
                    avg_ev = sum(ev) / max(1, len(ev))
                    slot_score = 0.8 * avg_ev

                score = (0.6 * aligned + canon + div + layout
                         + pitch_score + slot_score)
                if best is None or score > best[2]:
                    best = (col_a, col_b, score)
    return best


# ---------------------------------------------------------------------------
# Bible-driven slot projection
# ---------------------------------------------------------------------------

def project_seat_slots(panel, n_abilities=4, hsv=None):
    """Project 4 ability-icon bboxes per detected seat marker.

    Geometry comes from the bible (panel-internal pixels). Detection
    contributes a single scale factor `k = full_marker_w / 29`, so
    every grid distance scales coherently — slot 4 cannot drift away
    from slot 1 just because `med_marker_w` was a pixel low.

    Float-domain math: bible distances are kept as floats, the only
    rounding happens once per slot at the final pixel position. This
    eliminates cumulative error from `i * round(stride)` and the
    visual gap alternation that comes from independently rounding
    slot_w vs stride.

    Returns list of (seat_idx, slot_idx, x, y, w, h, seat_code).
    """
    if panel is None:
        return []
    a, b, _score = panel
    all_m = a + b
    if not all_m:
        return []

    if hsv is not None:
        widths = sorted(full_bar_extent(hsv, m)[0] for m in all_m)
    else:
        widths = sorted(m[2] for m in all_m)
    det_w = widths[len(widths) // 2]
    k = det_w / _BIBLE_MARKER_W

    stride_f    = k * _BIBLE_STRIDE_X
    gap_first_f = k * _BIBLE_GAP_FIRST
    ab_w        = max(1, int(round(k * _BIBLE_SLOT_W)))

    med_h    = sorted(m[3] for m in all_m)[len(all_m) // 2]
    ab_h     = max(1, int(round(med_h / 0.63)))
    gap_y    = int(round(med_h * 0.20))

    def col_anchor(col):
        if hsv is None:
            return max(m[0] + m[2] for m in col)
        return max(m[0] + full_bar_extent(hsv, m)[0] for m in col)
    a_anchor = col_anchor(a) if a else 0
    b_anchor = col_anchor(b) if b else 0

    out = []
    for seat_idx, m in enumerate(all_m):
        mx, my, mw, mh, code = m
        slot_y = my - ab_h - gap_y
        anchor = a_anchor if seat_idx < len(a) else b_anchor
        slot_x0_f = anchor + gap_first_f
        for k_idx in range(n_abilities):
            x = int(round(slot_x0_f + k_idx * stride_f))
            out.append((seat_idx, k_idx, x, slot_y, ab_w, ab_h, code))
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

# Marker codes split by panel family. Ground markers are dark-brown
# tabs (code 'G'); the remaining profession codes are space seats.
# Each family forms its own 2-col grid — a screen may show both
# (Space Stations + Standard Away Team), so detect_panel runs
# best_panel once per family and merges the results.
_SPACE_CODES = frozenset({'T', 'E', 'S', 'U'})
_GROUND_CODES = frozenset({'G'})


def detect_panel(img: np.ndarray) -> Optional[dict]:
    """Detect BOFF panel(s) in `img`. Returns None when no plausible
    panel is found.

    Markers are grouped by family (space: T/E/S/U; ground: G) and
    `best_panel` runs per family, so screens with both a Space Stations
    panel and a Standard Away Team panel emit seats from both in a
    single pass. Seats from each panel stay contiguous in `col_a`/
    `col_b` so that `seat_idx` in `slots` still indexes into
    `col_a + col_b`.

    Output dict:
      'col_a':   list[(x, y, w, h, code, spec_code | None)] — left
                 column markers concatenated panel-by-panel.
      'col_b':   list[(x, y, w, h, code, spec_code | None)] — right
                 column markers concatenated panel-by-panel.
      'score':   float — max RANSAC score across detected panels
      'seats':   list[(side, mx, my, mw, mh, seat_code, spec_code | None)]
      'slots':   list[(seat_idx, slot_idx, x, y, w, h, seat_code)]
                 — `seat_idx` indexes into `col_a + col_b`.
    """
    if img is None or img.size == 0:
        return None

    icon_w, icon_h = estimate_icon_dims(img)
    markers = detect_markers(img, icon_w, icon_h)
    if len(markers) < 3:
        return None

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    groups = [
        ('space',  [m for m in markers if m[4] in _SPACE_CODES]),
        ('ground', [m for m in markers if m[4] in _GROUND_CODES]),
    ]

    # Build per-family panels first so we can lay out global indices
    # before concatenating the column lists. Refine icon dims per family
    # so a space panel and a ground panel pasted into the same screen at
    # different scales cannot pollute each other's geometry hint.
    detected: list[tuple[list, list, float]] = []
    for _gname, gm in groups:
        if len(gm) < 3:
            continue
        fam_w, fam_h = _refine_dims_from_markers(gm, icon_w, icon_h)
        p = best_panel(gm, fam_w, fam_h, img=img)
        if p is None:
            continue
        detected.append(p)
    if not detected:
        return None

    n_a_total = sum(len(p[0]) for p in detected)

    col_a_all: list[tuple] = []
    col_b_all: list[tuple] = []
    slots_all: list[tuple] = []
    score_max = 0.0
    a_cursor = 0
    b_cursor = n_a_total
    for (a, b, score) in detected:
        score_max = max(score_max, float(score))
        a6 = [(mx, my, mw, mh, c, classify_stripe(hsv, (mx, my, mw, mh, c))[0])
              for (mx, my, mw, mh, c) in a]
        b6 = [(mx, my, mw, mh, c, classify_stripe(hsv, (mx, my, mw, mh, c))[0])
              for (mx, my, mw, mh, c) in b]
        col_a_all.extend(a6)
        col_b_all.extend(b6)
        # Local seat_idx convention from project_seat_slots: 0..len(a)-1
        # references `a`, len(a)..len(a)+len(b)-1 references `b`.
        # Remap each to the global col_a+col_b layout.
        len_a = len(a)
        for (lsi, slot_idx, x, y, w, h, code) in project_seat_slots(
                (a, b, score), hsv=hsv):
            if lsi < len_a:
                gsi = a_cursor + lsi
            else:
                gsi = b_cursor + (lsi - len_a)
            slots_all.append((gsi, slot_idx, x, y, w, h, code))
        a_cursor += len_a
        b_cursor += len(b)

    seats = (
        [('L', mx, my, mw, mh, c, sp) for (mx, my, mw, mh, c, sp) in col_a_all]
        + [('R', mx, my, mw, mh, c, sp) for (mx, my, mw, mh, c, sp) in col_b_all]
    )
    return {
        'col_a':  col_a_all,
        'col_b':  col_b_all,
        'score':  score_max,
        'seats':  seats,
        'slots':  slots_all,
    }
