"""EQ row→slot labelling must not lose a row (regression).

A SPACE_EQ screenshot lost its Warp Core entirely — no bbox to confirm, the
user had to draw it by hand. Three independent defects lined up:

1. `_collect_single_hits` collapsed keyword hits to one per row name by OCR
   confidence, before the x-column filter ran. A tooltip word elsewhere on
   screen ("Field" fuzzy-matching the "Shields" keyword) evicted the genuine
   in-column label, and was then dropped by the column filter itself — so
   the Shields row had no anchor at all.
2. `ShipDB._entry_to_profile` used bare truthiness on cargo's *string*
   fields, and `bool('0')` is True → phantom Sec-Def / Experimental slots,
   which shift the positional row order.
3. The positional fallback could hand a row the name of a row that OCR had
   already anchored, and `result[slot] = bboxes` silently replaced it.
"""
from __future__ import annotations

import numpy as np
import pytest


def _has_cv2() -> bool:
    try:
        import cv2  # noqa: F401
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_cv2(), reason='opencv not installed')

from warp.recognition import eq_geometry as eg          # noqa: E402
from warp.recognition import layout_detector as ld      # noqa: E402


# ── 1. hit collection keeps every candidate ───────────────────────────────
def _tok(name, cy, x1, conf, role='single'):
    return {'kw_role': role, 'kw_name': name, 'text': name, 'conf': conf,
            'cx': x1 - 20, 'cy': cy, 'x0': x1 - 60, 'y0': cy - 8,
            'x1': x1, 'y1': cy + 8}


def test_single_hits_are_not_collapsed_by_confidence():
    """The in-column label must survive an equally-confident stray hit."""
    toks = [
        _tok('Shields', 884, 951, 1.00),   # tooltip word, off panel, first
        _tok('Shields', 295, 569, 1.00),   # the real label, in the column
    ]
    hits = eg._collect_single_hits(toks)
    assert len(hits) == 2
    assert 295 in [h['cy'] for h in hits]


def test_column_filter_keeps_the_in_column_label():
    """Geometry decides, not confidence: the larger x-cluster wins."""
    toks = [
        _tok('Shields', 884, 951, 1.00),
        _tok('Shields', 295, 569, 1.00),
        _tok('Deflector', 119, 569, 1.00),
        _tok('Engines', 179, 569, 1.00),
        _tok('Devices', 413, 569, 1.00),
    ]
    hits = eg._collect_single_hits(toks)
    column = max(eg._cluster_by_x1(hits), key=len)
    cys = sorted(h['cy'] for h in column if h['row'] == 'Shields')
    assert cys == [295], cys


# ── 2. cargo's string fields ──────────────────────────────────────────────
def _profile(**cargo):
    from warp.warp_importer import ShipDB
    entry = {'fore': '5', 'aft': '3', 'devices': '3', 'consoleseng': '4',
             'consolessci': '2', 'consolestac': '5', **cargo}
    return ShipDB._entry_to_profile(object.__new__(ShipDB), entry, '')


def test_string_zero_does_not_create_a_slot():
    p = _profile(secdeflector='0', experimental='0')
    assert p['Sec-Def'] == 0
    assert p['Experimental'] == 0


def test_string_one_does_create_a_slot():
    p = _profile(secdeflector='1', experimental='1')
    assert p['Sec-Def'] == 1
    assert p['Experimental'] == 1


# ── 3. positional guess must not evict an OCR-anchored row ────────────────
def test_positional_guess_never_steals_an_ocr_anchored_slot(monkeypatch):
    """Rows 0-3 and 5+ are OCR-anchored; row 4's label was missed.

    With a phantom Sec-Def in the profile the positional order is shifted,
    so row 4's guess lands on 'Warp Core' — already anchored on row 3. The
    guess must be dropped, leaving row 3's Warp Core intact.
    """
    cys = [62, 119, 179, 237, 295, 354]
    geom = eg.EQGeometry(
        panel_x_start=573, panel_right=828, final_dx=42.5, row_pitch=58,
        row_cys=cys, mode='v8',
        eq_label_cys={
            eg.STD_ORDER['Fore Weapons']: 62,
            eg.STD_ORDER['Deflector']:    119,
            eg.STD_ORDER['Engines']:      179,
            eg.STD_ORDER['Warp Core']:    237,
            # 295 (Shields) deliberately missing
            eg.STD_ORDER['Aft Weapons']:  354,
        },
    )
    # `**_` because the detector now hands the shared OCR tokens down as a
    # keyword; what this test asserts is unchanged by that.
    monkeypatch.setattr(ld, 'detect_eq_geometry', lambda img, **_: geom)

    img = np.zeros((700, 900, 3), dtype=np.uint8)
    profile = {'Fore Weapons': 5, 'Deflector': 1, 'Sec-Def': 1, 'Engines': 1,
               'Warp Core': 1, 'Shield': 1, 'Aft Weapons': 3}
    det = ld.LayoutDetector()
    out = det._detect_via_pixel_analysis(
        img, ld.SPACE_SLOT_ORDER_STANDARD, profile)

    assert 'Warp Core' in out
    wc_cy = out['Warp Core'][0][1] + out['Warp Core'][0][3] // 2
    assert abs(wc_cy - 237) <= 2, out['Warp Core']
