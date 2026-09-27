"""The equipment rows' order, and the row pitch read from their labels.

`STD_ORDER` held the order the slot types are *listed* in
docs/sto_slots_rules.md — weapons first, then core equipment — which put Aft
and Experimental second and third. The game draws them after Shield. The
index is also the number of rows between two labels when the row pitch is
read, so every pair spanning the misplaced rows was off; the median hid it on
the measured corpus.

Offline: synthetic image and hand-made OCR tokens, no reader.
"""
from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip('cv2')

from warp.recognition import eq_geometry as eg

SCREEN_ORDER = [
    'Fore Weapons', 'Deflector', 'Sec-Def', 'Engines', 'Warp Core', 'Shields',
    'Aft Weapons', 'Experimental', 'Devices', 'Universal Consoles',
    'Engineering Consoles', 'Science Consoles', 'Tactical Consoles', 'Hangars',
]


def test_rows_are_indexed_in_the_order_the_game_draws_them():
    assert sorted(eg.STD_ORDER, key=eg.STD_ORDER.get) == SCREEN_ORDER


def test_the_optional_rows_are_the_ones_a_ship_may_lack():
    assert eg.OPTIONAL_ROWS == {'Sec-Def', 'Aft Weapons', 'Experimental',
                                'Universal Consoles', 'Hangars'}


def test_the_german_labels_point_at_the_same_rows_as_the_english_ones():
    assert eg.GERMAN_ORDER['Heck Waffen'] == eg.STD_ORDER['Aft Weapons']
    assert eg.GERMAN_ORDER['Geraete'] == eg.STD_ORDER['Devices']


def test_rows_between_two_fixed_rows_are_counted_for_certain():
    # Engines → Shield: Warp Core between them, which every ship has
    assert eg._row_steps(eg.STD_ORDER['Engines'], eg.STD_ORDER['Shields']) == (2, True)


def test_an_optional_row_between_makes_the_count_uncertain():
    # Deflector → Engines: Sec-Def may or may not be drawn
    assert eg._row_steps(eg.STD_ORDER['Deflector'], eg.STD_ORDER['Engines']) == (1, False)


def _token(text, cy, x0=100, x1=200):
    return {'text': text, 'low': text.lower(), 'conf': 0.99,
            'cx': (x0 + x1) // 2, 'cy': cy, 'x0': x0, 'y0': cy - 8,
            'x1': x1, 'y1': cy + 8, 'w': x1 - x0, 'h': 16}


def _panel():
    """Grey label column, dark panel from x=250 — enough for the stripe scan."""
    img = np.full((720, 700, 3), 120, np.uint8)
    img[:, 250:] = 30
    return img


def test_the_pitch_comes_from_pairs_whose_row_count_is_certain():
    # Rows 50 px apart. Drawn: Deflector, Engines (no Sec-Def), Warp Core,
    # Shield, Aft, Experimental, Devices, Universal, Eng, Sci, Tac, Hangars.
    # Only Engines → Singularity (Warp Core) is a certain pair; the others
    # span optional rows that ARE drawn here, so counting them absent would
    # read 100 px and 62.5 px and pull the median off 50.
    tokens = [_token('Deflector', 100), _token('Engines', 150),
              _token('Singularity', 200), _token('Devices', 400),
              _token('Hangars', 650)]
    geom = eg.detect_eq_geometry(_panel(), ocr_tokens=tokens)
    assert geom is not None
    assert geom.row_pitch == 50
