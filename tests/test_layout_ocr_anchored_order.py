"""The OCR-anchored fallback places an unread row where the game draws it.

A row whose label OCR missed is interpolated between the labelled rows around
it, in the sequence the ship's profile implies. That sequence put Hangars
right after Aft Weapons; the game draws Hangars last. So an unread Hangars
label on a carrier was placed between Aft and Devices, where no row exists.

Offline: synthetic panel, the section labels stubbed, no OCR.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip('cv2')

from warp.recognition import layout_detector as ld

LABEL_CX = 60
ROWS = {  # every label read except Hangars
    'Fore Weapons': 50, 'Deflector': 100, 'Engines': 150, 'Warp Core': 200,
    'Shield': 250, 'Aft Weapons': 300, 'Devices': 350,
    'Engineering Consoles': 400, 'Science Consoles': 450, 'Tactical Consoles': 500,
}


@pytest.fixture
def detector(tmp_path, monkeypatch):
    from warp import userdata
    monkeypatch.setattr(userdata, 'training_data_dir', lambda: tmp_path)
    det = ld.LayoutDetector()
    monkeypatch.setattr(det, '_ocr_section_labels',
                        lambda img: {s: (LABEL_CX, cy) for s, cy in ROWS.items()})
    monkeypatch.setattr(det, '_get_eq_geometry', lambda img: None)
    return det


def _panel():
    img = np.zeros((650, 900, 3), np.uint8)
    for cy in list(ROWS.values()) + [550]:     # a bright cell per row, Hangars too
        img[cy - 15:cy + 15, 300:336] = 200
    return img


def test_an_unread_hangars_row_is_placed_below_the_consoles(detector):
    result = detector._detect_via_ocr_anchored(
        _panel(), 'SPACE_MIXED', ld.SPACE_SLOT_ORDER_CARRIER, {'Hangars': 2})
    hangar_cys = [y + h / 2 for (_, y, _, h) in result.get('Hangars', [])]
    assert hangar_cys, 'no Hangars row was placed at all'
    assert min(hangar_cys) > ROWS['Tactical Consoles']
