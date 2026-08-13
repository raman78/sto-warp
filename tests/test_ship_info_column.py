"""The ship-info column window must not grow with the anchor token's width.

Regression: OCR returned a whole class line as one 404 px token, the window
became ±808 px (81 % of a 1544 px screenshot) and stopped rejecting anything.
A token from the far-right traits legend merged into ship_type, dragging its
bbox across the equipment column — 1255 px wide — and pulling the fuzzy class
lookup towards the wrong ship.

Tokens below are the real OCR output for that screenshot; feeding them through
the extractor's scan cache exercises the whole path without running OCR.
"""
from __future__ import annotations

import pytest

pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from warp.recognition.text_extractor import TextExtractor


def _tok(text: str, x0: int, y0: int, x1: int, y1: int, conf: float) -> dict:
    return {
        'text': text, 'low': text.lower(), 'conf': conf,
        'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1,
        'cx': (x0 + x1) // 2, 'cy': (y0 + y1) // 2,
        'w': x1 - x0, 'h': y1 - y0,
    }


# Top band of Ragna/Untitled.png (1544x823), verbatim from easyocr.
_TOKENS = [
    _tok('r.R.W', 46, 2, 126, 28, 0.44),
    _tok('NOCTILUM', 138, 2, 264, 28, 1.00),
    _tok('Active Space Duty', 1219, 7, 1357, 23, 0.96),
    _tok('Fore', 593, 9, 629, 25, 1.00),
    _tok('Personal Space Traits', 936, 10, 1152, 38, 0.97),
    _tok('Thebe officers', 1219, 21, 1301, 37, 0.66),
    _tok('active in Space', 1369, 23, 1455, 37, 0.86),
    _tok('Weapons', 558, 25, 629, 46, 1.00),
    _tok('polare', 1304, 26, 1344, 34, 0.28),
    _tok('Legendary Scimitar Intel Dreadnought Warbird [T6-X2]',
         46, 28, 450, 52, 0.73),
    _tok('Tactical', 1267, 43, 1331, 59, 1.00),
    _tok('Collapse AII', 1439, 45, 1523, 61, 0.35),
    _tok('Deflector', 559, 69, 629, 85, 0.93),
    _tok('@]@ FFC', 1469, 73, 1529, 91, 0.20),
    _tok('Prince Darrin Senay', 1283, 74, 1421, 92, 0.97),
]

_EQ_COLUMN_X = 558      # leftmost equipment label ('Weapons') on this screenshot


@pytest.fixture
def info():
    img = np.zeros((823, 1544, 3), dtype=np.uint8)
    te = TextExtractor()
    te._scan_cache_key = id(img)          # feed tokens, skip OCR
    te._scan_cache_tokens = _TOKENS
    return te.extract_ship_info(img)


def test_far_token_does_not_join_the_ship_type(info):
    """'Thebe officers' sits at x=1219, in the traits legend — not the ship."""
    assert info['ship_type'] == 'Legendary Scimitar Intel Dreadnought Warbird'


def test_ship_type_bbox_stays_out_of_the_equipment_column(info):
    x, _y, w, _h = info['ship_type_bbox']
    assert x + w <= _EQ_COLUMN_X, 'type bbox runs into the equipment column'


def test_ship_name_and_tier_still_read(info):
    """The cap must not cost the fields that were already correct."""
    assert info['ship_tier'] == 'T6-X2'
    assert 'NOCTILUM' in info['ship_name']
