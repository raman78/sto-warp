"""A tier badge that OCR cut in half must still be read, and must be boxed.

Regression: `[T6-X2]` came back as two tokens, `'[Te-'` and `'X2]'`. Neither
carries a closed bracket, so the bracket anchors saw nothing; neither satisfies
RE_TIER_LOOSE ('Te-' has no digit), so the loose anchor saw nothing either. The
screenshot fell through to the anchorless path, where ship_type became whichever
single OCR token won the ShipDB lookup — a box covering part of the class line —
and the tier, having no box of its own, borrowed that same wrong one.

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


# Top band of Screeny4/l1.png (694x622), verbatim from easyocr.
# The class line is 'Terran Lexington Dreadnought Cruiser [T6-X2]'; OCR dropped
# the leading 'L' and cut the badge between '[Te-' and 'X2]'.
_TOKENS = [
    _tok('1.5.5. Pure Immersion', 36, 8, 294, 34, 0.65),
    _tok('Fore', 431, 17, 467, 33, 1.00),
    _tok('[Te-', 318, 30, 354, 56, 0.27),
    _tok('Weapons', 396, 32, 467, 52, 1.00),
    _tok('Terran', 36, 33, 94, 57, 1.00),
    _tok('exington Dreadnought Cruiser', 103, 37, 323, 55, 0.79),
    _tok('X2]', 351, 37, 383, 53, 0.49),
    _tok('(NCC-93719)', 37, 51, 141, 71, 0.68),
    _tok('Deflector', 397, 77, 467, 93, 1.00),
]

_BADGE_X0, _BADGE_X1 = 318, 383      # where '[T6-X2]' actually sits
_CLASS_X0, _CLASS_X1 = 36, 323       # where the class line actually sits


@pytest.fixture
def info():
    img = np.zeros((622, 694, 3), dtype=np.uint8)
    te = TextExtractor()
    te._scan_cache_key = id(img)          # feed tokens, skip OCR
    te._scan_cache_tokens = _TOKENS
    return te.extract_ship_info(img)


def test_split_badge_is_read_as_a_tier(info):
    assert info['ship_tier'] == 'T6-X2'


def test_tier_bbox_covers_the_badge(info):
    x, _y, w, _h = info['ship_tier_bbox']
    assert (x, x + w) == (_BADGE_X0, _BADGE_X1)


def test_type_bbox_covers_the_whole_class_line(info):
    """Both halves — the box used to start at 103, cutting off 'Terran'."""
    x, _y, w, _h = info['ship_type_bbox']
    assert (x, x + w) == (_CLASS_X0, _CLASS_X1)


def test_class_line_is_read_whole(info):
    assert info['ship_type'] == 'Terran exington Dreadnought Cruiser'


def test_row_above_the_badge_does_not_leak_in(info):
    """'1.5.5. Pure Immersion' is an FPS watermark and 'Fore' a column header;
    neither is blacklisted, so only the same-row rule keeps them out."""
    assert 'Immersion' not in info['ship_type']
    assert 'Fore' not in info['ship_type']


def test_no_anchorless_fallback_needed(info):
    """A real anchor fired, so the ShipDB rescue candidates stay empty."""
    assert info['anchorless_candidates'] == []


def test_hud_label_right_of_the_badge_is_not_joined_to_it(info):
    """'Weapons' sits 13 px right of 'X2]' — outside the fragment gap cap."""
    x, _y, w, _h = info['ship_tier_bbox']
    assert x + w < 396
