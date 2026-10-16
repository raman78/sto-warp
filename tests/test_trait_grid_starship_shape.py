"""The 5+N Starship Traits shape must not outrank what its icons say (trait_grid).

A full row with one or two icons under columns 0-1 is what a Starship Traits
section looks like, and `detect_traits` lets that shape win outright. On
`image-decdcfd6ab849bba.png` the Space Reputation row with the captain's rank
insignia under its first column had the same shape, beat the real Starship
Traits row, and the build got reputation traits as starship traits.

Uses a stub matcher/cache — no torch, no icon DB.
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

from warp.recognition import trait_grid as tg  # noqa: E402

SHIP, REP = 'Test Starship Trait', 'Test Rep Trait'
V_SHIP, V_REP, V_BADGE = 250, 200, 150


class _Cache:
    traits = {'space': {'personal': {}, 'rep': {REP: {}}, 'active_rep': {}},
              'ground': {'personal': {}, 'rep': {}, 'active_rep': {}}}
    starship_traits = {SHIP: {}}


class _Matcher:
    """Names an icon by its brightness; the insignia is nothing it knows."""
    def classify_patch(self, patch):
        m = patch.mean()
        if m > (V_SHIP + V_REP) / 2:
            return SHIP, 0.95
        if m > (V_REP + V_BADGE) / 2:
            return REP, 0.95
        return '', 0.0


def _cell(img, x, y, value, w=36, h=48):
    img[y:y + h, x:x + w] = value


def _row(img, y, x0, value, n=5, dx=42):
    for i in range(n):
        _cell(img, x0 + i * dx, y, value)


def _screen():
    """A bare 5-row of starship traits above a reputation row with a badge
    under column 0, 39 px down — inside the ship-name divider gap."""
    img = np.zeros((420, 400, 3), dtype=np.uint8)
    _row(img, 40, 20, V_SHIP)
    _row(img, 200, 20, V_REP)
    _cell(img, 20, 200 + 48 + 39, V_BADGE)
    return img


def _detect(img):
    return tg.detect_traits(img, _Matcher(), _Cache(),
                            build_type='SPACE_MIXED')


def test_the_starship_section_is_the_row_whose_icons_are_starship_traits():
    boxes = _detect(_screen())['Starship Traits']
    assert len(boxes) == 5, boxes
    assert all(b[1] < 120 for b in boxes), boxes


def test_the_reputation_row_keeps_its_own_section_without_the_badge():
    boxes = _detect(_screen())['Space Reputation']
    assert len(boxes) == 5, boxes
    assert all(150 < b[1] < 260 for b in boxes), boxes


def test_a_real_5_plus_2_starship_section_still_wins_on_its_shape():
    """The shortcut stays for the case it was made for: the full row reads as
    starship traits, so the overflow belongs to it."""
    img = np.zeros((420, 400, 3), dtype=np.uint8)
    _row(img, 40, 20, V_SHIP)
    _cell(img, 20, 40 + 48 + 39, V_SHIP)
    _cell(img, 62, 40 + 48 + 39, V_SHIP)
    boxes = _detect(img)['Starship Traits']
    assert len(boxes) == 7, boxes
