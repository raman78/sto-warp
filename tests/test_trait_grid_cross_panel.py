"""A trait section must be emitted by ONE panel only (trait_grid).

`detect_traits` used to deduplicate a section within a panel and then
concatenate across panels, so a second grid misclassified as an existing
section doubled that section (e.g. 10 "Space Reputation" where the game
allows 5). The winner is now picked across all panels by recognition score.

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

TRAIT = 'Test Space Trait'
BRIGHT, DIM = 255, 170


class _Cache:
    """Minimal app_cache: one personal space trait, no starship traits."""
    traits = {'space': {'personal': {TRAIT: {}}, 'rep': {}, 'active_rep': {}},
              'ground': {'personal': {}, 'rep': {}, 'active_rep': {}}}
    starship_traits: dict = {}


class _Matcher:
    """Both panels classify to the same section; the bright one is confident."""
    def classify_patch(self, patch):
        conf = 0.90 if patch.mean() > (BRIGHT + DIM) / 2 else 0.40
        return TRAIT, conf


def _row(img, y, x0, value, n=5, dx=42, w=36, h=48):
    for i in range(n):
        x = x0 + i * dx
        img[y:y + h, x:x + w] = value


def _detect(img):
    return tg.detect_traits(img, _Matcher(), _Cache(),
                            build_type='SPACE_TRAITS')


def test_single_panel_is_emitted_whole():
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    _row(img, 40, 20, BRIGHT)
    res = _detect(img)
    assert list(res) == ['Personal Space Traits']
    assert len(res['Personal Space Traits']) == 5


def test_second_panel_with_same_section_does_not_double_it():
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    _row(img, 40, 20, BRIGHT)     # real panel
    _row(img, 250, 340, DIM)      # decoy grid, same section, weaker score

    res = _detect(img)
    boxes = res['Personal Space Traits']
    assert len(boxes) == 5, boxes


def test_the_higher_scoring_panel_wins():
    """The kept bboxes must be the confident panel's, not merely the first."""
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    _row(img, 40, 20, DIM)        # weak panel first in Y
    _row(img, 250, 340, BRIGHT)   # confident panel

    boxes = _detect(img)['Personal Space Traits']
    assert len(boxes) == 5
    # Bright row sits at y=250 and starts at x=340.
    assert all(b[1] > 200 for b in boxes), boxes
    assert min(b[0] for b in boxes) > 300, boxes
