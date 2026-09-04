"""Inside the navy window, brightness variance separates the three states.

A BOFF panel draws a locked seat as a navy cell with an X across it. An empty
seat on the same panel is the navy background with nothing drawn on it, and a
real ability is an icon. All three are blue-saturated, so hue and saturation
cannot tell them apart — how much pattern the cell carries can.

Measured 2026-09-04 by running `_classify_cell` itself over 985 confirmed
virtual crops and a 1200-crop sample of real icons from the published mirror:

    flat navy fill   std_v < 5     empty
    navy + X         std_v 5-33    inactive
    real icon        std_v > 33    active   (dimmest seen: 33.4)

Before this split, `empty` scored 75.8% and 33 blank cells read as active —
the damaging direction, since a blank cell called active gets given an item
name. After: empty 87.1%, inactive 96.2% → 98.0%, active unchanged at 100%,
and the blank-called-active count halved to 17.

Run standalone:
    python -m pytest tests/test_classify_cell_navy.py -v
"""
from __future__ import annotations

import pytest

np = pytest.importorskip('numpy')
cv2 = pytest.importorskip('cv2')

from warp.recognition.layout_detector import LayoutDetector


def _navy_cell(std_v: float, mean_v: float = 70.0, size=(64, 49)) -> np.ndarray:
    """A cell in the navy window with a chosen brightness variance.

    Built in HSV so hue and saturation sit squarely inside the window under
    test and only `std_v` varies — the thing the rule keys on.
    """
    h, w = size
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[:, :, 0] = 110                       # navy hue
    hsv[:, :, 1] = 200                       # strongly saturated
    rng = np.random.default_rng(0)
    v = rng.normal(mean_v, std_v, (h, w))
    hsv[:, :, 2] = np.clip(v, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _measured(crop) -> tuple[float, float, float]:
    """The features the rule reads, so a test can assert on its own fixture."""
    ih, iw = crop.shape[:2]
    mx, my = max(1, int(iw * 0.20)), max(1, int(ih * 0.20))
    inner = crop[my:ih - my, mx:iw - mx]
    hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(float)
    return hsv[:, 2].std(), hsv[:, 1].mean(), hsv[:, 0].mean()


def test_the_fixture_really_lands_in_the_navy_window():
    """Guards the test itself: if the fixture drifted out of the window the
    assertions below would pass for the wrong reason."""
    _, mean_s, mean_h = _measured(_navy_cell(std_v=2))

    assert mean_s > 100
    assert 95 < mean_h < 130


# ── The three states ───────────────────────────────────────────────────────

def test_a_flat_navy_fill_is_empty():
    """The panel showing through an unused seat — no X drawn on it."""
    assert LayoutDetector._classify_cell(_navy_cell(std_v=2)) == 'empty'


def test_navy_with_an_x_is_inactive():
    assert LayoutDetector._classify_cell(_navy_cell(std_v=8)) == 'inactive'


def test_a_dim_navy_icon_is_still_active():
    """The dimmest real icon measured inside the window sits at std_v 33.4."""
    assert LayoutDetector._classify_cell(_navy_cell(std_v=40)) == 'active'


# ── The boundaries the measurement fixed ───────────────────────────────────

def test_an_x_at_the_old_upper_gate_no_longer_reads_as_active():
    """std_v 24 is the median of the 28 inactive cells the old gate of 20
    let through as active."""
    assert LayoutDetector._classify_cell(_navy_cell(std_v=24)) == 'inactive'


def test_the_upper_gate_still_clears_the_dimmest_real_icon():
    """30 was chosen over 33 to keep a margin below the 33.4 minimum rather
    than to score every last sample."""
    assert LayoutDetector._classify_cell(_navy_cell(std_v=34)) == 'active'


# ── Directions that must not regress ───────────────────────────────────────

def test_a_black_cell_outside_the_navy_window_is_still_empty():
    assert LayoutDetector._classify_cell(np.zeros((64, 49, 3), np.uint8)) == 'empty'


def test_a_bright_desaturated_icon_is_still_active():
    """The generic brightness gate, untouched by the navy split."""
    crop = np.full((64, 49, 3), 160, np.uint8)

    assert LayoutDetector._classify_cell(crop) == 'active'


def test_an_empty_crop_is_treated_as_active():
    """Unknown input must not be reported as a blank slot — that would delete
    a real item rather than fail visibly."""
    assert LayoutDetector._classify_cell(None) == 'active'
    assert LayoutDetector._classify_cell(np.zeros((0, 0, 3), np.uint8)) == 'active'
