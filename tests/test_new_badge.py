"""The game's yellow 'NEW' ribbon is chrome, not slot content.

Cells are synthesised rather than taken from the community mirror so the
test says what the rule is and runs anywhere. The shapes match what was
measured on the mirror 2026-09-18: amber at hue ~22, a band from the top
edge ending at about a third of the cell's height, with the word NEW drawn
black on it so the middle rows are only partly amber.
"""
import numpy as np
import pytest

cv2 = pytest.importorskip('cv2')

from warp.recognition.layout_detector import LayoutDetector  # noqa: E402
from warp.recognition.icon_matcher import _virtual_crop_looks_real  # noqa: E402

AMBER = (30, 190, 245)      # BGR — hue 22, saturated, bright


def _cell(h=45, w=35, ribbon_frac=0.33, lettering=True):
    """An empty (near-black) slot carrying a 'NEW' ribbon."""
    img = np.full((h, w, 3), 8, dtype=np.uint8)
    rows = int(h * ribbon_frac)
    img[:rows, :] = AMBER
    if lettering:
        # Black letters across the middle of the ribbon, leaving roughly a
        # third of each row amber — the thin rows that defeat a test asking
        # for a solid full-width band.
        img[rows // 4:rows - rows // 4, int(w * 0.15):int(w * 0.80)] = 8
    return img


def test_ribbon_is_found_on_an_empty_slot():
    img = _cell()
    rows = LayoutDetector._new_badge_rows(img)
    assert rows == pytest.approx(int(45 * 0.33), abs=2)


def test_ribboned_empty_slot_reads_as_empty_not_active():
    assert LayoutDetector._classify_cell(_cell()) == 'empty'


def test_ribboned_empty_slot_is_not_poison():
    """The crop a user correctly labelled `__empty__` must still seed."""
    assert _virtual_crop_looks_real(_cell()) is False


def test_amber_icon_is_not_mistaken_for_a_ribbon():
    """An icon that is amber all the way down never terminates, so the
    band test refuses it and the cell keeps being treated as an icon."""
    img = np.full((45, 35, 3), AMBER, dtype=np.uint8)
    assert LayoutDetector._new_badge_rows(img) == 0
    assert LayoutDetector._classify_cell(img) == 'active'


def test_an_amber_sliver_down_one_edge_is_not_a_ribbon():
    """A cell box that sits a little off catches the edge of the icon next
    door. That stripe is amber and reaches the top row, but it is a stripe,
    not a band — on `image-9542d3c56fb6c860.png` cutting it turned a real
    Personal Space Trait into an inactive cell.
    """
    img = np.full((45, 35, 3), 8, dtype=np.uint8)
    # Right-hand quarter, and only as far down as a ribbon would reach — so
    # the run does terminate, and the solid-strip rule is the only thing
    # left that can refuse it.
    img[:int(45 * 0.33), int(35 * 0.75):] = AMBER
    assert LayoutDetector._new_badge_rows(img) == 0


def test_a_real_icon_under_a_ribbon_still_reads_as_active():
    img = _cell()
    rows = int(45 * 0.33)
    # A bright, colour-rich icon in the rest of the cell.
    img[rows:, :] = (200, 60, 220)
    assert LayoutDetector._new_badge_rows(img) > 0
    assert LayoutDetector._classify_cell(img) == 'active'
