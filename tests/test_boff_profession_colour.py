"""The BOFF profession classifier must read the icon, not the slot chrome.

`_classify_boff_profession` samples a ring near the edge of the crop, which is
where a slot bbox also contains UI chrome: the slot border, tinted by the
officer's name-plate colour rather than by the ability.

On kvort_elp.png that border is mint green. Three Science abilities each showed
about a third of their sampled ring in the H48-72 "green" band against a blue
majority twice its size, so all three classified as Miracle Worker. The seat is
Universal, so the marker carries no profession and the pipeline falls back to
voting over those icons — 3/3 for Miracle Worker. Every ability in the seat was
then looked up in the wrong candidate list, matched at 0.3-0.4 confidence, and
the seat was reported as Intelligence and Pilot.

Measured over 1071 confirmed BOFF crops from 79 screenshots, insetting past the
chrome takes Science from 76.5% to 99.4% and the classifier overall from 85.1%
to 94.2%.

Run standalone:
    python -m pytest tests/test_boff_profession_colour.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip('cv2')


def _icon(body_bgr, frame_bgr=None, size=52, frame_px=3):
    """A crop the shape of a BOFF ability slot: flat art, optional UI frame."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:, :] = body_bgr
    if frame_bgr is not None:
        img[:frame_px, :] = frame_bgr
        img[-frame_px:, :] = frame_bgr
        img[:, :frame_px] = frame_bgr
        img[:, -frame_px:] = frame_bgr
    return img


# Science ability art is blue; the slot chrome on the reported screenshot is
# mint green, which is what the classifier used to answer with.
_SCIENCE_BLUE = (200, 120, 30)      # BGR — H≈105
_MINT_CHROME = (150, 220, 130)      # BGR — the green slot border


def _classify(img):
    from warp.recognition.layout_detector import LayoutDetector

    return LayoutDetector._classify_boff_profession(img)


def test_a_green_slot_border_does_not_become_miracle_worker():
    """The reported failure, reduced to its cause."""
    assert _classify(_icon(_SCIENCE_BLUE, _MINT_CHROME)) != 'miracle worker'


def test_a_blue_icon_still_reads_as_science_through_the_chrome():
    assert _classify(_icon(_SCIENCE_BLUE, _MINT_CHROME)) == 'science'


def test_a_blue_icon_with_no_chrome_is_unaffected():
    """Insetting must not cost anything when the crop caught no border — 241 of
    the 1071 measured crops are in that state, and they improved too."""
    assert _classify(_icon(_SCIENCE_BLUE)) == 'science'


@pytest.mark.parametrize('sides', [
    ('top',), ('left',), ('top', 'left'), ('top', 'bottom', 'left'),
])
def test_chrome_on_only_some_edges_is_still_ignored(sides):
    """A bbox often clips the border on one or two sides only: 470 of the 1071
    measured crops caught it on 1-3 edges, and those were the worst served."""
    img = _icon(_SCIENCE_BLUE)
    f = 3
    if 'top' in sides:
        img[:f, :] = _MINT_CHROME
    if 'bottom' in sides:
        img[-f:, :] = _MINT_CHROME
    if 'left' in sides:
        img[:, :f] = _MINT_CHROME
    if 'right' in sides:
        img[:, -f:] = _MINT_CHROME

    assert _classify(img) == 'science'


def test_a_genuinely_green_icon_is_still_miracle_worker():
    """The inset removes chrome, not signal: a real lime accent fills the icon
    rather than tracing its edge."""
    lime = (60, 220, 130)           # BGR — H≈55, inside the MW band

    assert _classify(_icon(lime)) == 'miracle worker'


def test_a_red_icon_is_still_tactical():
    assert _classify(_icon((40, 40, 220), _MINT_CHROME)) == 'tactical'
