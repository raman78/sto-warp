"""Tests for the virtual-label poison guard `_virtual_crop_looks_real`.

The guard decides whether a crop labeled `__empty__` / `__inactive__` is in
fact a real, colourful icon (mislabeled poison) or a genuine dim empty/inactive
slot. Its threshold was raised 0.07 → 0.15 on 2026-07-17 after a visual review
of community-mirror crops showed genuine empty/inactive BOFF slots reach
~12% bright/rich, well above the old gate. These tests lock that boundary.
"""
from __future__ import annotations

import numpy as np
import pytest


def _has_cv2() -> bool:
    try:
        import cv2  # noqa: F401
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(not _has_cv2(), reason='opencv not installed')


def _crop_with_fraction(frac: float, size: int = 40) -> np.ndarray:
    """Return a BGR crop where `frac` of pixels are bright+saturated red
    (V=255, S=255 → counts as both bright and rich) and the rest are dim
    grey (V=30, S=0). So bright ≈ rich ≈ frac."""
    total = size * size
    n_bright = round(frac * total)
    flat = np.full((total, 3), 30, dtype=np.uint8)   # dim grey
    flat[:n_bright] = (0, 0, 255)                     # pure red in BGR
    return flat.reshape(size, size, 3)


def test_thresholds_are_calibrated_to_0_15():
    from warp.recognition import icon_matcher as im
    assert im.VIRTUAL_SEED_BRIGHT_RATIO == 0.15
    assert im.VIRTUAL_SEED_RICH_RATIO == 0.15


def test_genuine_dim_slot_is_not_flagged():
    """A uniformly dim crop (a real empty/inactive slot) is never poison."""
    from warp.recognition.icon_matcher import _virtual_crop_looks_real
    dim = np.full((40, 40, 3), 30, dtype=np.uint8)
    assert _virtual_crop_looks_real(dim) is False


def test_boff_level_tint_is_not_flagged():
    """~12% bright/rich (genuine empty/inactive BOFF navy tint) stays under
    the 0.15 gate — this is the false-positive class the recalibration fixed."""
    from warp.recognition.icon_matcher import _virtual_crop_looks_real
    assert _virtual_crop_looks_real(_crop_with_fraction(0.12)) is False


def test_real_icon_is_flagged_as_poison():
    """A clearly bright + colour-rich crop (a real mislabeled icon) is poison."""
    from warp.recognition.icon_matcher import _virtual_crop_looks_real
    assert _virtual_crop_looks_real(_crop_with_fraction(0.40)) is True


def test_boundary_just_above_threshold_is_flagged():
    """Just over 0.15 on both axes crosses into poison territory."""
    from warp.recognition.icon_matcher import _virtual_crop_looks_real
    assert _virtual_crop_looks_real(_crop_with_fraction(0.20)) is True
