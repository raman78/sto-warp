"""A MIXED screen with no equipment anchor keeps its BOFFs and traits, and
nothing sweeps the image with the matcher.

When every equipment strategy failed, the chain used to end in a full-image
scan: a sliding window over the whole screenshot, each window sent through
the icon matcher. Measured 2026-09-27 on the 98 annotated screenshots that
can reach it, with OCR switched off so no label existed anywhere: the final
layout was identical box for box with and without it, and it cost about 12 s
per screenshot it ran on. It was removed; the marker BOFFs and the trait
grid are what such a screen gets.

Offline: every strategy is stubbed, no model, no OCR.
"""
from __future__ import annotations

import types

import numpy as np
import pytest

from warp.recognition import layout_detector as ld

BOFF_SEAT = 'Boff Seat L[T]_100'
TRAITS = 'Personal Space Traits'


class _Matcher:
    """Any call means something is sweeping the image with the matcher."""

    def classify_patch(self, patch):
        raise AssertionError('the matcher was asked to classify a sliding window')


@pytest.fixture
def detector(tmp_path, monkeypatch):
    from warp import userdata
    monkeypatch.setattr(userdata, 'training_data_dir', lambda: tmp_path)
    det = ld.LayoutDetector()
    monkeypatch.setattr(det, '_detect_boffs_via_markers',
                        lambda img: {BOFF_SEAT: [(500, 100, 26, 35)]})
    monkeypatch.setattr(det, '_get_eq_geometry', lambda img: None)
    monkeypatch.setattr(det, '_detect_via_ocr_anchored', lambda *a, **k: {})
    monkeypatch.setattr(det, '_detect_via_learned_layouts', lambda *a, **k: None)
    monkeypatch.setattr(ld._trait_grid, 'detect_traits', lambda *a, **k: {
        TRAITS: [(100 + 40 * i, 400, 33, 45) for i in range(5)]})
    return det


def _screenshot():
    # Textured, so a window scan could not skip it as flat background.
    return np.random.default_rng(0).integers(0, 255, (600, 1000, 3), dtype=np.uint8)


def _detect(det):
    return det.detect(_screenshot(), 'SPACE_MIXED', {},
                      icon_matcher=_Matcher(), app_cache=types.SimpleNamespace())


def test_no_window_of_the_image_is_sent_to_the_matcher(detector):
    _detect(detector)


def test_the_boff_seats_found_by_their_markers_are_kept(detector):
    assert BOFF_SEAT in _detect(detector)


def test_the_trait_grid_is_kept(detector):
    assert len(_detect(detector)[TRAITS]) == 5
