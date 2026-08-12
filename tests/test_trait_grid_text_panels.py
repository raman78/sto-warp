"""Text rows must never lock as a trait panel (warp.recognition.trait_grid).

Evenly-spaced capital letters — the ship-name divider between the two
Starship-Traits rows — pass every structural test a trait row does: icon-like
aspect ratio, one baseline, regular pitch. Before the scale guard they locked
as a full 5-column panel of 9x11 "icons" and emitted five junk bboxes.

Synthetic: bright rectangles on black, so only cv2/numpy are needed.
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


def _blank(h=435, w=240):
    return np.zeros((h, w, 3), dtype=np.uint8)


def _row(img, y, n=5, x0=17, dx=42, w=36, h=48):
    """Paint one row of n bright icon-sized rectangles."""
    for i in range(n):
        x = x0 + i * dx
        img[y:y + h, x:x + w] = 255


def _panels(img):
    ccs = tg._detect_icon_ccs(img)
    rows = tg._find_trait_rows(ccs)
    return tg._lock_grids_multi(tg._cluster_row_groups(rows))


def test_icon_rows_lock_a_single_panel():
    img = _blank()
    for y in (34, 88, 184, 376):
        _row(img, y)
    panels = _panels(img)
    assert len(panels) == 1
    assert panels[0]['icon_h'] == 48


def test_letter_sized_row_does_not_lock_a_second_panel():
    img = _blank()
    for y in (34, 88, 184, 376):
        _row(img, y)
    # Ship-name divider text: 9x11 glyphs at a regular 10.5 px pitch,
    # measured off a real Traits screenshot.
    _row(img, 256, n=8, x0=21, dx=10, w=9, h=11)

    panels = _panels(img)
    assert len(panels) == 1, [(p['icon_w'], p['icon_h']) for p in panels]
    assert panels[0]['icon_h'] == 48


def test_panels_at_similar_scale_are_both_kept():
    """Composed screenshots legitimately mix panel scales — the guard must
    only cut text, not a smaller-but-real panel (corpus low is 0.63x)."""
    img = _blank(h=600, w=600)
    for y in (34, 88):
        _row(img, y)
    for y in (300, 360):  # second panel, 0.65x scale, own x-origin
        _row(img, y, x0=300, dx=28, w=24, h=31)

    panels = _panels(img)
    assert len(panels) == 2
    assert sorted(round(p['icon_h']) for p in panels) == [31, 48]


def test_drop_text_scale_panels_keeps_a_lone_panel():
    """A single panel has no reference to compare against — never drop it."""
    only = [{'icon_w': 9.0, 'icon_h': 11.0, 'cols': [], 'col_dx': 10.0,
             'y_top': 256, 'y_bot': 267}]
    assert tg._drop_text_scale_panels(only) == only
