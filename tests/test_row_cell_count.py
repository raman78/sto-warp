"""A row is counted only where the screenshot shows it.

`_count_icons_in_row` walks a row right to left, one cell per step, up to the
6 cells a space equipment row can hold. On image-c8be3f34ec234254.png the
screenshot ends at the panel's left edge, so the window for a sixth position
starts 2 px left of the image. The guard meant to stop there came after a
clamp to 0 and could never fire; the walk read the leftover sliver — the
panel's edge and the fifth icon's border — as a cell, and a five-cell Fore
Weapons row counted six.

Offline: a synthetic row, no OCR, no model.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip('cv2')

from warp.recognition.layout_detector import LayoutDetector

RIGHT, CELL_W, TOP, BOTTOM = 196, 34, 60, 104


@pytest.fixture
def detector(tmp_path, monkeypatch):
    from warp import userdata
    monkeypatch.setattr(userdata, 'training_data_dir', lambda: tmp_path)
    return LayoutDetector()


def _row(first_cell_x=25):
    """Five framed cells right-justified at RIGHT, the panel's bar from x=17
    and black image border before it — the layout of the measured screenshot."""
    img = np.zeros((140, 215, 3), np.uint8)
    img[TOP - 4:BOTTOM + 4, 17:] = 40                      # the row's bar
    for j in range(5):
        x2 = RIGHT - j * CELL_W
        x1 = first_cell_x if j == 4 else x2 - CELL_W + 1
        img[TOP:BOTTOM, x1:x2] = 90                          # the icon
        img[TOP:BOTTOM, x1] = img[TOP:BOTTOM, x2 - 1] = 220  # its frame
        img[TOP, x1:x2] = img[BOTTOM - 1, x1:x2] = 220
    return img


def test_a_position_the_screenshot_cuts_off_is_not_counted(detector):
    _, states = detector._count_icons_in_row(
        _row(), TOP, BOTTOM, RIGHT, CELL_W, 'Fore Weapons', panel_x_start=RIGHT - 6 * CELL_W)
    assert len(states) == 5
