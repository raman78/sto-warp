"""The equipment panel is found, and its rows named, without any label.

A screenshot cropped past the label column used to get no equipment boxes at
all. `eq_stack.detect_eq_stack` finds the grid from the cells: a right-hand
column one cell per row, and the run of one-cell rows (Deflector ... Shield)
that sits under Fore Weapons on every ship. Rows under Aft Weapons are named
by the items in them, only where one arrangement explains them.

Offline: a synthetic panel laid out like image-c8be3f34ec234254.png, and a
stub matcher that reads an item from the colour it was painted in.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip('cv2')

from warp.recognition import eq_geometry as eg
from warp.recognition import eq_stack
from warp.recognition import layout_detector as ld

CELLS = [5, 1, 1, 1, 1, 3, 5, 2, 4, 2, 5]
# Rows under Aft are painted one shade each, so the stub can tell them apart.
FILL = [200, 200, 200, 200, 200, 200, 110, 140, 170, 230, 250]
ITEM_BY_FILL = {110: 'Dev A', 140: 'Uni A', 170: 'Eng A', 230: 'Sci A', 250: 'Tac A'}
EQ_CACHE = {
    'experimental': {'Exp A': {}}, 'devices': {'Dev A': {}}, 'hangars': {'Pet A': {}},
    'uni_consoles': {'Uni A': {}, 'Eng A': {}, 'Sci A': {}, 'Tac A': {}},
    'eng_consoles': {'Uni A': {}, 'Eng A': {}},
    'sci_consoles': {'Uni A': {}, 'Sci A': {}},
    'tac_consoles': {'Uni A': {}, 'Tac A': {}},
}
RIGHT, DX, PITCH, W, H, TOP = 196, 34.5, 48, 33, 42, 60


def _panel(cells=CELLS):
    img = np.zeros((TOP + PITCH * len(cells) + 40, 240, 3), np.uint8)
    for r, n in enumerate(cells):
        f = FILL[r] if r < len(FILL) else 200
        y = TOP + r * PITCH
        img[y - 3:y + H + 3, 17:RIGHT + 4] = 40                     # the row's bar
        for j in range(n):
            x1 = int(round(RIGHT - (j + 1) * DX)) + 1
            img[y:y + H, x1:x1 + W] = f                              # the icon
            img[y:y + H, x1] = img[y:y + H, x1 + W - 1] = 255        # its frame
            img[y, x1:x1 + W] = img[y + H - 1, x1:x1 + W] = 255
            img[y + 8:y + 20, x1 + 8:x1 + 20] = max(0, f - 90)       # some texture
    return img


class _ColourMatcher:
    """Reads the item a cell was painted as, from a pixel off its texture."""

    def match(self, crop, candidate_names=None):
        v = int(crop[crop.shape[0] - 8, crop.shape[1] // 2][0])
        name = min(ITEM_BY_FILL.items(), key=lambda kv: abs(kv[0] - v))[1]
        return name, 0.9, None, False


@pytest.fixture
def detector(tmp_path, monkeypatch):
    from warp import userdata
    monkeypatch.setattr(userdata, 'training_data_dir', lambda: tmp_path)
    return ld.LayoutDetector()


def _named(geom):
    return {ld._STD_IDX_TO_PROD_SLOT[i]: cy for i, cy in geom.eq_label_cys.items()}


def _row_cy(r):
    return TOP + r * PITCH + H // 2


def test_a_panel_without_labels_is_found_from_its_cells(detector):
    geom = eq_stack.detect_eq_stack(_panel(), detector)
    assert geom is not None and geom.mode == 'stack'
    assert len(geom.row_cys) == len(CELLS)
    assert abs(geom.panel_right - RIGHT) <= 1


def test_the_cell_pitch_is_read_to_a_fraction_of_a_pixel(detector):
    geom = eq_stack.detect_eq_stack(_panel(), detector)
    assert abs(geom.final_dx - DX) <= 0.5


def test_its_shape_names_fore_the_one_cell_run_and_aft(detector):
    named = _named(eq_stack.detect_eq_stack(_panel(), detector))
    assert named == {'Fore Weapons': _row_cy(0), 'Deflector': _row_cy(1),
                     'Engines': _row_cy(2), 'Warp Core': _row_cy(3),
                     'Shield': _row_cy(4), 'Aft Weapons': _row_cy(5)}


def test_rows_under_aft_are_named_by_what_is_in_them(detector):
    named = _named(eq_stack.detect_eq_stack(_panel(), detector, _ColourMatcher(), EQ_CACHE))
    assert [s for s, _ in sorted(named.items(), key=lambda kv: kv[1])][6:] == [
        'Devices', 'Universal Consoles', 'Engineering Consoles',
        'Science Consoles', 'Tactical Consoles']


def test_rows_no_arrangement_explains_are_left_unnamed(detector):
    """Every console read as a device: no arrangement fits, so nothing under
    Aft is named rather than something being named wrongly."""
    class _AllDevices:
        def match(self, crop, candidate_names=None):
            return 'Dev A', 0.9, None, False
    named = _named(eq_stack.detect_eq_stack(_panel(), detector, _AllDevices(), EQ_CACHE))
    assert set(named) == {'Fore Weapons', 'Deflector', 'Engines', 'Warp Core',
                          'Shield', 'Aft Weapons'}


def test_a_grid_with_no_one_cell_rows_is_not_an_equipment_panel(detector):
    """A trait grid draws the frame of every slot, so its rows are full."""
    assert eq_stack.detect_eq_stack(_panel([5] * 10), detector) is None


def test_the_detector_falls_back_to_it_when_no_label_is_read(detector, monkeypatch):
    monkeypatch.setattr(ld, 'detect_eq_geometry', lambda img, ocr_tokens=None: None)
    geom = detector._get_eq_geometry(_panel())
    assert geom is not None and geom.mode == 'stack'


def test_labels_still_come_first(detector, monkeypatch):
    from_labels = eg.EQGeometry(0, 196, 34.5, 48, [81], 'v8', {})
    monkeypatch.setattr(ld, 'detect_eq_geometry', lambda img, ocr_tokens=None: from_labels)
    assert detector._get_eq_geometry(_panel()) is from_labels


FEW = {'Fore Weapons': [(163, 60, 33, 42)]}                           # 1 of 11 slots


def _space_eq_detect(detector, monkeypatch, mode):
    geom = eg.EQGeometry(0, 196, 34.5, 48, [81], mode, {})
    few = FEW
    monkeypatch.setattr(detector, '_get_eq_geometry', lambda img: geom)
    monkeypatch.setattr(detector, '_detect_via_pixel_analysis', lambda *a: dict(few))
    monkeypatch.setattr(detector, '_detect_via_learned_layouts', lambda *a: None)
    return detector._detect_raw(_panel(), 'SPACE', {})


def test_a_label_less_grid_is_kept_though_it_names_few_rows(detector, monkeypatch):
    assert _space_eq_detect(detector, monkeypatch, 'stack') == FEW


def test_a_labelled_grid_naming_few_rows_still_falls_through(detector, monkeypatch):
    """The 70% bar stays for label geometry — only the stack is exempt."""
    assert _space_eq_detect(detector, monkeypatch, 'v8') != FEW


def test_a_grid_found_without_a_matcher_is_not_reused_once_one_is_there(detector, monkeypatch):
    """The geometry is cached per screenshot. Without a matcher the rows under
    Aft stay unnamed; a later call with one must name them, not be handed the
    poorer grid from the cache."""
    monkeypatch.setattr(ld, 'detect_eq_geometry', lambda img, ocr_tokens=None: None)
    img = _panel()
    detector._stack_matcher, detector._stack_eq_cache = None, None
    assert len(detector._get_eq_geometry(img).eq_label_cys) == 6
    detector._stack_matcher, detector._stack_eq_cache = _ColourMatcher(), EQ_CACHE
    assert len(detector._get_eq_geometry(img).eq_label_cys) == 11


def test_rows_left_unnamed_that_hold_items_are_recorded_for_the_user(detector, monkeypatch):
    """Without a matcher, and with a profile that expects four rows under Aft
    where five are drawn, those rows stay unnamed and get no boxes. The ones
    known to be panel rows — the four every ship draws under Aft — are
    recorded so the importer can ask for them; the fifth may lie past the
    panel's bottom and is not."""
    monkeypatch.setattr(ld, 'detect_eq_geometry', lambda img, ocr_tokens=None: None)
    detector._stack_matcher, detector._stack_eq_cache = None, None
    detector._detect_via_pixel_analysis(
        _panel(), ld.SPACE_SLOT_ORDER_STANDARD, {'Universal Consoles': 0})
    assert [r['row'] for r in detector.last_unnamed_rows] == [7, 8, 9, 10]
