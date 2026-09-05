"""A row whose label OCR could not read takes its name from the rows it could.

The equipment panel's rows are one ladder: OCR labels pin the ones it reads and
the gaps between them are interpolated, so the row *positions* were never the
problem. The row *names* were: an unanchored row used to take
`extended_order[i]`, a flat list indexed by row number, and that list did not
match the panel whenever it contained `Hangars` — which was inserted right
after `Aft Weapons` while the game draws it last.

Measured on `image-4e7c6849dd28da67.png`, where the `Devices` and
`Universal Consoles` labels are covered by a tooltip:

    row 6  guess 'Hangars'  -> already anchored at row 11 -> left empty
    row 7  guess 'Devices'  -> accepted, one row too low
           'Universal Consoles' never placed

giving 11 slot groups and 28 boxes, a hole under Aft Weapons, and Devices drawn
over the Universal Consoles row. After: 12 groups, 29 boxes, and the Devices
row's own pixel count agrees with the profile (4 = 4) where it read 1 before.

Offline: pure sequence logic, no image and no OCR.
"""
from __future__ import annotations

import pytest

pytest.importorskip("cv2")

from warp.recognition.layout_detector import fill_unanchored_rows


ROWS = [94, 158, 224, 286, 350, 412, 476, 540, 604, 668, 731, 797]
EXPECTED = ['Fore Weapons', 'Deflector', 'Engines', 'Warp Core', 'Shield',
            'Aft Weapons', 'Devices', 'Universal Consoles',
            'Engineering Consoles', 'Science Consoles', 'Tactical Consoles',
            'Hangars']
ANCHORED = {94: 'Fore Weapons', 158: 'Deflector', 224: 'Engines',
            286: 'Warp Core', 350: 'Shield', 412: 'Aft Weapons',
            604: 'Engineering Consoles', 668: 'Science Consoles',
            731: 'Tactical Consoles', 797: 'Hangars'}


# ── The production case ───────────────────────────────────────────────────

@pytest.fixture
def filled():
    return fill_unanchored_rows(ROWS, ANCHORED, EXPECTED)


def test_the_two_covered_labels_are_recovered(filled):
    assert filled[476] == 'Devices'
    assert filled[540] == 'Universal Consoles'


def test_every_row_gets_a_name(filled):
    assert [filled.get(cy) for cy in ROWS] == EXPECTED


def test_anchored_rows_are_never_overwritten(filled):
    for cy, slot in ANCHORED.items():
        assert filled[cy] == slot


def test_hangars_is_not_pulled_up_under_aft_weapons(filled):
    """The old failure: `Hangars` sat at position 6 of the flat list."""
    assert filled[476] != 'Hangars'
    assert filled[797] == 'Hangars'


# ── Only fill when the count is unambiguous ───────────────────────────────

def test_a_gap_larger_than_the_slots_that_fit_is_left_unnamed():
    """Three empty rows but only two slots between the anchors — naming any of
    them would be a guess, and a wrong name writes an item into the wrong
    slot."""
    rows = [10, 20, 30, 40, 50]
    anchored = {10: 'Aft Weapons', 50: 'Engineering Consoles'}
    out = fill_unanchored_rows(rows, anchored, EXPECTED)
    assert 20 not in out and 30 not in out and 40 not in out


def test_a_gap_smaller_than_the_slots_that_fit_is_left_unnamed():
    rows = [10, 20, 30]
    anchored = {10: 'Fore Weapons', 30: 'Aft Weapons'}
    out = fill_unanchored_rows(rows, anchored, EXPECTED)
    assert 20 not in out


def test_anchors_out_of_sequence_do_not_fill():
    """If the anchors contradict the expected order, nothing between them can
    be trusted."""
    rows = [10, 20, 30]
    anchored = {10: 'Tactical Consoles', 30: 'Deflector'}
    out = fill_unanchored_rows(rows, anchored, EXPECTED)
    assert 20 not in out


# ── The ends of the panel ─────────────────────────────────────────────────

def test_rows_above_the_first_anchor_are_filled():
    rows = [10, 20, 30]
    anchored = {30: 'Engines'}
    out = fill_unanchored_rows(rows, anchored, EXPECTED)
    assert out[10] == 'Fore Weapons'
    assert out[20] == 'Deflector'


def test_rows_below_the_last_anchor_are_filled():
    rows = [10, 20, 30]
    anchored = {10: 'Science Consoles'}
    out = fill_unanchored_rows(rows, anchored, EXPECTED)
    assert out[20] == 'Tactical Consoles'
    assert out[30] == 'Hangars'


def test_a_trailing_run_longer_than_what_is_left_is_not_filled():
    rows = [10, 20, 30, 40]
    anchored = {10: 'Tactical Consoles'}       # only 'Hangars' remains
    out = fill_unanchored_rows(rows, anchored, EXPECTED)
    assert 20 not in out and 30 not in out and 40 not in out


# ── Degenerate inputs ─────────────────────────────────────────────────────

def test_no_anchors_at_all_fills_nothing():
    """With nothing read, there is nothing to align against — the panel could
    start anywhere."""
    assert fill_unanchored_rows(ROWS, {}, EXPECTED) == {}


def test_an_empty_expected_sequence_changes_nothing():
    assert fill_unanchored_rows(ROWS, ANCHORED, []) == ANCHORED


def test_no_rows_changes_nothing():
    assert fill_unanchored_rows([], ANCHORED, EXPECTED) == ANCHORED


def test_an_anchor_outside_the_expected_sequence_is_ignored_as_a_reference():
    """A ship whose profile says 0 of a slot, but whose label OCR read anyway:
    it cannot position anything, and it must not crash."""
    rows = [10, 20, 30]
    anchored = {10: 'Fore Weapons', 20: 'Sec-Def', 30: 'Engines'}
    out = fill_unanchored_rows(rows, anchored, EXPECTED)
    assert out[20] == 'Sec-Def'          # kept, since anchors always win


def test_the_input_mapping_is_not_modified():
    before = dict(ANCHORED)
    fill_unanchored_rows(ROWS, ANCHORED, EXPECTED)
    assert ANCHORED == before
