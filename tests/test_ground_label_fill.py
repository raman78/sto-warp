"""A ground label OCR missed is placed from the ones it read.

Ground rows are not evenly spaced — `Weapons` holds two stacked cells, so
`Ground Devices` follows it about one and a half rows down, not one. But the
spacing is a fixed proportion of the panel's scale, and a very stable one.
Measured over the 17 ground screenshots with confirmed boxes, whose row pitch
runs from 58 to 106 px, each row's offset from `Kit Modules` came out as

    Kit Modules 0.000  Kit 1.010  Body/EV 2.013
    Personal Shield 3.004  Weapons 4.021  Ground Devices 5.721

with a standard deviation of 0.013–0.024 of a pitch — about 1.2 px on a typical
panel. So one label positions all the others.

Before this, only `Kit` was ever recovered, and only from `Body Armor`. Every
other missing label lost its whole row of slots — which is what one mistyped
character in `Kit Modules` cost until the matcher learned to fuzz.

Offline: pure arithmetic on label positions, no image and no OCR.
"""
from __future__ import annotations

import pytest

pytest.importorskip("cv2")

from warp.recognition.ground_eq_geometry import ROW_RATIO, _fill_missing_labels


# Real label positions from ScreenShot-2025-08-18-at-10-02-52-PM.png,
# row pitch 73.
PITCH = 73
TRUTH = {'Kit Modules': 13, 'Kit': 87, 'Body Armor': 159, 'EV Suit': 159,
         'Personal Shield': 233, 'Weapons': 307, 'Ground Devices': 432}


def _err(got: dict, slot: str) -> int:
    return abs(got[slot] - TRUTH[slot])


# ── Reconstruction from what survived ─────────────────────────────────────

def test_two_labels_reconstruct_the_whole_panel():
    """Kit and Weapons only — everything else has to be placed."""
    out = _fill_missing_labels({'Kit': 87, 'Weapons': 307}, PITCH)
    assert set(out) == set(ROW_RATIO)
    for slot in TRUTH:
        assert _err(out, slot) <= 2, f'{slot}: {out[slot]} vs {TRUTH[slot]}'


def test_one_label_is_enough():
    """A single anchor still places every row — less accurately, but the
    alternative is no panel at all."""
    out = _fill_missing_labels({'Weapons': 307}, PITCH)
    assert set(out) == set(ROW_RATIO)
    for slot in TRUTH:
        assert _err(out, slot) <= 3, f'{slot}: {out[slot]} vs {TRUTH[slot]}'


def test_the_stacked_weapons_row_is_accounted_for():
    """Devices is 1.7 rows below Weapons, not 1. Placing it one pitch down
    would put it 20 px high on this panel."""
    out = _fill_missing_labels({'Weapons': 307}, PITCH)
    assert out['Ground Devices'] > 307 + PITCH + 10


def test_kit_modules_is_recovered_too():
    """The case that cost six cells: its label misread, so it was absent."""
    partial = {k: v for k, v in TRUTH.items() if k != 'Kit Modules'}
    out = _fill_missing_labels(partial, PITCH)
    assert _err(out, 'Kit Modules') <= 2


@pytest.mark.parametrize('dropped', sorted(TRUTH))
def test_any_single_missing_label_is_recovered(dropped):
    partial = {k: v for k, v in TRUTH.items() if k != dropped}
    out = _fill_missing_labels(partial, PITCH)
    assert _err(out, dropped) <= 2, f'{dropped}: {out[dropped]} vs {TRUTH[dropped]}'


# ── What must not change ──────────────────────────────────────────────────

def test_labels_that_were_read_are_never_moved():
    """A real reading beats a projection, always."""
    out = _fill_missing_labels({'Kit': 87, 'Weapons': 307}, PITCH)
    assert out['Kit'] == 87
    assert out['Weapons'] == 307


def test_a_complete_panel_is_returned_unchanged():
    assert _fill_missing_labels(dict(TRUTH), PITCH) == TRUTH


def test_one_low_label_cannot_drag_the_panel():
    """The median of the candidates is taken, not the first, so a single
    mis-placed anchor is outvoted by the others."""
    skewed = dict(TRUTH)
    del skewed['Ground Devices']
    skewed['Kit'] = 87 + 30                    # one anchor 30 px too low
    out = _fill_missing_labels(skewed, PITCH)
    assert _err(out, 'Ground Devices') <= 3


# ── Degenerate inputs ─────────────────────────────────────────────────────

def test_nothing_read_places_nothing():
    assert _fill_missing_labels({}, PITCH) == {}


def test_an_unusable_pitch_changes_nothing():
    got = {'Kit': 87}
    assert _fill_missing_labels(got, 0) == got


def test_an_unknown_slot_is_not_used_as_an_anchor_and_is_kept():
    out = _fill_missing_labels({'Kit': 87, 'Nonsense': 500}, PITCH)
    assert out['Nonsense'] == 500
    assert _err(out, 'Weapons') <= 3


def test_a_label_projected_above_the_image_is_dropped():
    """A negative cy is not a row — better absent than at a bogus position."""
    out = _fill_missing_labels({'Ground Devices': 20}, PITCH)
    assert 'Kit Modules' not in out
