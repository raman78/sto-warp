"""The trait grid may improve a section's boxes; it may not delete them.

`trait_grid` is the structure-driven detector, measured at 91.5% slot IoU
against the OCR-header baseline, so where it finds a whole section its
positions win. It replaced the profile-sized rows unconditionally, though, and
the counter tracking the change could come out negative.

Measured on `image-4391ccd9d2683d4e.png`: the grid split one Starship Traits
section across two row groups, dropped the second as a duplicate section — a
section really does appear once per screen — and the merge then took the
survivor. `Starship Traits` went from the 7 the ship's profile says it has down
to 2, `Space Reputation` 5 to 3, `Personal Space Traits` 11 to 10. Eight boxes
vanished and the slots behind them were never drawn, so nothing marked them for
review and the user had to spot the gap and draw them by hand.

Offline: `merge_trait_boxes` is called directly, no image and no OCR.
"""
from __future__ import annotations

import pytest

pytest.importorskip("cv2")
np = pytest.importorskip("numpy")



def _merge(result, grid, boff_panel=None):
    """Call the shipped merge directly — it is a module-level function."""
    logged: list[str] = []

    class _Log:
        def info(self, m): logged.append(m)
        def warning(self, m): logged.append(m)

    import warp.recognition.layout_detector as ld
    real, ld._slog = ld._slog, _Log()
    try:
        out = ld.merge_trait_boxes(result, grid,
                                   boff_panel or (lambda b: False))
    finally:
        ld._slog = real
    return out, logged


# ── The production case ───────────────────────────────────────────────────

PROFILE_ROWS = {
    'Personal Space Traits': [(0, 0, 9, 9)] * 11,
    'Starship Traits':       [(0, 0, 9, 9)] * 7,
    'Space Reputation':      [(0, 0, 9, 9)] * 5,
}
GRID_FOUND = {
    'Personal Space Traits': [(1, 1, 9, 9)] * 10,
    'Starship Traits':       [(1, 1, 9, 9)] * 2,
    'Space Reputation':      [(1, 1, 9, 9)] * 3,
}


def test_a_shorter_section_does_not_replace_the_rows():
    out, _ = _merge(dict(PROFILE_ROWS), GRID_FOUND)
    assert len(out['Starship Traits']) == 7
    assert len(out['Space Reputation']) == 5
    assert len(out['Personal Space Traits']) == 11


def test_the_kept_rows_are_the_original_boxes():
    out, _ = _merge(dict(PROFILE_ROWS), GRID_FOUND)
    assert out['Starship Traits'] == PROFILE_ROWS['Starship Traits']


def test_the_refusal_is_reported():
    """Rule: nothing is set aside quietly. A count that keeps appearing is how
    anyone learns the grid is splitting a section."""
    _out, logged = _merge(dict(PROFILE_ROWS), GRID_FOUND)
    text = ' '.join(logged)
    assert 'Starship Traits 2<7' in text
    assert 'Space Reputation 3<5' in text


# ── What must still happen ────────────────────────────────────────────────

def test_a_longer_section_still_wins():
    """Its positions are the better ones, so more boxes replace fewer."""
    out, _ = _merge({'Starship Traits': [(0, 0, 9, 9)] * 2},
                    {'Starship Traits': [(1, 1, 9, 9)] * 7})
    assert len(out['Starship Traits']) == 7
    assert out['Starship Traits'][0] == (1, 1, 9, 9)


def test_an_equal_count_takes_the_grids_positions():
    out, _ = _merge({'Starship Traits': [(0, 0, 9, 9)] * 5},
                    {'Starship Traits': [(1, 1, 9, 9)] * 5})
    assert out['Starship Traits'][0] == (1, 1, 9, 9)


def test_a_section_the_rows_do_not_have_is_added():
    out, _ = _merge({}, {'Starship Traits': [(1, 1, 9, 9)] * 3})
    assert len(out['Starship Traits']) == 3


def test_boxes_inside_the_boff_panel_are_still_dropped():
    """Unchanged behaviour — a trait box over the BOFF markers is not a trait."""
    out, logged = _merge(
        {'Starship Traits': []},
        {'Starship Traits': [(1, 1, 9, 9), (2, 2, 9, 9)]},
        boff_panel=lambda b: b[0] == 2)
    assert len(out['Starship Traits']) == 1
    assert 'overlapping BOFF marker panel' in ' '.join(logged)


def test_no_grid_result_leaves_everything_alone():
    out, _ = _merge(dict(PROFILE_ROWS), {})
    assert out == PROFILE_ROWS
