"""Rows nothing else named are named from what is in them — or guessed, or
left Unknown, and said to be.

`eq_row_naming.name_run` takes a run of unnamed equipment rows between two
named ones, the items the matcher read in each, and the cargo groups that say
which slot an item can sit in. It names a row when one arrangement of the
game's rows explains the run; guesses when an arrangement fits but not
certainly; answers Unknown when nothing fits; and marks trailing rows past
the panel's bottom 'outside'.

Offline: item names and groups are made up for the test.
"""
from __future__ import annotations

from warp.recognition.eq_row_naming import arrangements, name_run

ALLOWED = {
    'Aft Weapons': {'Turret'}, 'Experimental': {'Exp'}, 'Devices': {'Dev'},
    'Universal Consoles': {'Uni', 'Eng', 'Sci', 'Tac'},
    'Engineering Consoles': {'Uni', 'Eng'}, 'Science Consoles': {'Uni', 'Sci'},
    'Tactical Consoles': {'Uni', 'Tac'}, 'Hangars': {'Pet'},
    'Engines': {'Imp'}, 'Sec-Def': {'SecDef'},
}
UPPER = {'Fore Weapons', 'Deflector', 'Engines', 'Warp Core', 'Shield', 'Aft Weapons'}


def test_a_mandatory_row_between_two_named_ones_is_forced():
    """Between Deflector and Warp Core one row: Engines must be drawn, so it
    is Engines whatever is read — Sec-Def would leave Engines out."""
    assert arrangements('Deflector', 'Warp Core', 1, {'Deflector', 'Warp Core'}) == [['Engines']]
    assert name_run([[]], 'Deflector', 'Warp Core', {'Deflector', 'Warp Core'}, ALLOWED) == [
        ('name', 'Engines')]


def test_the_items_decide_which_optional_rows_are_drawn():
    items = [['Dev'] * 5, ['Uni', 'Uni'], ['Eng'] * 4, ['Sci'] * 2, ['Tac'] * 5]
    out = name_run(items, 'Aft Weapons', None, UPPER, ALLOWED)
    assert out == [('name', 'Devices'), ('name', 'Universal Consoles'),
                   ('name', 'Engineering Consoles'), ('name', 'Science Consoles'),
                   ('name', 'Tactical Consoles')]


def test_rows_the_items_cannot_tell_apart_are_guessed_not_named():
    """Five rows under Aft, the top two with nothing read, the rest holding
    Universal consoles (which fit any console row). 'Experimental, Devices,
    Eng, Sci, Tac' and 'Devices, Universal, Eng, Sci, Tac' explain it equally.
    The bottom three are the same in both and are named; the top two are
    only guessed."""
    items = [[], [], ['Uni'], ['Uni'], ['Uni']]
    out = name_run(items, 'Aft Weapons', None, UPPER, ALLOWED)
    assert [o for o, _ in out] == ['guess', 'guess', 'name', 'name', 'name']
    assert [s for _, s in out][2:] == ['Engineering Consoles', 'Science Consoles',
                                      'Tactical Consoles']


def test_a_row_whose_items_fit_no_free_slot_is_unknown():
    """Hangar pets two rows under Aft: no arrangement puts Hangars there, and
    the row is not guessed into a slot its items cannot sit in."""
    items = [['Dev'], ['Pet', 'Pet'], ['Eng'], ['Sci'], ['Tac']]
    out = name_run(items, 'Aft Weapons', 'Hangars', UPPER, ALLOWED)
    assert out[1] == ('unknown', None)


def test_no_arrangement_at_all_leaves_every_row_unknown():
    assert name_run([['Dev'], ['Dev']], 'Deflector', 'Engines', {'Deflector', 'Engines'},
                    ALLOWED) == [('unknown', None), ('unknown', None)]


def test_empty_rows_past_the_last_named_one_lie_outside_the_panel():
    """Under Tactical only Hangars can follow; two rows with nothing read
    there are taken to be past the panel's bottom, not asked for."""
    used = UPPER | {'Devices', 'Engineering Consoles', 'Science Consoles', 'Tactical Consoles'}
    out = name_run([[], []], 'Tactical Consoles', None, used, ALLOWED, open_end=True)
    assert out == [('outside', None), ('outside', None)]


def test_a_row_with_equipment_is_never_dropped_as_outside():
    used = UPPER | {'Devices', 'Engineering Consoles', 'Science Consoles', 'Tactical Consoles'}
    out = name_run([['Pet', 'Pet']], 'Tactical Consoles', None, used, ALLOWED, open_end=True)
    assert out == [('name', 'Hangars')]


def test_a_row_of_another_panels_icons_under_the_panel_is_left_out():
    """Measured on Carrier Tank Hybrid.png with OCR off: the stack ran on
    into the specialisation icons under the panel, the matcher read a few
    stray hits there, and the only arrangement for seven rows shifted every
    row under Aft by one. Leaving a row out costs its items, so the stray
    row is dropped and the rest are named where they are."""
    items = [['Dev'] * 6, ['Uni'] * 2, ['Eng'] * 5, ['Sci'] * 3, ['Tac'] * 3,
             ['Pet'] * 2, ['Dev', 'Exp', 'Pet']]
    out = name_run(items, 'Aft Weapons', None, UPPER, ALLOWED, open_end=True)
    assert out == [('name', 'Devices'), ('name', 'Universal Consoles'),
                   ('name', 'Engineering Consoles'), ('name', 'Science Consoles'),
                   ('name', 'Tactical Consoles'), ('name', 'Hangars'), ('outside', None)]


def test_rows_cut_off_above_an_unnamed_run_are_not_required():
    """Measured on Hirogen Predator.png with OCR off: a tooltip over the
    Deflector row hid Fore Weapons above the visible stack, and nothing above
    the run was named. Requiring Fore at the top shifted every row by one and
    turned Engines, Warp Core and Shield — read at 97-100% — into Unknown."""
    items = [['Imp'], ['Core'], ['Shield'], ['Turret', 'Turret'], ['Dev'] * 4]
    allowed = dict(ALLOWED, **{'Warp Core': {'Core'}, 'Shield': {'Shield'}})
    out = name_run(items, None, 'Universal Consoles', set(), allowed)
    assert out == [('name', 'Engines'), ('name', 'Warp Core'), ('name', 'Shield'),
                   ('name', 'Aft Weapons'), ('name', 'Devices')]
