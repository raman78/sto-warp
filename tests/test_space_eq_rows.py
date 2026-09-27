"""The space equipment rows are one table, and every module reads it.

The order used to be written out five times and the copies drifted: one
followed the grouping of docs/sto_slots_rules.md rather than the screen,
another put Hangars after Aft Weapons. These tests pin the table, the
sequence built from it for a ship, and that every other view is derived
from it rather than typed out again.

Offline: no image, no model.
"""
from __future__ import annotations

from warp.recognition import space_eq_rows as R

SCREEN_ORDER = [
    'Fore Weapons', 'Deflector', 'Sec-Def', 'Engines', 'Warp Core', 'Shield',
    'Aft Weapons', 'Experimental', 'Devices', 'Universal Consoles',
    'Engineering Consoles', 'Science Consoles', 'Tactical Consoles', 'Hangars',
]


def test_rows_are_in_the_order_the_game_draws_them():
    assert list(R.SLOTS) == SCREEN_ORDER


def test_the_optional_rows_are_the_ones_a_ship_may_lack():
    assert R.OPTIONAL == {'Sec-Def', 'Aft Weapons', 'Experimental',
                          'Universal Consoles', 'Hangars'}


def test_only_the_shield_row_prints_a_label_other_than_its_slot_name():
    assert {r.slot: r.label for r in R.ROWS if r.slot != r.label} == {'Shield': 'Shields'}


def test_a_ship_with_no_optional_rows_draws_the_base_order():
    assert R.row_sequence(R.BASE_ORDER, {'Universal Consoles': 0}) == [
        s for s in R.BASE_ORDER if s != 'Universal Consoles']


def test_a_secondary_deflector_goes_under_the_deflector():
    seq = R.row_sequence(R.BASE_ORDER, {'Sec-Def': 1})
    assert seq[seq.index('Deflector') + 1] == 'Sec-Def'


def test_an_experimental_weapon_goes_under_the_aft_weapons():
    seq = R.row_sequence(R.BASE_ORDER, {'Experimental': 1})
    assert seq[seq.index('Aft Weapons') + 1] == 'Experimental'


def test_hangars_are_drawn_last():
    assert R.row_sequence(R.BASE_ORDER, {'Hangars': 2})[-1] == 'Hangars'
    assert R.row_sequence(R.CARRIER_ORDER, {'Hangars': 2})[-1] == 'Hangars'


def test_a_shuttle_without_aft_weapons_draws_no_aft_row():
    assert 'Aft Weapons' not in R.row_sequence(R.BASE_ORDER, {'Aft Weapons': 0})


def test_a_row_the_profile_does_not_mention_is_kept():
    assert R.row_sequence(R.BASE_ORDER, {}) == list(R.BASE_ORDER)


def test_a_ground_order_passes_through_with_its_zero_rows_dropped():
    ground = ['Kit Modules', 'Kit', 'Body Armor', 'EV Suit']
    assert R.row_sequence(ground, {'EV Suit': 0}) == ['Kit Modules', 'Kit', 'Body Armor']


def test_every_module_reads_the_table():
    from warp.recognition import eq_geometry as eg
    from warp.recognition import layout_detector as ld
    import warp.warp_importer as wi
    assert sorted(eg.STD_ORDER, key=eg.STD_ORDER.get) == [r.label for r in R.ROWS]
    assert set(eg.OPTIONAL_ROWS) == {r.label for r in R.ROWS if r.optional}
    assert ld._STD_IDX_TO_PROD_SLOT == R.SLOT_BY_INDEX
    assert ld.SPACE_SLOT_ORDER_STANDARD == list(R.BASE_ORDER)
    assert ld.SPACE_SLOT_ORDER_CARRIER == list(R.CARRIER_ORDER)
    assert [d['name'] for d in wi.SPACE_SLOT_ORDER] == list(R.SLOTS)
    assert [not d['mandatory'] for d in wi.SPACE_SLOT_ORDER] == [r.optional for r in R.ROWS]
