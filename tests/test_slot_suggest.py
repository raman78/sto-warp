"""The slot WARP CORE suggests for a box drawn by hand.

`WarpCoreWindow._suggest_slot_from_position` names a new box from the slot
of the nearest item above it and the order the rows come in. On
image-c8be3f34ec234254.png it suggested 'Sec-Def' for the Engines row of a
ship with no secondary deflector, and 'Boff Science' for the Shield row,
because a BOFF seat in the next panel sat just above it. The order now
skips rows the identified ship lacks, and a box in the equipment column
takes only an equipment row as the one above it.

Offline: the method is called on a stand-in holding only the state it reads.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip('PySide6')

from warp.trainer.trainer_window import WarpCoreWindow

SHOT = 'image-c8be3f34ec234254.png'
# The boxes on that screenshot at the moment each suggestion was asked for.
FORE = [{'slot': 'Fore Weapons', 'bbox': (x, 60, 34, 42)} for x in (24, 59, 95, 128, 164)]
DEFLECTOR = {'slot': 'Deflector', 'bbox': (161, 105, 36, 47)}
ENGINES = {'slot': 'Engines', 'bbox': (162, 155, 33, 43)}
WARP_CORE = {'slot': 'Warp Core', 'bbox': (162, 201, 33, 43)}
BOFF = {'slot': 'Boff Engineering', 'bbox': (468, 213, 26, 35)}   # seat in the next panel
NO_SEC_DEF = {'Sec-Def': 0, 'Experimental': 0, 'Hangars': 0, 'Fore Weapons': 5}


def _suggest(items, bbox, profile=None):
    win = SimpleNamespace(
        _current_idx=0, _screenshots=[SimpleNamespace(name=SHOT)],
        _screen_types={SHOT: 'SPACE_MIXED'}, _recognition_items=items,
        _ship_profiles={SHOT: profile} if profile is not None else {},
        _STYPE_TO_BUILD={})
    return WarpCoreWindow._suggest_slot_from_position(win, bbox)


def test_a_row_the_identified_ship_lacks_is_not_suggested():
    assert _suggest(FORE + [DEFLECTOR], (162, 155, 33, 43), NO_SEC_DEF) == 'Engines'


def test_without_an_identified_ship_every_row_stays_in_the_order():
    """No profile says the ship lacks Sec-Def, so it is still the row after
    Deflector — the suggestion is only narrowed on evidence."""
    assert _suggest(FORE + [DEFLECTOR], (162, 155, 33, 43)) == 'Sec-Def'


def test_a_seat_in_the_next_panel_is_not_the_row_above_an_equipment_box():
    items = FORE + [DEFLECTOR, ENGINES, WARP_CORE, BOFF]
    assert _suggest(items, (162, 248, 34, 45), NO_SEC_DEF) == 'Shield'
