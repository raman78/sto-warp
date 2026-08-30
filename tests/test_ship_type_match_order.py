"""Ship-type matching must prefer a ship name over a generic class.

`ShipDB._by_type` is keyed on the generic `type` field, so many ships share
a key and the load loop keeps only the last one written. A fuzzy hit there
identifies a *class* and then hands back an arbitrary member of it.

Before the stage reorder, that loose match (cutoff 0.68) ran ahead of the
fuzzy match against real ship names (cutoff 0.85) and short-circuited it.
A single dropped letter in the OCR text was enough to trigger it: the
l1.png screenshot read 'Terran exington Dreadnought Cruiser' and was
reported as 'Universe Temporal Heavy Dreadnought Cruiser' — a different
ship, with a different slot and BOFF profile.

Run standalone:
    python -m pytest tests/test_ship_type_match_order.py -v
"""
from __future__ import annotations

import json

import pytest


def _ship(name: str, stype: str, fore: str = '4') -> dict:
    return {
        'Page': name,
        'name': name,
        'type': stype,
        'boffs': 'Commander Tactical,Lieutenant Universal,Ensign Science',
        'tier': '6',
        'fore': fore,
        'hangars': '',
    }


# Both ships share the generic type, so `_by_type['Heavy Dreadnought Cruiser']`
# keeps whichever is written last — the decoy.
_TARGET = _ship('Terran Lexington Dreadnought Cruiser', 'Heavy Dreadnought Cruiser')
_DECOY = _ship('Universe Temporal Heavy Dreadnought Cruiser', 'Heavy Dreadnought Cruiser')

# A two-word ship name, the shape the old ≥3-word guard discarded outright.
_SHORT = _ship('Advanced Escort', 'Escort')

# The collision the class-read guard exists for, taken from the real roster:
# 'warbird battlecruiser' is a generic class *and* sits 0.9048 from the ship
# name 'arbiter battlecruiser'. Reading the class must not name the ship.
_COLLIDER = _ship('Arbiter Battlecruiser', 'Battlecruiser')
_COLLIDER_CLASS = _ship('Vastam Tactical Command Warbird', 'Warbird Battlecruiser')

_ROSTER = [_TARGET, _DECOY, _SHORT, _COLLIDER, _COLLIDER_CLASS]


@pytest.fixture
def shipdb(tmp_path):
    from warp.warp_importer import ShipDB

    (tmp_path / 'ship_list.json').write_text(
        json.dumps(_ROSTER), encoding='utf-8')
    return ShipDB(tmp_path)


def test_decoy_owns_the_generic_type_key(shipdb):
    """Guards the premise: the wrong ship really is what the class key holds."""
    entry = shipdb._by_type['heavy dreadnought cruiser']
    assert entry['name'] == 'Universe Temporal Heavy Dreadnought Cruiser'


def test_clean_ocr_resolves_to_the_right_ship(shipdb):
    shipdb.get_profile('', 'Terran Lexington Dreadnought Cruiser', 'T6-X2')

    assert shipdb.last_match['name'] == 'Terran Lexington Dreadnought Cruiser'


def test_dropped_letter_still_resolves_to_the_right_ship(shipdb):
    """The reported l1.png failure: 'Lexington' read as 'exington'."""
    shipdb.get_profile('', 'Terran exington Dreadnought Cruiser', 'T6-X2')

    assert shipdb.last_match['name'] == 'Terran Lexington Dreadnought Cruiser'
    assert shipdb.last_match_strategy == 'fuzzy-display'


def test_a_bare_class_read_does_not_claim_a_specific_ship(shipdb):
    """A read that is itself a class name must not be attributed to a ship.

    'warbird battlecruiser' is a generic class and sits 0.9048 from the ship
    name 'arbiter battlecruiser', so the name matcher would happily claim it.
    Falling through to the class match is fine; naming a ship is not.
    """
    shipdb.get_profile('', 'Warbird Battlecruiser', 'T6')

    assert shipdb.last_match_strategy != 'fuzzy-display'
    assert (shipdb.last_match or {}).get('name') != 'Arbiter Battlecruiser'


def test_a_damaged_class_read_is_also_rejected(shipdb):
    """The guard is fuzzy, so OCR damage does not smuggle a class read past
    it: 'arbird battlecruiser' is still recognisably the class."""
    shipdb.get_profile('', 'arbird Battlecruiser', 'T6')

    assert (shipdb.last_match or {}).get('name') != 'Arbiter Battlecruiser'


def test_two_word_ship_name_survives_a_dropped_letter(shipdb):
    """What the old ≥3-word guard threw away.

    'dvanced escort' is 0.9655 from its own name, but the word count alone
    disqualified it and the read ended up unmatched.
    """
    shipdb.get_profile('', 'dvanced Escort', 'T6')

    assert shipdb.last_match['name'] == 'Advanced Escort'
    assert shipdb.last_match_strategy == 'fuzzy-display'
