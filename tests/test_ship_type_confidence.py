"""Ship Type confidence must reflect what the lookup actually established.

Ship Type used to be emitted at a flat 1.0 regardless of how ShipDB reached
it. Four of the lookup strategies read `_by_type`, which is keyed on the
generic `type` field and therefore holds an arbitrary member of the class —
measured, its slot profile is wrong on 4.9 of 11 slots for the ship really on
screen. Reporting that as certain meant WARP CORE auto-accepted it into
`annotations.json` as ground truth.

Run standalone:
    python -m pytest tests/test_ship_type_confidence.py -v
"""
from __future__ import annotations

import json

import pytest

# The lowest value WARP CORE's auto-accept spinbox allows
# (`trainer_window.py`: `_spin_auto_conf.setRange(0.5, 1.0)`). Anything a user
# must review by hand has to sit strictly below this, not at it.
MIN_AUTO_ACCEPT_SETTING = 0.5


def _resolution(strategy: str, matched: bool = True):
    from warp.warp_importer import ShipResolution

    return ShipResolution(
        name='', type='Some Cruiser', tier='T6', profile={},
        strategy=strategy, matched=matched, ocr_name='', ocr_type='',
    )


@pytest.mark.parametrize('strategy', [
    'exact-type', 'word-subset', 'word-subset-best', 'fuzzy-type',
])
def test_class_only_strategies_are_not_auto_acceptable(strategy):
    """These four resolve to a class and hand back an arbitrary member."""
    from warp.warp_importer import ship_type_confidence

    assert ship_type_confidence(_resolution(strategy)) < MIN_AUTO_ACCEPT_SETTING


@pytest.mark.parametrize('strategy', [
    'display-name', 'display-name-best', 'fuzzy-display',
    'token-overlap', 'anchorless-rescue',
])
def test_ship_level_strategies_stay_fully_confident(strategy):
    """These identify a specific ship_list.json entry — unchanged behaviour."""
    from warp.warp_importer import ship_type_confidence

    assert ship_type_confidence(_resolution(strategy)) == 1.0


def test_unmatched_lookup_is_unverified():
    from warp.warp_importer import ship_type_confidence

    conf = ship_type_confidence(_resolution('keyword-fallback', matched=False))

    assert conf < MIN_AUTO_ACCEPT_SETTING


def test_absent_resolution_is_unverified():
    """Trait-only panels skip the ShipDB lookup entirely."""
    from warp.warp_importer import ship_type_confidence

    assert ship_type_confidence(None) < MIN_AUTO_ACCEPT_SETTING


def test_class_only_still_outranks_a_bare_ocr_read():
    """Knowing the right class is worth more than knowing nothing, so the two
    low bands stay ordered — the review list sorts on this."""
    from warp.warp_importer import ship_type_confidence

    assert (ship_type_confidence(_resolution('fuzzy-type'))
            > ship_type_confidence(None))


def _ship(name: str, stype: str) -> dict:
    return {
        'Page': name, 'name': name, 'type': stype,
        'boffs': 'Commander Tactical,Ensign Science',
        'tier': '6', 'fore': '4', 'hangars': '',
    }


def test_a_real_class_read_lands_below_auto_accept(tmp_path):
    """End to end through ShipDB: reading only the class must not produce an
    auto-acceptable Ship Type, whichever ship the index happens to hold."""
    from warp.warp_importer import ShipDB, ship_type_confidence

    (tmp_path / 'ship_list.json').write_text(json.dumps([
        _ship('Zahl Heavy Cruiser', 'Cruiser'),
        _ship('Excelsior Cruiser', 'Cruiser'),
    ]), encoding='utf-8')

    resolution = ShipDB(tmp_path).resolve('', 'Cruiser', 'T6')

    assert resolution.strategy in {'exact-type', 'word-subset',
                                   'word-subset-best', 'fuzzy-type'}
    assert ship_type_confidence(resolution) < MIN_AUTO_ACCEPT_SETTING


def test_a_named_ship_read_stays_confident(tmp_path):
    from warp.warp_importer import ShipDB, ship_type_confidence

    (tmp_path / 'ship_list.json').write_text(json.dumps([
        _ship('Zahl Heavy Cruiser', 'Cruiser'),
        _ship('Excelsior Cruiser', 'Cruiser'),
    ]), encoding='utf-8')

    resolution = ShipDB(tmp_path).resolve('', 'Excelsior Cruiser', 'T6')

    assert resolution.type == 'Excelsior Cruiser'
    assert ship_type_confidence(resolution) == 1.0
