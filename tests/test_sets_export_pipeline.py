"""End-to-end format check: ImportResult → build_writer → SETS JSON.

Runs entirely offline against the bundled cargo baseline, so it needs
neither the ML stack nor the network — the format contract is exercised
on every push, while recognition quality stays a separate concern.
"""
from __future__ import annotations

import pytest

from warp import sets_schema
from warp.build_writer import build_from_result
from warp.data import cargo
from warp.sets_export import build_sets_v3_dict
from warp.warp_importer import ImportResult, RecognisedItem


@pytest.fixture
def offline_cache(tmp_path, monkeypatch):
    """Cargo view that can only answer from `warp/data/baseline/`."""
    monkeypatch.setenv('WARP_CACHE_DIR', str(tmp_path / 'cache'))

    def _no_network(name):
        raise RuntimeError('offline test')

    monkeypatch.setattr(cargo, '_fetch', _no_network)
    return cargo.cache_view()


def _item(slot: str, index: int, name: str) -> RecognisedItem:
    return RecognisedItem(slot=slot, slot_index=index, name=name,
                          confidence=0.9, bbox=(0, 0, 10, 10))


@pytest.fixture
def space_result(offline_cache) -> ImportResult:
    """A small but real space build — names taken from the baseline cargo."""
    equipment = offline_cache.equipment
    fore = sorted(equipment['fore_weapons'])[:2]
    console = sorted(equipment['tac_consoles'])[:1]
    return ImportResult(
        build_type='SPACE',
        ship_type='Avenger Battlecruiser',
        ship_tier='T6',
        items=[_item('Fore Weapons', i, name) for i, name in enumerate(fore)]
              + [_item('Tactical Consoles', 0, console[0])],
    )


def test_exported_build_matches_the_sets_contract(space_result, offline_cache):
    build, _ = build_from_result(space_result, cache=offline_cache)
    payload = build_sets_v3_dict(build, offline_cache)

    assert sets_schema.validate_sets_build(payload, offline_cache) == []


def test_seat_specs_come_from_the_ship_layout(space_result, offline_cache):
    """The Avenger seats: Cmdr Eng, LtCmdr Tac, Lt Universal, Lt Sci, Ens Tac.

    Universal defaults to Tactical the way SETS' own seat setup does, and
    the sixth seat the ship doesn't have stays empty.
    """
    build, _ = build_from_result(space_result, cache=offline_cache)

    assert build['space']['boff_specs'] == [
        ['Engineering', ''], ['Tactical', ''], ['Tactical', ''],
        ['Science', ''], ['Tactical', ''], ['', ''],
    ]


def test_seat_specs_survive_a_comma_joined_cargo_field(offline_cache, monkeypatch):
    """`raman78/warp-cargo-data` serves `boffs` as one comma-joined string."""
    from warp import build_writer

    flat = {'boffs': 'Commander Tactical-Miracle Worker,Ensign Science'}

    assert build_writer._seat_strings(flat) == [
        'Commander Tactical-Miracle Worker', 'Ensign Science']


def test_written_items_survive_the_cargo_lookup(space_result, offline_cache):
    """Names WARP writes must be names SETS can resolve, or it drops them."""
    build, report = build_from_result(space_result, cache=offline_cache)
    payload = build_sets_v3_dict(build, offline_cache)

    violations = sets_schema.validate_sets_build(payload, offline_cache)

    assert report.n_equipment == 3
    assert not [v for v in violations if v.rule.startswith('unknown_')]


def test_ability_rank_follows_the_seat_it_unlocks_at(offline_cache):
    """Rank tiers are not one-per-slot: cargo's `rank<N>rank` decides.

    `Cannons: Rapid Fire` unlocks I/II/III at Lieutenant / Lt. Commander /
    Commander, so slot 1 must carry rank I — not the highest rank the
    slot could theoretically hold.
    """
    from warp.sets_export import _resolve_rank

    assert _resolve_rank('Cannons: Rapid Fire', 1, offline_cache) == 'I'
    assert _resolve_rank('Cannons: Rapid Fire', 2, offline_cache) == 'II'
    assert _resolve_rank('Cannons: Rapid Fire', 3, offline_cache) == 'III'


def test_ability_that_starts_above_a_slot_keeps_its_lowest_rank(offline_cache):
    """`Aceton Beam` has no tier below Lt. Commander (slot 2)."""
    from warp.sets_export import _resolve_rank

    assert _resolve_rank('Aceton Beam', 2, offline_cache) == 'I'
    assert _resolve_rank('Aceton Beam', 3, offline_cache) == 'III'
