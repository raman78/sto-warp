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


def test_the_temporal_seat_is_written_the_way_sets_spells_it(offline_cache):
    """Every Temporal seat in the ship roster says `Temporal Operative`;
    SETS' vocabulary is `Temporal`, and its seat label is a non-editable
    combo that ignores a string it does not offer. Folded on the way out,
    the way `parse_boff_stations` folds it on the way in."""
    from warp import build_writer

    _rank, profession, spec = build_writer._get_boff_spec(
        'Lieutenant Commander Universal-Temporal Operative')

    assert (profession, spec) == ('Universal', 'Temporal')


def test_every_seat_the_roster_has_keeps_its_spec_affinity(offline_cache):
    """`_prepare_seats` feeds `_get_boff_spec`'s output to `_SPEC_TO_PROF`,
    so folding a spelling on one side and not the other costs the seat its
    specialisation affinity — silently, since a miss just means no bonus."""
    from warp import build_writer

    seen = set()
    for ship in cargo.ships().values():
        for station in build_writer._seat_strings(ship):
            spec = build_writer._get_boff_spec(station)[2]
            if spec:
                seen.add(spec)

    assert seen, 'no specialised seats in the baseline roster'
    assert seen <= set(build_writer._SPEC_TO_PROF), \
        f'no ability profession for {seen - set(build_writer._SPEC_TO_PROF)}'


def test_other_specialisations_are_left_alone(offline_cache):
    """The fold is one spelling of one spec, not a general truncation."""
    from warp import build_writer

    assert build_writer._get_boff_spec(
        'Commander Tactical-Miracle Worker')[2] == 'Miracle Worker'
    assert build_writer._get_boff_spec('Ensign Science')[2] == ''


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


def test_eleven_personal_traits_name_the_captain_alien(offline_cache):
    """Eleven personal traits are only possible for an Alien captain, and
    SETS needs the species said out loud or it hides the eleventh slot."""
    from warp.build_writer import _apply_alien_species
    from warp.data.empty_build import empty_build

    build = empty_build('full')
    for i in range(11):
        build['space']['traits'][i] = {'item': f'Trait {i}'}

    _apply_alien_species(build)

    assert build['captain']['species'] == 'Alien'
    assert build['captain']['faction'] == 'Federation'


def test_ten_personal_traits_leave_the_species_alone(offline_cache):
    """Ten is what an Elite Captain of any species gets."""
    from warp.build_writer import _apply_alien_species
    from warp.data.empty_build import empty_build

    build = empty_build('full')
    for i in range(10):
        build['space']['traits'][i] = {'item': f'Trait {i}'}

    _apply_alien_species(build)

    assert build['captain']['species'] == ''
    assert build['captain']['faction'] == ''


def test_a_faction_already_in_the_build_is_kept(offline_cache):
    """The faction is a placeholder only when there is nothing better."""
    from warp.build_writer import _apply_alien_species
    from warp.data.empty_build import empty_build

    build = empty_build('full')
    build['captain']['faction'] = 'Romulan'
    build['ground']['traits'][10] = {'item': 'Trait'}

    _apply_alien_species(build)

    assert build['captain']['species'] == 'Alien'
    assert build['captain']['faction'] == 'Romulan'


def test_a_stated_species_is_not_overwritten(offline_cache):
    """One-way, like the elite flag: a build that names its captain keeps
    that name, and the export check reports the hidden slot instead."""
    from warp.build_writer import _apply_alien_species
    from warp.data.empty_build import empty_build

    build = empty_build('full')
    build['captain']['faction'] = 'Federation'
    build['captain']['species'] = 'Vulcan'
    build['space']['traits'][10] = {'item': 'Trait'}

    _apply_alien_species(build)

    assert build['captain']['species'] == 'Vulcan'
