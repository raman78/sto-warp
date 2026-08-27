"""The SETS build-JSON contract, enforced.

Covers three layers: the frozen contract in `warp/data/sets_contract.json`,
the value rules in `warp/sets_schema.py`, and the export hook in
`warp/sets_export.py`. The last test compares the contract against a real
SETS install and is skipped when one isn't present — that comparison is
the workflow's job (`.github/workflows/sets-contract-watch.yml`).
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from warp import sets_schema
from warp.data.empty_build import empty_build
from warp.sets_export import write_sets_build


@pytest.fixture
def clean_build() -> dict:
    """Skeleton fixed up the way a correct export should look.

    Two departures from `warp.data.empty_build`: `_version` (added by
    `sets_export`) and seat specs as strings rather than `None`.
    """
    build = empty_build('full')
    build['_version'] = sets_schema.contract()['build_version']
    build['space']['boff_specs'] = [['Tactical', ''] for _ in range(6)]
    return build


@pytest.fixture
def stub_cache():
    """Cargo view holding exactly one name per bucket."""
    return SimpleNamespace(
        equipment={'fore_weapons': {'Phaser Beam Array': {}}},
        traits={'space': {'personal': {'Astrophysicist': {}}, 'rep': {}, 'active_rep': {}},
                'ground': {'personal': {}, 'rep': {}, 'active_rep': {}}},
        starship_traits={'Emergency Weapon Cycle': {}},
        boff_abilities={'all': {'Emergency Power to Weapons': {
            'rank1rank': 'Ensign', 'rank2rank': 'Lieutenant',
            'rank3rank': 'Lt. Commander'}}},
    )


def test_clean_build_has_no_violations(clean_build):
    assert sets_schema.validate_sets_build(clean_build) == []


def test_null_seat_spec_is_an_error(clean_build):
    """The bug this module was written for: SETS renders 'None / None'."""
    clean_build['space']['boff_specs'][2] = [None, None]

    rules = {(v.path, v.rule) for v in sets_schema.validate_sets_build(clean_build)}

    assert ('/space/boff_specs[2][0]', 'seat_profession') in rules
    assert ('/space/boff_specs[2][1]', 'seat_specialisation') in rules


def test_bare_skeleton_is_export_clean():
    """`empty_build` plus the version stamp must already satisfy SETS."""
    build = empty_build('full')
    build['_version'] = sets_schema.contract()['build_version']

    assert sets_schema.validate_sets_build(build) == []


def test_empty_seat_spec_is_allowed(clean_build):
    """`['', '']` is how SETS stores a seat the ship doesn't have."""
    clean_build['space']['boff_specs'][5] = ['', '']

    assert sets_schema.validate_sets_build(clean_build) == []


def test_a_specialisation_sets_does_not_know_is_caught(clean_build):
    """The seat label is a non-editable combo: `setCurrentText` with a
    string it does not offer is a no-op, so the seat quietly keeps the
    wrong label. `Temporal Operative` is how the wiki spells the one SETS
    calls `Temporal`."""
    clean_build['space']['boff_specs'][0] = ['Tactical', 'Temporal Operative']

    rules = {(v.path, v.rule) for v in sets_schema.validate_sets_build(clean_build)}

    assert ('/space/boff_specs[0][1]', 'seat_specialisation') in rules


def test_the_spelling_sets_uses_passes(clean_build):
    clean_build['space']['boff_specs'][0] = ['Tactical', 'Temporal']

    assert sets_schema.validate_sets_build(clean_build) == []


def test_pilot_is_a_space_seat_only(clean_build):
    """`GROUND_BOFF_SPECS` has no Pilot — a ground seat asking for one
    would land on a combo that cannot show it."""
    clean_build['ground']['boff_specs'][0] = 'Pilot'

    rules = {(v.path, v.rule) for v in sets_schema.validate_sets_build(clean_build)}

    assert ('/ground/boff_specs[0]', 'seat_specialisation') in rules


def test_missing_version_is_a_shape_violation():
    violations = sets_schema.validate_sets_build(empty_build('full'))

    assert any(v.path == '/' and v.rule == 'shape' for v in violations)


def test_short_slot_list_is_a_shape_violation(clean_build):
    clean_build['space']['fore_weapons'].pop()

    violations = sets_schema.validate_sets_build(clean_build)

    assert any(v.path == '/space/fore_weapons' and v.rule == 'shape' for v in violations)


def test_extra_section_key_is_a_shape_violation(clean_build):
    clean_build['space']['skill_desc'] = ''

    violations = sets_schema.validate_sets_build(clean_build)

    assert any(v.path == '/space' and v.rule == 'shape' for v in violations)


def test_unknown_rarity_is_an_error(clean_build):
    clean_build['space']['fore_weapons'][0] = {
        'item': 'Phaser Beam Array', 'rarity': 'Legendary',
        'mark': 'XV', 'modifiers': [None] * 4}

    violations = sets_schema.validate_sets_build(clean_build)

    assert any(v.rule == 'rarity' for v in violations)


def test_item_missing_from_cargo_is_flagged(clean_build, stub_cache):
    """SETS deletes what it cannot resolve — silently."""
    clean_build['space']['fore_weapons'][0] = {
        'item': 'Nonexistent Beam Array', 'rarity': 'Rare',
        'mark': 'XV', 'modifiers': [None] * 4}

    violations = sets_schema.validate_sets_build(clean_build, stub_cache)

    assert any(v.rule == 'unknown_item' for v in violations)


def test_known_item_passes_the_cargo_check(clean_build, stub_cache):
    clean_build['space']['fore_weapons'][0] = {
        'item': 'Phaser Beam Array', 'rarity': 'Rare',
        'mark': 'XV', 'modifiers': [None] * 4}

    violations = sets_schema.validate_sets_build(clean_build, stub_cache)

    assert not [v for v in violations if v.rule in ('unknown_item', 'item_keys')]


def test_ability_without_rank_is_an_error(clean_build):
    clean_build['space']['boffs'][0][0] = {'item': 'Emergency Power to Weapons'}

    violations = sets_schema.validate_sets_build(clean_build)

    assert any(v.rule == 'ability_rank' for v in violations)


def test_ability_at_a_rank_the_slot_never_offers_is_a_warning(clean_build, stub_cache):
    """Rank II of this ability unlocks at Lieutenant — slot index 1."""
    clean_build['space']['boffs'][0][0] = {
        'item': 'Emergency Power to Weapons', 'rank': 'II'}

    violations = sets_schema.validate_sets_build(clean_build, stub_cache)
    rank_slot = [v for v in violations if v.rule == 'rank_slot']

    assert len(rank_slot) == 1
    assert rank_slot[0].severity == 'warning'


def test_rank_matching_the_slot_is_accepted(clean_build, stub_cache):
    clean_build['space']['boffs'][0][0] = {
        'item': 'Emergency Power to Weapons', 'rank': 'I'}

    violations = sets_schema.validate_sets_build(clean_build, stub_cache)

    assert not [v for v in violations if v.rule.startswith(('ability', 'rank', 'unknown'))]


def _broken_build() -> dict:
    build = empty_build('full')
    build['space']['boff_specs'][0] = [None, None]
    return build


def test_export_writes_the_file_even_when_violations_exist(tmp_path):
    """A user who asked for an export gets one; the log carries the rest."""
    out = tmp_path / 'build.json'
    violations = []

    write_sets_build(_broken_build(), out, report_to=violations)

    assert json.loads(out.read_text(encoding='utf-8'))['_version'] == 1
    assert [v for v in violations if v.rule == 'seat_profession']


def test_strict_mode_refuses_to_write(tmp_path, monkeypatch):
    monkeypatch.setenv('WARP_STRICT_EXPORT', '1')
    out = tmp_path / 'build.json'

    with pytest.raises(ValueError, match='SETS schema violations'):
        write_sets_build(_broken_build(), out)

    assert not out.exists()


def test_strict_mode_writes_a_clean_build(tmp_path, monkeypatch, clean_build):
    monkeypatch.setenv('WARP_STRICT_EXPORT', '1')
    out = tmp_path / 'build.json'

    write_sets_build(clean_build, out)

    assert out.exists()


def test_issue_url_stays_within_github_limits():
    violations = [sets_schema.Violation(f'/space/fore_weapons[{i}]', 'unknown_item',
                                        'name present in cargo', f"'Item {i}'")
                  for i in range(200)]

    url = sets_schema.issue_url(violations, {'sto-warp': '1.0.29'})

    assert len(url) <= sets_schema._MAX_URL_LEN
    assert url.startswith(sets_schema.ISSUE_BASE_URL + '?')
    assert 'labels=sets-schema' in url


def _sets_empty_build():
    try:
        from src.buildhelpers import empty_build as sets_empty_build
    except Exception:
        return None
    return sets_empty_build


@pytest.mark.skipif(_sets_empty_build() is None, reason='SETS not installed')
def test_contract_matches_the_installed_sets():
    """Local mirror of the contract-watch workflow."""
    stored = sets_schema.contract()['shape']

    live = sets_schema.shape_of(_sets_empty_build()('full'))

    assert live == stored


class _OverlayCache:
    """Cargo view whose ground weapon came from the harvested overlay."""

    equipment = {'weapons': {
        'Elite Fleet Colony Security Antiproton Blast Assault':
            {'name': 'Elite Fleet Colony Security Antiproton Blast Assault',
             'type': 'Ground Weapon', 'source': 'listing-scrape'},
        'Phaser Compression Pistol': {'name': 'Phaser Compression Pistol'},
    }}
    traits = {'space': {'personal': {}, 'rep': {}, 'active_rep': {}},
              'ground': {'personal': {}, 'rep': {}, 'active_rep': {}}}
    starship_traits = {}
    boff_abilities = {'all': {}}


def _ground_weapon(build, name):
    build['ground']['weapons'][0] = {'item': name, 'rarity': 'Ultra Rare',
                                     'mark': 'Mk XII', 'modifiers': [None] * 4}
    return build


def test_an_item_only_the_overlay_knows_is_reported(clean_build):
    """WARP resolves it; SETS reads the tables that omit it and drops it.

    Adding the overlay made a build carrying one of these come back clean
    while SETS still stripped the weapon — the exact prediction the
    validator exists to make.
    """
    build = _ground_weapon(
        clean_build, 'Elite Fleet Colony Security Antiproton Blast Assault')

    violations = sets_schema.validate_sets_build(build, _OverlayCache())

    assert [v.rule for v in violations] == ['not_in_sets']


def test_an_item_the_cargo_tables_carry_is_not_reported(clean_build):
    build = _ground_weapon(clean_build, 'Phaser Compression Pistol')

    assert sets_schema.validate_sets_build(build, _OverlayCache()) == []


def test_an_unknown_item_is_still_unknown(clean_build):
    """`not_in_sets` narrows the old check, it does not replace it."""
    build = _ground_weapon(clean_build, 'Nonexistent Rifle')

    violations = sets_schema.validate_sets_build(build, _OverlayCache())

    assert [v.rule for v in violations] == ['unknown_item']


# ── The eleventh personal trait ────────────────────────────────────────

def test_an_eleventh_trait_without_an_alien_captain_is_caught(clean_build):
    """Only an Alien captain has eleven personal trait slots. SETS hides the
    eleventh for anyone else, so the trait sits in the file and vanishes from
    the UI."""
    clean_build['space']['traits'][10] = {'item': 'Astrophysicist'}

    rules = {(v.path, v.rule) for v in sets_schema.validate_sets_build(clean_build)}

    assert ('/space/traits[10]', 'alien_trait_slot') in rules


def test_an_eleventh_trait_with_an_alien_captain_passes(clean_build):
    clean_build['captain']['faction'] = 'Federation'
    clean_build['captain']['species'] = 'Alien'
    clean_build['space']['traits'][10] = {'item': 'Astrophysicist'}

    assert sets_schema.validate_sets_build(clean_build) == []


def test_a_species_with_no_faction_is_caught(clean_build):
    """The species dropdown is filled from the faction; with none set it is
    empty and the species is dropped on load without a word."""
    clean_build['captain']['species'] = 'Alien'

    rules = {(v.path, v.rule) for v in sets_schema.validate_sets_build(clean_build)}

    assert ('/captain/species', 'species') in rules


def test_a_species_the_faction_does_not_offer_is_caught(clean_build):
    clean_build['captain']['faction'] = 'Dominion'
    clean_build['captain']['species'] = 'Alien'

    rules = {(v.path, v.rule) for v in sets_schema.validate_sets_build(clean_build)}

    assert ('/captain/species', 'species') in rules


def test_a_faction_sets_does_not_know_is_caught(clean_build):
    """Not a silent mismatch: SETS indexes its species table with the faction
    outright, so an unknown one raises while the build is loading."""
    clean_build['captain']['faction'] = 'Terran Empire'

    rules = {(v.path, v.rule) for v in sets_schema.validate_sets_build(clean_build)}

    assert ('/captain/faction', 'faction') in rules


def test_every_faction_offers_at_least_one_species():
    """A faction with no species would make its captains unrepresentable."""
    assert set(sets_schema.SPECIES) == set(sets_schema.FACTIONS)
    assert all(sets_schema.SPECIES[f] for f in sets_schema.FACTIONS)


# ── Mirrored vocabularies ──────────────────────────────────────────────
#
# `warp/data/sets_contract.json` snapshots the SETS constants this module
# copies by value. The weekly contract watch compares that snapshot against
# the newest SETS release; these compare the snapshot against the copies. One
# side catches SETS moving, the other catches the copy going stale — a
# vocabulary needs both, because either alone still lets the two drift.

def _vocab() -> dict:
    return sets_schema.contract()['vocabularies']


def test_the_mirrored_factions_match_the_contract():
    assert sets_schema.FACTIONS == set(_vocab()['FACTIONS'])


def test_the_mirrored_species_match_the_contract():
    snapshot = {k: set(v) for k, v in _vocab()['SPECIES'].items()}

    assert {k: set(v) for k, v in sets_schema.SPECIES.items()} == snapshot


def test_the_mirrored_seat_specialisations_match_the_contract():
    assert sets_schema.SPACE_SPECS == set(_vocab()['PRIMARY_SPECS'])
    assert sets_schema.GROUND_SPECS == set(_vocab()['GROUND_BOFF_SPECS'])


def test_the_faction_the_exporter_writes_still_exists():
    """The exporter names a faction so an eleven-trait build displays. That
    choice is only sound while SETS still offers it — a founding faction and
    a catch-all species are unlikely to move, and nothing else here would
    notice if they did."""
    from warp.build_writer import _ALIEN_FACTION

    vocab = _vocab()

    assert _ALIEN_FACTION in vocab['FACTIONS']
    assert 'Alien' in vocab['SPECIES'][_ALIEN_FACTION]
