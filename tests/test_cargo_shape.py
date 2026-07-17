"""Shape contract for the cargo cache.

Loads every `warp.data.cargo` accessor against the bundled baseline and
asserts the invariants every downstream consumer (warp_importer,
build_writer, sets_export, recognition/*) relies on.

The v1.0.16 BOFF regression slipped through because the loader's output
shape diverged from the consumer contract, and ~20 `try/except: pass`
sites along the way swallowed the AttributeError silently. This test
makes the contract explicit so the same class of port-mismatch fails
the build instead of degrading recognition for weeks.

Run standalone:
    python -m pytest tests/test_cargo_shape.py -v
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_cargo(monkeypatch):
    """Force cargo to load only from the bundled baseline.

    Points WARP_CACHE_DIR at an empty temp dir so the user's actual
    cache (which may carry stale or hand-edited files) does not affect
    the test. Clears the in-process memo so each test gets a fresh
    load.
    """
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setenv('WARP_CACHE_DIR', tmp)
        from warp.data import cargo
        cargo._MEMO.clear()
        cargo._BUCKET_MEMO.clear()
        yield
        cargo._MEMO.clear()
        cargo._BUCKET_MEMO.clear()


def test_validate_passes_against_baseline():
    """`cargo.validate(on_problem='raise')` raises on the first shape
    drift; this test guards every consumer-visible invariant in one shot.
    """
    from warp.data import cargo
    problems = cargo.validate(on_problem='raise')
    assert problems == []


def test_boff_abilities_bucketed_shape():
    """Regression guard for v1.0.16: the upstream JSON is a flat list but
    every consumer expects `{space|ground: {prof: [rank_dict]}, all: {}}`.
    """
    from warp.data.cargo import boff_abilities
    bo = boff_abilities()
    assert set(bo.keys()) >= {'space', 'ground', 'all'}

    # `all` is a flat name→info map with a 'profession' field — that's
    # what `warp_importer.py:2119` reads.
    assert bo['all'], 'boff_abilities[all] empty — baseline drift?'
    for name, info in bo['all'].items():
        assert isinstance(info, dict), f'{name!r} info not dict'
        assert info.get('profession'), f'{name!r} missing profession'

    # Per-env buckets are `{prof: [{ability_name: dict}]}` — that's what
    # `_lookup_boff_profession`, `_item_valid_for_slot`, build_writer
    # iterate.
    for env in ('space', 'ground'):
        for prof, ranks in bo[env].items():
            assert isinstance(ranks, list) and ranks
            assert isinstance(ranks[0], dict)
            for ab_name in ranks[0]:
                assert isinstance(ab_name, str)


def test_equipment_bucketed_by_build_key():
    """`warp_importer._item_valid_for_slot` iterates
    `eq_cache.values()` and reads `entry.get('type')` on each item dict.
    """
    from warp.data.cargo import equipment
    eq = equipment()
    assert eq, 'equipment empty'
    for bucket, items in eq.items():
        assert isinstance(items, dict), f'equipment[{bucket!r}] not dict'
        if not items:
            continue
        sample_name, sample_entry = next(iter(items.items()))
        assert isinstance(sample_entry, dict), \
            f'equipment[{bucket!r}][{sample_name!r}] not dict'


def test_traits_env_and_kind():
    """`trait_grid` / `warp_importer` read
    `traits[env][trait_type][name]` where `env` ∈ {space, ground} and
    `trait_type` ∈ {personal, rep, active_rep}.
    """
    from warp.data.cargo import traits
    tr = traits()
    for env in ('space', 'ground'):
        assert env in tr
        for kind in ('personal', 'rep', 'active_rep'):
            assert kind in tr[env], f'traits[{env}] missing {kind}'


def test_ships_keyed_by_page():
    """`build_writer` reads `cache.ships[ship_name]['tier']`."""
    from warp.data.cargo import ships
    ss = ships()
    assert ss, 'ships empty'
    sample = next(iter(ss.values()))
    assert isinstance(sample, dict)


def test_starship_traits_flat():
    """`warp_importer._item_valid_for_slot` does `name in
    starship_traits` — must be a flat name-keyed dict."""
    from warp.data.cargo import starship_traits
    st = starship_traits()
    assert st, 'starship_traits empty'
    sample_name = next(iter(st))
    assert isinstance(sample_name, str)


def test_canonical_names_unions_every_source():
    """`canonical_names()` is the flat validation set consumed by external
    maintainer tooling (crop relabeling). It must union every source and
    never contain empties, so a real item/ability/trait/specialization
    validates and a typo does not."""
    from warp.data import cargo

    names = cargo.canonical_names()
    assert names, 'canonical_names empty'
    assert all(isinstance(n, str) and n for n in names), 'empty / non-str name'

    # Captain specializations are folded in (not present in cargo JSON).
    assert cargo.SPECIALIZATION_NAMES <= names

    # A sample from each cargo source resolves through the union.
    eq_sample = next(iter(next(iter(cargo.equipment().values()))))
    assert eq_sample in names, 'equipment name missing from canonical set'
    boff_sample = next(iter(cargo.boff_abilities()['all']))
    assert boff_sample in names, 'boff ability missing from canonical set'
    st_sample = next(iter(cargo.starship_traits()))
    assert st_sample in names, 'starship trait missing from canonical set'
