"""Every BOFF ability must be reachable from the slot that owns it.

Abilities are bucketed by the wiki's `type` value, and the trainer looks a
bucket up from its slot name. A `type` the slot names do not cover is not an
error anywhere — the lookup just returns nothing and those abilities quietly
cannot be picked.

That is what happened with `Temporal Operative`: the wiki uses it alongside
`Temporal` for the same specialisation, and the seven ground abilities filed
under the longer spelling were unselectable.
"""
from __future__ import annotations

import pytest

from warp.data import cargo

# The BOFF slots the trainer offers, minus the `Boff ` prefix.
# Mirrors SLOT_GROUPS in warp/trainer/constants.py.
SLOT_CAREERS = {'Tactical', 'Engineering', 'Science', 'Intelligence',
                'Command', 'Pilot', 'Miracle Worker', 'Temporal'}


def _built(raw: list[dict]) -> dict:
    return cargo._build_boff_abilities.__wrapped__(raw) if hasattr(
        cargo._build_boff_abilities, '__wrapped__') else None


@pytest.fixture
def buckets(monkeypatch):
    """Build the cache from a fixed set of rows — no network, no user cache."""
    rows = [
        {'name': 'Causal Entanglement', 'type': 'Temporal Operative',
         'region': 'Ground'},
        {'name': 'Causal Reversion', 'type': 'Temporal', 'region': 'Space'},
        {'name': 'Chronoplasty', 'type': 'Temporal Operative', 'region': 'Ground'},
        {'name': 'Overwhelm Emitters', 'type': 'Tactical', 'region': 'Space'},
    ]
    monkeypatch.setattr(cargo, '_load_raw', lambda name: rows)
    return cargo._build_boff_abilities()


def _names(bucket) -> set[str]:
    out: set[str] = set()
    for rank in bucket:
        if isinstance(rank, dict):
            out |= set(rank)
    return out


def test_the_long_spelling_lands_in_the_slot_the_trainer_asks_for(buckets):
    assert 'Causal Entanglement' in _names(buckets['ground']['Temporal'])


def test_both_spellings_share_one_bucket(buckets):
    assert 'Temporal Operative' not in buckets['ground']
    assert _names(buckets['ground']['Temporal']) == {
        'Causal Entanglement', 'Chronoplasty'}


def test_unrelated_careers_are_untouched(buckets):
    assert _names(buckets['space']['Tactical']) == {'Overwhelm Emitters'}
    assert _names(buckets['space']['Temporal']) == {'Causal Reversion'}


def test_every_career_in_the_live_cache_maps_to_a_slot():
    """Guards the real data: a new wiki spelling must not go unnoticed."""
    try:
        live = cargo.boff_abilities()
    except Exception as exc:                       # no cache, no network
        pytest.skip(f'cargo unavailable: {exc}')

    unreachable = {
        f'{env}/{career}'
        for env in ('space', 'ground')
        for career in live.get(env, {})
        if career not in SLOT_CAREERS
    }
    assert not unreachable, (
        f'abilities filed under {sorted(unreachable)} cannot be picked from '
        f'any trainer slot — add the spelling to _BOFF_CAREER_ALIASES')
