"""`wiki_url` must build the title from the row's `Page`, not from the name.

`Page` is cargo's alias for the MediaWiki built-in `_pageName` (see
warp-cargo-bay `fetcher/cargo_api.py`), i.e. the page the row was extracted
from — so it is the one title guaranteed to resolve. A display name is not a
title: traits and abilities carry a disambiguator the name does not, and an
item without a page of its own is documented on its set's.

Measured against the live wiki on 2026-08-31, sampling each table, the
name-based title resolved for 0/40 starship traits, 4/40 BOFF abilities,
10/40 personal traits and 85/100 equipment items. Every `Page` title
resolved.

Run standalone:
    python -m pytest tests/test_wiki_url.py -v
"""
from __future__ import annotations

import pytest

import warp.data.cargo as cargo


# One name deliberately appears as both a space and a ground trait — that is
# the case a name alone cannot answer, and 135 real trait names are in it.
_TRAITS = [
    {'name': 'Harmonic Shield Linkage', 'type': 'char', 'environment': 'space',
     'Page': 'Harmonic Shield Linkage (space trait)'},
    {'name': 'Adaptive Offense', 'type': 'char', 'environment': 'space',
     'Page': 'Adaptive Offense (space trait)'},
    {'name': 'Adaptive Offense', 'type': 'char', 'environment': 'ground',
     'Page': 'Adaptive Offense (ground trait)'},
    {'name': 'Precision', 'type': 'reputation', 'environment': 'space',
     'Page': 'Precision (space trait)'},
]
_STARSHIP_TRAITS = [
    {'name': 'Thirst for Battle', 'Page': 'Thirst for Battle (starship trait)'},
]
_ABILITIES = [
    {'name': 'Heisenberg Amplifier', 'type': 'Science',
     'Page': 'Heisenberg Amplifier (ability)'},
]
_EQUIPMENT = [
    # An item with no page of its own: the row lives on the set's page.
    {'name': 'Console - Universal - Causal Anchor',
     'type': 'Ship Engineering Console',
     'Page': '31st Century Temporal Technologies Set'},
]

_RAW = {
    'traits.json': _TRAITS,
    'starship_traits.json': _STARSHIP_TRAITS,
    'boff_abilities.json': _ABILITIES,
    'equipment.json': _EQUIPMENT,
}


@pytest.fixture(autouse=True)
def _stub_cargo(monkeypatch):
    """Serve the fixed rows above instead of the cache or the network."""
    monkeypatch.setattr(cargo, '_load_raw', lambda src: list(_RAW.get(src, [])))
    monkeypatch.setattr(cargo, '_MEMO', {})
    monkeypatch.setattr(cargo, '_BUCKET_MEMO', {})


def _title(url: str) -> str:
    assert url.startswith('https://stowiki.net/wiki/')
    return url.rsplit('/wiki/', 1)[1]


# ── The reported failure ───────────────────────────────────────────────────

def test_a_space_trait_links_to_its_disambiguated_page():
    assert _title(cargo.wiki_url('Harmonic Shield Linkage',
                                 'Personal Space Traits')) == \
        'Harmonic_Shield_Linkage_(space_trait)'


def test_a_boff_ability_links_to_its_disambiguated_page():
    assert _title(cargo.wiki_url('Heisenberg Amplifier', 'Boff Science')) == \
        'Heisenberg_Amplifier_(ability)'


def test_a_starship_trait_links_to_its_disambiguated_page():
    assert _title(cargo.wiki_url('Thirst for Battle', 'Starship Traits')) == \
        'Thirst_for_Battle_(starship_trait)'


# ── The slot is what tells two identical names apart ───────────────────────

def test_the_ground_slot_picks_the_ground_page():
    assert _title(cargo.wiki_url('Adaptive Offense',
                                 'Personal Ground Traits')) == \
        'Adaptive_Offense_(ground_trait)'


def test_the_space_slot_picks_the_space_page():
    assert _title(cargo.wiki_url('Adaptive Offense',
                                 'Personal Space Traits')) == \
        'Adaptive_Offense_(space_trait)'


def test_a_reputation_slot_reaches_the_reputation_bucket():
    """Reputation traits are filed under `rep`, not `personal` — a slot that
    resolved only the personal bucket would miss every one of them."""
    assert _title(cargo.wiki_url('Precision', 'Space Reputation')) == \
        'Precision_(space_trait)'


# ── Everything else ────────────────────────────────────────────────────────

def test_an_item_without_its_own_page_links_to_the_set_page():
    assert _title(cargo.wiki_url('Console - Universal - Causal Anchor',
                                 'Engineering Consoles')) == \
        '31st_Century_Temporal_Technologies_Set'


def test_a_name_cargo_does_not_know_keeps_the_name():
    """Skill nodes and specialisations are not in any cargo table; the name
    is all there is to go on."""
    assert _title(cargo.wiki_url('Hull Restoration', 'Skills')) == \
        'Hull_Restoration'


def test_a_wrong_slot_still_finds_the_row():
    """A misrecognised item is shown under the wrong slot, and checking it on
    the wiki is exactly what the user does about that."""
    assert _title(cargo.wiki_url('Heisenberg Amplifier', 'Tactical Consoles')) \
        == 'Heisenberg_Amplifier_(ability)'


def test_parentheses_are_not_percent_encoded():
    """Both forms resolve, but the wiki writes its own links unencoded and
    the user compares the address bar against a link they were sent."""
    assert '%28' not in cargo.wiki_url('Heisenberg Amplifier', 'Boff Science')


def test_an_equipment_slot_prefers_the_equipment_table():
    """An equipment slot reads equipment first, so a trait sharing the name
    cannot answer for a console. No such overlap exists in cargo today; the
    ordering is what keeps it that way if one appears."""
    _EQUIPMENT.append({'name': 'Adaptive Offense',
                       'type': 'Ship Tactical Console',
                       'Page': 'Adaptive Offense Console'})
    try:
        cargo._BUCKET_MEMO.clear()
        cargo._MEMO.clear()
        assert _title(cargo.wiki_url('Adaptive Offense', 'Tactical Consoles')) \
            == 'Adaptive_Offense_Console'
    finally:
        _EQUIPMENT.pop()
