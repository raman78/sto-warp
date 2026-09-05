"""A tier the badge could not establish is settled by cargo, or not claimed.

OCR reads `[T6]` as `[TB]` — a clean 6/B substitution — and `TB` scores exactly
0.50 against every one of `T1` … `T6`. A six-way tie. The old `_fuzzy_tier`
short-circuited on `best_ratio < 0.6` and returned `scored[0][0]`, which under
a stable sort is whichever entry comes first in `SHIP_TIER_VALUES`: `T1`.

So any unreadable second character in a two-character badge became `T1`,
deterministically, and was reported at `SHIP_TIER_CONF_BADGE` (0.90) — above
the trainer's auto-accept floor, so it was confirmed into the training data
without anyone seeing it. Measured 2026-09-05, that was four of the five tier
errors over 172 screenshots, and cargo records all four ships as tier 6.

Offline: no OCR, no network. The cargo-backed halves build a ShipDB from a
temporary roster.
"""
from __future__ import annotations

import json

import pytest

pytest.importorskip("cv2")

from warp.recognition.text_extractor import _fuzzy_tier, _fuzzy_tier_ex
from warp.warp_importer import (SHIP_TIER_CONF_AMBIGUOUS,
                                SHIP_TIER_CONF_BADGE,
                                SHIP_TIER_CONF_INFERRED, ShipDB)


# ── The tie itself ─────────────────────────────────────────────────────────

@pytest.mark.parametrize('cand', ['TB', 'T8', 'TG'])
def test_an_unreadable_second_character_is_reported_as_a_tie(cand):
    _tier, ambiguous = _fuzzy_tier_ex(cand)
    assert ambiguous


@pytest.mark.parametrize('cand', ['T6', 'T6-X', 'T6-X2', 'T5-U', 'T6-XZ'])
def test_a_clean_badge_is_not_a_tie(cand):
    _tier, ambiguous = _fuzzy_tier_ex(cand)
    assert not ambiguous


def test_the_tie_still_returns_something_to_start_from():
    """A caller that can settle it needs a value; one that cannot is better
    off with a plausible tier than with none — as long as it is told."""
    tier, ambiguous = _fuzzy_tier_ex('TB')
    assert tier in ('T1', 'T2', 'T3', 'T4', 'T5', 'T6')
    assert ambiguous


def test_noise_still_produces_no_tier():
    assert _fuzzy_tier_ex('XX') == (None, False)


def test_the_old_entry_point_is_unchanged():
    """`_fuzzy_tier` keeps its signature — the flag is additive."""
    assert _fuzzy_tier('T6-X2') == 'T6-X2'
    assert _fuzzy_tier('XX') is None


# ── Cargo settles it ───────────────────────────────────────────────────────

_ROSTER = [
    {'name': 'Kor Bird-of-Prey',                 'tier': 6, 'type': ['Raider']},
    {'name': 'Deimos Pilot Destroyer',           'tier': 6, 'type': ['Destroyer']},
    {'name': 'Maelstrom Pilot Destroyer',        'tier': 6, 'type': ['Destroyer']},
    {'name': 'Nova Science Vessel',              'tier': 1, 'type': ['Science Vessel']},
    {'name': 'Rhode Island Science Vessel',      'tier': 6, 'type': ['Science Vessel']},
]


@pytest.fixture
def db(tmp_path):
    (tmp_path / 'ship_list.json').write_text(json.dumps(_ROSTER))
    return ShipDB(tmp_path)


def test_a_class_whose_ships_agree_settles_the_tier(db):
    """No single ship identified, but every `Pilot Destroyer` is a T6."""
    assert db.tier_for_class('Pilot Destroyer') == 6


def test_a_class_whose_ships_disagree_settles_nothing(db):
    """`Science Vessel` spans T1 and T6 here, as it does in the real roster."""
    assert db.tier_for_class('Science Vessel') == 0


def test_a_full_ship_name_works_too(db):
    assert db.tier_for_class('Kor Bird-of-Prey') == 6


def test_an_unknown_class_settles_nothing(db):
    assert db.tier_for_class('Borg Cube') == 0


def test_an_empty_class_settles_nothing(db):
    assert db.tier_for_class('') == 0
    assert db.tier_for_class('   ') == 0


def test_the_match_is_case_insensitive(db):
    assert db.tier_for_class('pilot destroyer') == 6


def test_a_partial_word_does_not_match(db):
    """`endswith` on the class string, not a substring search anywhere."""
    assert db.tier_for_class('Pilot') == 0


# ── What the confidences have to say ───────────────────────────────────────

def test_an_unsettled_tie_stays_under_auto_accept():
    """0.75 is the trainer's auto-accept floor; the old value was 0.90."""
    assert SHIP_TIER_CONF_AMBIGUOUS < 0.75
    assert SHIP_TIER_CONF_BADGE > 0.75


def test_a_tie_is_worth_less_than_an_inference():
    """An inference from slot counts is at least an argument. A tie is one of
    six equally-scoring options."""
    assert SHIP_TIER_CONF_AMBIGUOUS < SHIP_TIER_CONF_INFERRED
