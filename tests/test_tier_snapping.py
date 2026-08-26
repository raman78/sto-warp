"""Snapping an OCR tier bracket to a canonical tier.

The bracket in `Mirror Strike Wing Escort [T6-X]` is the game's own tier
delimiter, so its contents are fuzzy-matched against `SHIP_TIER_VALUES`
rather than read literally — OCR mangles the short suffix constantly.

What makes this worth its own file: the tier is not a label, it is a slot
count. `_apply_ship_and_tier_bonuses` grants +1 Universal Console, Device and
Starship Trait for `-X` and +2 for `-X2`, so promoting a tier by one step
invents three rows the ship does not have, and the layout detector then draws
boxes for them on every run.
"""
from __future__ import annotations

import pytest

from warp.recognition.text_extractor import SHIP_TIER_VALUES, _fuzzy_tier


@pytest.mark.parametrize('token', ['T6', 'T6-X', 'T6-X2', 'T5-U', 'T5-X2'])
def test_a_clean_token_snaps_to_itself(token):
    assert _fuzzy_tier(token) == token


@pytest.mark.parametrize('token', ['TG-X', 'TB-X', 'T8-X', 'TS-X'])
def test_a_mangled_digit_does_not_promote_the_suffix(token):
    """The bug this file exists for: `[T6-X]` read as `[TG-X]` came out as
    `T6-X2`, and the layout then asked for three rows too many."""
    assert _fuzzy_tier(token) == 'T6-X'


@pytest.mark.parametrize('token', ['T6-XZ', 'T6-XL', 'TB-X2', 'T8-X2'])
def test_a_mangled_x2_still_recovers_its_suffix(token):
    """The case the "prefer higher tier" tiebreaker was written for: an X2
    ship whose `2` was misread must not be demoted to plain `-X`."""
    assert _fuzzy_tier(token) == 'T6-X2'


@pytest.mark.parametrize('token', ['TS-U', 'T5-U'])
def test_the_t5_upgrade_survives_a_mangled_digit(token):
    assert _fuzzy_tier(token) == 'T5-U'


def test_the_tiebreaker_still_prefers_the_higher_tier():
    """Same length, equal score, so the rule that erred upward still does:
    `T5-X` and `T6-X` both score 0.75 against `TG-X`, and T6 is what STO
    actually sells."""
    assert _fuzzy_tier('TG-X') == 'T6-X'


def test_noise_snaps_to_nothing():
    assert _fuzzy_tier('Hangars') is None


def test_every_canonical_value_is_reachable():
    """A rule that made some tier unreachable would be silent — the ship
    would just come out one upgrade off."""
    assert {_fuzzy_tier(v) for v in SHIP_TIER_VALUES} == set(SHIP_TIER_VALUES)
