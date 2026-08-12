"""Recovering the T6-X / T6-X2 bonus when the tier badge is off-screen.

Many screenshots do not show `[T6-X2]`, so `ship_tier` is empty and no bonus
is applied: Universal Consoles drops to 0 (the layout detector then skips the
whole row), Devices and Starship Traits lose up to 2 each.

`_infer_x_bonus` recovers it from what the run already measured. Every piece
of evidence is a LOWER bound — a pixel count only sees filled slots — so the
rule is max, not majority.
"""
from __future__ import annotations

from warp.warp_importer import _infer_x_bonus


def _profile(devices=4, universal=0):
    return {'Devices': devices, 'Universal Consoles': universal,
            'Starship Traits': 5}


def test_no_evidence_leaves_the_profile_alone():
    assert _infer_x_bonus(_profile(), {}, {}) == 0


def test_measurements_matching_the_profile_mean_no_bonus():
    px = {'Devices': 4, 'Universal Consoles': 0}
    assert _infer_x_bonus(_profile(), px, {'Starship Traits': [0] * 5}) == 0


def test_two_extra_devices_and_consoles_infer_x2():
    px = {'Devices': 6, 'Universal Consoles': 2}
    assert _infer_x_bonus(_profile(), px, {}) == 2


def test_one_extra_infers_x1():
    px = {'Devices': 5, 'Universal Consoles': 1}
    assert _infer_x_bonus(_profile(), px, {}) == 1


def test_starship_trait_count_alone_is_enough():
    """Traits are detected on MIXED screens even when EQ rows are absent."""
    assert _infer_x_bonus(_profile(), {}, {'Starship Traits': [0] * 7}) == 2


def test_max_wins_over_majority_when_a_slot_is_left_empty():
    """Two of six device slots empty → device evidence reads +1, not +2.

    Majority voting picked +1 here on the corpus; the console and trait
    evidence are right and must not be outvoted by an under-filled row.
    """
    px = {'Devices': 5, 'Universal Consoles': 2}
    layout = {'Starship Traits': [0] * 7}
    assert _infer_x_bonus(_profile(), px, layout) == 2


def test_a_partly_filled_row_never_lowers_the_bonus():
    """An empty slot makes evidence smaller, never larger — it must not veto."""
    px = {'Devices': 4, 'Universal Consoles': 2}   # devices reads +0
    assert _infer_x_bonus(_profile(), px, {}) == 2


def test_impossible_evidence_is_discarded_not_clamped():
    """The game grants at most +2; a larger reading means a miscount."""
    px = {'Devices': 9}
    assert _infer_x_bonus(_profile(), px, {}) == 0


def test_a_negative_reading_is_ignored():
    px = {'Devices': 2}          # fewer filled than the base profile
    assert _infer_x_bonus(_profile(), px, {}) == 0


def test_miracle_worker_base_console_is_not_counted_as_a_bonus():
    """MW ships already carry +1 Universal in the profile before the tier."""
    px = {'Devices': 4, 'Universal Consoles': 1}
    assert _infer_x_bonus(_profile(universal=1), px, {}) == 0


# ── composing the tier string ─────────────────────────────────────────────
from warp.warp_importer import _compose_inferred_tier  # noqa: E402


def test_base_tier_comes_from_cargo_not_from_a_guess():
    assert _compose_inferred_tier({'tier': '6'}, 2) == 'T6-X2'
    assert _compose_inferred_tier({'tier': '6'}, 1) == 'T6-X'
    assert _compose_inferred_tier({'tier': '5'}, 2) == 'T5-X2'


def test_no_evidence_claims_no_tier():
    """x=0 means 'no evidence of an upgrade', never 'not upgraded'."""
    assert _compose_inferred_tier({'tier': '6'}, 0) == ''


def test_hulls_below_t5_take_no_x_upgrade():
    assert _compose_inferred_tier({'tier': '4'}, 2) == ''


def test_unusable_cargo_tier_yields_nothing():
    assert _compose_inferred_tier({'tier': None}, 2) == ''
    assert _compose_inferred_tier({}, 2) == ''
    assert _compose_inferred_tier(None, 2) == ''


def test_result_is_always_a_canonical_tier_value():
    from warp.recognition.text_extractor import SHIP_TIER_VALUES
    for base in ('5', '6'):
        for x in (1, 2):
            assert _compose_inferred_tier({'tier': base}, x) in SHIP_TIER_VALUES
