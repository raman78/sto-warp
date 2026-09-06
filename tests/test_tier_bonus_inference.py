"""Recovering the T6-X / T6-X2 bonus when the tier badge is off-screen.

Many screenshots do not show `[T6-X2]`, so `ship_tier` is empty and no bonus
is applied: Universal Consoles drops to 0 (the layout detector then skips the
whole row), Devices and Starship Traits lose up to 2 each.

`_infer_x_bonus` recovers it from what the run already measured. Every piece
of evidence is a LOWER bound — a pixel count only sees filled slots — so the
rule is max, not majority.
"""
from __future__ import annotations

from warp.recognition.layout_detector import merge_trait_boxes
from warp.warp_importer import _infer_x_bonus


def _bonus(profile, px, traits):
    """The bonus alone — the function also returns its evidence for the log."""
    return _infer_x_bonus(profile, px, traits)[0]


def _profile(devices=4, universal=0):
    return {'Devices': devices, 'Universal Consoles': universal,
            'Starship Traits': 5}


def test_no_evidence_leaves_the_profile_alone():
    assert _bonus(_profile(), {}, {}) == 0


def test_measurements_matching_the_profile_mean_no_bonus():
    px = {'Devices': 4, 'Universal Consoles': 0}
    assert _bonus(_profile(), px, {'Starship Traits': 5}) == 0


def test_two_extra_devices_and_consoles_infer_x2():
    px = {'Devices': 6, 'Universal Consoles': 2}
    assert _bonus(_profile(), px, {}) == 2


def test_one_extra_infers_x1():
    px = {'Devices': 5, 'Universal Consoles': 1}
    assert _bonus(_profile(), px, {}) == 1


def test_starship_trait_count_alone_is_enough():
    """Traits are detected on MIXED screens even when EQ rows are absent."""
    assert _bonus(_profile(), {}, {'Starship Traits': 7}) == 2


def test_max_wins_over_majority_when_a_slot_is_left_empty():
    """Two of six device slots empty → device evidence reads +1, not +2.

    Majority voting picked +1 here on the corpus; the console and trait
    evidence are right and must not be outvoted by an under-filled row.
    """
    px = {'Devices': 5, 'Universal Consoles': 2}
    assert _bonus(_profile(), px, {'Starship Traits': 7}) == 2


def test_a_partly_filled_row_never_lowers_the_bonus():
    """An empty slot makes evidence smaller, never larger — it must not veto."""
    px = {'Devices': 4, 'Universal Consoles': 2}   # devices reads +0
    assert _bonus(_profile(), px, {}) == 2


def test_impossible_evidence_is_discarded_not_clamped():
    """The game grants at most +2; a larger reading means a miscount."""
    px = {'Devices': 9}
    assert _bonus(_profile(), px, {}) == 0


def test_a_negative_reading_is_ignored():
    px = {'Devices': 2}          # fewer filled than the base profile
    assert _bonus(_profile(), px, {}) == 0


def test_miracle_worker_base_console_is_not_counted_as_a_bonus():
    """MW ships already carry +1 Universal in the profile before the tier."""
    px = {'Devices': 4, 'Universal Consoles': 1}
    assert _bonus(_profile(universal=1), px, {}) == 0


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


# ── Raising a tier the OCR did read ────────────────────────────────────
#
# `_infer_x_bonus` subtracts the profile, and the profile already carries
# whatever the OCR tier granted — so what it returns is the surplus beyond
# that tier. These lock the arithmetic that turns a surplus into a new tier.

def _raise(ocr_tier: str, surplus: int) -> tuple[int, int]:
    """`(new level, what to add to the profile)` for an OCR tier and surplus."""
    ocr_x = 2 if '-X2' in ocr_tier else 1 if '-X' in ocr_tier else 0
    raised_to = min(2, ocr_x + surplus)
    return raised_to, max(0, raised_to - ocr_x)


def test_a_surplus_over_a_read_tier_raises_it():
    """`[T6-X]` read correctly but the screen holds one more of each row:
    the ship is X2 and three rows would otherwise come out short."""
    assert _raise('T6-X', 1) == (2, 1)


def test_a_surplus_over_no_tier_behaves_as_before():
    assert _raise('', 2) == (2, 2)
    assert _raise('', 1) == (1, 1)


def test_no_surplus_changes_nothing():
    assert _raise('T6-X', 0) == (1, 0)
    assert _raise('T6-X2', 0) == (2, 0)


def test_a_tier_is_never_raised_past_the_game_maximum():
    """The game grants at most +2; a larger reading is a miscount."""
    assert _raise('T6-X2', 2) == (2, 0)
    assert _raise('T6-X', 2) == (2, 1)


def test_measuring_less_than_the_badge_claims_is_not_evidence():
    """Pixel counts are lower bounds — an unfilled slot looks exactly like a
    slot the ship does not have, so a shortfall can never lower a tier."""
    profile = {'Devices': 5, 'Universal Consoles': 2, 'Starship Traits': 7}
    px = {'Devices': 4, 'Universal Consoles': 1}

    assert _bonus(profile, px, {'Starship Traits': 6}) == 0


def test_a_screen_agreeing_with_the_read_tier_reports_no_surplus():
    """The trap this nearly walked into: with `-X` already applied, six
    starship traits is exactly right. Measured against the constant base of
    five it reads as one more upgrade and promotes the ship to `-X2`."""
    profile = {'Devices': 4, 'Universal Consoles': 1, 'Starship Traits': 6}
    px = {'Devices': 4, 'Universal Consoles': 1}

    assert _bonus(profile, px, {'Starship Traits': 6}) == 0


def test_a_genuine_surplus_over_a_read_tier_is_still_seen():
    """Same `-X` profile, but the screen holds an X2 ship's rows."""
    profile = {'Devices': 4, 'Universal Consoles': 1, 'Starship Traits': 6}
    px = {'Devices': 5, 'Universal Consoles': 2}

    assert _bonus(profile, px, {'Starship Traits': 7}) == 1


# ── The trait evidence has to be a measurement ────────────────────────────
#
# It was not. The third argument used to be the finished layout, and the
# trait projector draws `Starship Traits` at the game maximum so that no slot
# goes undrawn — always 7, whatever the ship. The evidence therefore read
# `7 - profile`, at least 1 for every ship below `-X2`, and the max rule
# promoted it. The tests above did not catch it because they chose that count
# freely; production never does.

def test_the_measured_count_comes_from_the_grid_not_from_the_drawn_row():
    """`merge_trait_boxes` reports what `trait_grid` found. The row it leaves
    in `result` is profile-sized and says nothing about the ship."""
    counts: dict[str, int] = {}
    result = {'Starship Traits': [(0, 0, 9, 9)] * 7}          # drawn at max
    merge_trait_boxes(result, {'Starship Traits': [(0, 0, 9, 9)] * 5},
                      lambda b: False, counts)
    assert counts['Starship Traits'] == 5
    assert len(result['Starship Traits']) == 7


def test_boxes_dropped_into_the_boff_panel_are_not_counted():
    counts: dict[str, int] = {}
    merge_trait_boxes({}, {'Starship Traits': [(0, 0, 9, 9), (50, 0, 9, 9)]},
                      lambda b: b[0] == 50, counts)
    assert counts['Starship Traits'] == 1


def test_a_t6x_ship_whose_grid_finds_fewer_traits_keeps_its_tier():
    """`image-817e2e37c01aed8c.png`, a T6-X Terran Adamant: devices 3-3 = 0,
    universal consoles 1-1 = 0, and the grid found 5 trait icons against a
    profile of 6. Reading the drawn row instead gave 7-6 = 1 and raised it."""
    profile = {'Devices': 3, 'Universal Consoles': 1, 'Starship Traits': 6}
    px = {'Devices': 3, 'Universal Consoles': 1}
    assert _bonus(profile, px, {'Starship Traits': 5}) == 0


# ── The log reports the evidence, not the outcome ─────────────────────────

def test_the_evidence_names_every_measurement_it_used():
    _x, seen = _infer_x_bonus({'Devices': 4, 'Universal Consoles': 0,
                               'Starship Traits': 5},
                              {'Devices': 6, 'Universal Consoles': 2},
                              {'Starship Traits': 7})
    assert 'Devices 6-4' in seen
    assert 'Universal Consoles 2-0' in seen
    assert 'Starship Traits 7-5' in seen


def test_no_measurement_says_so_rather_than_printing_zeros():
    _x, seen = _infer_x_bonus(_profile(), {}, {})
    assert seen == 'nothing measured'


# ── "Fewer" is measured against the profile, not the drawn row ────────────

def test_the_grid_wins_when_it_matches_the_profile():
    """`image-8ee54291302414af.png`, a T6 with five starship traits: the grid
    had the row exactly on the icons and it was refused as `5 < 7`, because
    the row it was compared against is padded to the game maximum. Against
    the profile, five is not fewer than five."""
    grid = [(831, 221, 42, 56)] * 5
    result = {'Starship Traits': [(0, 0, 41, 52)] * 7}       # drawn at max
    merge_trait_boxes(result, {'Starship Traits': grid}, lambda b: False,
                      None, {'Starship Traits': 5})
    assert result['Starship Traits'] == grid


def test_a_real_shrink_is_still_refused():
    """The case the rule exists for: the grid split a section and kept two of
    seven. `image-4391ccd9d2683d4e.png`."""
    rows = [(0, 0, 41, 52)] * 7
    result = {'Starship Traits': list(rows)}
    merge_trait_boxes(result, {'Starship Traits': [(9, 9, 42, 56)] * 2},
                      lambda b: False, None, {'Starship Traits': 7})
    assert result['Starship Traits'] == rows


def test_without_a_profile_the_drawn_row_is_the_yardstick():
    rows = [(0, 0, 41, 52)] * 7
    result = {'Starship Traits': list(rows)}
    merge_trait_boxes(result, {'Starship Traits': [(9, 9, 42, 56)] * 5},
                      lambda b: False)
    assert result['Starship Traits'] == rows
