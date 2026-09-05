"""One mistyped character must not delete a row of ground slots.

Ground equipment labels were matched by exact string equality. Measured on
`Screenshot_2025-03-19_122129.png`, the reader returned `'Kil Modules'` for a
perfectly legible `Kit Modules` — one character in eleven, similarity 0.909,
and at *higher* confidence than the correct reading on the same image. The
keyword lookup missed, the label was discarded without a trace, the detector
dropped from `OCR_FULL` to `OCR_PARTIAL`, and all six Kit Module cells were
lost.

The space equipment detector has fuzzed its labels all along
(`eq_geometry._fuzzy_best`, cutoff 0.65). The ground one simply never did.

Scored against the 299 human-confirmed ground boxes in one maintainer's store,
this took the grid from 295 hits to 298.

Offline: no OCR, no network — these call the matcher directly.
"""
from __future__ import annotations

import pytest

pytest.importorskip("cv2")

from warp.recognition import ground_eq_geometry as g


def _tok(text: str, x0: int = 400, w: int = 40) -> dict:
    return {'low': text.lower(), 'text': text, 'x0': x0, 'w': w}


# ── The case that was losing six cells ─────────────────────────────────────

def test_the_measured_misread_now_matches():
    assert g._match_label(_tok('Kil Modules')) == g.SLOT_KIT_MODULES


def test_the_correct_spelling_still_matches():
    assert g._match_label(_tok('Kit Modules')) == g.SLOT_KIT_MODULES


@pytest.mark.parametrize('text,slot', [
    ('Devlces',  'Ground Devices'),
    ('Weapors',  'Weapons'),
    ('Shlelds',  'Personal Shield'),
    ('Kil',      'Kit'),
])
def test_other_single_character_slips_match(text, slot):
    assert g._match_label(_tok(text)) == slot


# ── What must still be refused ────────────────────────────────────────────

@pytest.mark.parametrize('text', [
    'Crit Severity:', 'Random Word', 'Fleet Admiral 65', 'Gunner Anderson',
])
def test_unrelated_text_matches_nothing(text):
    assert g._match_label(_tok(text)) is None


def test_a_short_word_cannot_reach_a_long_keyword():
    """The length guard is what keeps tolerance from becoming invention:
    only keywords within two characters of the token are considered."""
    assert g._fuzzy_slot('devlces') != g.SLOT_KIT_MODULES


def test_the_stat_bar_shields_is_still_rejected():
    """`Shields:` in the HUD stat column is not the Personal Shield slot."""
    assert g._match_label({'low': 'shields:', 'x0': 50, 'w': 40}) is None


def test_a_misread_stat_bar_shields_is_rejected_too():
    """The column filter now runs on the resolved slot, so a fuzzy hit is
    filtered exactly as an exact hit is — it used to test the raw string and
    would have let `Shlelds` through from the stat bar."""
    assert g._match_label(_tok('Shlelds', x0=50)) is None


def test_a_real_shields_label_in_the_equipment_column_survives():
    assert g._match_label(_tok('Shields', x0=489)) == g.SLOT_PERSONAL_SHIELD


def test_a_merged_body_ev_token_is_still_special_cased():
    assert g._match_label(_tok('Body EV Suit')) == g.SLOT_BODY_ARMOR


# ── The two matchers cannot drift apart ───────────────────────────────────

def test_candidate_collection_uses_the_same_matcher():
    """`_collect_candidates` had the exact-string test written out a second
    time; fuzzing only one of them would have left the other strict."""
    out = g._collect_candidates([_tok('Kil Modules', x0=317)])
    assert list(out) == [g.SLOT_KIT_MODULES]


def test_candidate_collection_still_splits_a_merged_body_ev_token():
    out = g._collect_candidates([_tok('Body EV Suit', x0=489, w=100)])
    assert set(out) == {g.SLOT_BODY_ARMOR, g.SLOT_EV_SUIT}
    assert out[g.SLOT_EV_SUIT][0]['_split'] is True
    assert out[g.SLOT_EV_SUIT][0]['x0'] > out[g.SLOT_BODY_ARMOR][0]['x0']


def test_candidate_collection_rejects_the_stat_bar():
    assert g._collect_candidates([{'low': 'shields:', 'x0': 50, 'w': 40}]) == {}


def test_the_cutoff_matches_the_space_detector():
    """Same problem, same number — if one moves, the other should be
    considered too."""
    from warp.recognition import eq_geometry
    assert g.LABEL_FUZZY_CUTOFF == eq_geometry._FUZZY_CUTOFF
