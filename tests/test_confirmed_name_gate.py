"""Nothing may be confirmed under a name the slot's candidate list lacks.

Typing has been gated since 2026-05-18: `_on_accept` refuses a name that is
not an exact match. Auto-accept was not, and the recogniser is the one thing
that can produce a name nobody ever typed.

That is not hypothetical. Wiki art is filed under names that qualify the
picture — `Fire on my Mark (Ground)`, `Liberated Borg Kingdom Nanoprobes
(space)` — which cargo does not carry, because cargo stores those as two rows
under one `name`. Art enrolment put them in the gallery, the recogniser
returned them, auto-accept wrote them as ground truth, and one then beat the
correct cargo name in a merge vote.

Run standalone:
    python -m pytest tests/test_confirmed_name_gate.py -v
"""
from __future__ import annotations

import pytest

pytest.importorskip('PySide6')

from warp.trainer.trainer_window import WarpCoreWindow


class _Window:
    """`_name_is_acceptable` reads one collaborator, so bind it to a stub.

    Constructing the real window starts model loading and sync threads, and
    the gate under test is a pure function of (slot, name, candidates).
    """

    def __init__(self, candidates):
        self._candidates = candidates

    def _build_search_candidates(self, slot: str = ''):
        return list(self._candidates.get(slot, []))

    _name_is_acceptable = WarpCoreWindow._name_is_acceptable


CANDIDATES = {
    'Personal Space Traits': ['Liberated Borg Kingdom Nanoprobes', 'Astrophysicist'],
    'Boff Tactical':         ['Fire on my Mark', 'Attack Pattern Beta'],
    'Engineering Consoles':  ['Console - Engineering - RCS Accelerator'],
}


@pytest.fixture
def win():
    return _Window(CANDIDATES)


# ── What must be refused ───────────────────────────────────────────────────

def test_an_art_qualified_name_is_refused():
    """The exact pair found in production."""
    w = _Window(CANDIDATES)

    assert not w._name_is_acceptable('Boff Tactical', 'Fire on my Mark (Ground)')
    assert not w._name_is_acceptable('Personal Space Traits',
                                     'Liberated Borg Kingdom Nanoprobes (space)')


def test_a_typo_is_refused(win):
    assert not win._name_is_acceptable('Boff Tactical', 'Attack Pattern Beta ')


def test_a_fragment_is_refused(win):
    """`mart` reached production in an Engineering Consoles slot."""
    assert not win._name_is_acceptable('Engineering Consoles', 'mart')


def test_a_name_valid_for_another_slot_is_refused(win):
    """The candidate list is per slot, so a real name in the wrong slot is
    still wrong — that is a mislabelled row, not a naming question."""
    assert not win._name_is_acceptable('Engineering Consoles', 'Fire on my Mark')


# ── What must still pass ───────────────────────────────────────────────────

def test_an_exact_candidate_is_accepted(win):
    assert win._name_is_acceptable('Boff Tactical', 'Fire on my Mark')


def test_an_empty_name_is_accepted(win):
    """That is how a slot is recorded as Unknown."""
    assert win._name_is_acceptable('Boff Tactical', '')


def test_a_virtual_class_is_accepted(win):
    """`__empty__` / `__inactive__` are labels the embedder needs and no
    cargo list carries."""
    from warp.trainer.trainer_window import VIRTUAL_ITEM_NAMES

    for v in VIRTUAL_ITEM_NAMES:
        assert win._name_is_acceptable('Boff Tactical', v)


def test_a_text_slot_bypasses_the_item_vocabulary(win):
    """Ship Type and Ship Tier are OCR reads edited through their own combos,
    not names picked from an item list."""
    assert win._name_is_acceptable('Ship Type', 'Fleet Yamaguchi Support Cruiser')
    assert win._name_is_acceptable('Ship Tier', 'T6-X')


def test_an_unreachable_cargo_falls_open():
    """An empty candidate list means cargo could not be consulted. Refusing
    everything then would block the trainer outright."""
    w = _Window({'Boff Tactical': []})

    assert w._name_is_acceptable('Boff Tactical', 'anything at all')
