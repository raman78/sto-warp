"""The status column says how far along a row is, not what the slot holds.

Reported from a real session: a confirmed empty or inactive slot showed the
status `Inactive`, so a reviewer could not tell a row they had checked from
one they had not. The two facts were being written to the same column while
the item text and the colour already carried the second one.

Run standalone:
    python -m pytest tests/test_review_row_status.py -v
"""
from __future__ import annotations

import pytest

pytest.importorskip('PySide6')

from warp.trainer.trainer_window import WarpCoreWindow


def _visuals(name: str, *, confirmed: bool, auto: bool = False, conf: float = 1.0):
    return WarpCoreWindow._review_row_visuals(
        WarpCoreWindow, name, conf,
        confirmed=confirmed, cross_check_failed=False, auto_confirmed=auto,
        conflict_disk_name='', inferred=False,
    )


def _status(*a, **kw) -> str:
    return _visuals(*a, **kw)[2]


def _text(*a, **kw) -> str:
    return _visuals(*a, **kw)[0]


# ── The status column ──────────────────────────────────────────────────────

@pytest.mark.parametrize('virtual', ['__empty__', '__inactive__'])
def test_a_confirmed_blank_slot_reads_as_confirmed(virtual):
    assert _status(virtual, confirmed=True) == 'Confirmed'


@pytest.mark.parametrize('virtual', ['__empty__', '__inactive__'])
def test_an_auto_accepted_blank_slot_reads_as_auto(virtual):
    """The same distinction the rest of the list draws — a row the detector
    took on a threshold is not one a person checked."""
    assert _status(virtual, confirmed=True, auto=True) == 'Auto'


@pytest.mark.parametrize('virtual', ['__empty__', '__inactive__'])
def test_a_blank_slot_awaiting_review_reads_as_pending(virtual):
    assert _status(virtual, confirmed=False) == 'Pending'


def test_a_real_item_is_unaffected():
    assert _status('Hazard Emitters', confirmed=True) == 'Confirmed'
    assert _status('Hazard Emitters', confirmed=True, auto=True) == 'Auto'


# ── What the slot holds still shows ────────────────────────────────────────

def test_the_row_still_says_which_kind_of_blank_it_is():
    """Moving the state into the status column must not lose the fact — it
    lives in the item text, which is where a reader looks for it."""
    assert _text('__empty__', confirmed=True) == '[empty slot]'
    assert _text('__inactive__', confirmed=True) == '[inactive slot]'
    assert _text('__inactive__', confirmed=False) == '[inactive slot]'


def test_blank_rows_keep_their_own_colour():
    """Colour is the third channel carrying "this is not an item", so the two
    kinds of row stay distinguishable at a glance."""
    confirmed_blank = _visuals('__empty__', confirmed=True)[3]
    confirmed_item = _visuals('Hazard Emitters', confirmed=True)[3]

    assert confirmed_blank != confirmed_item
