"""Enter on an already-confirmed row steps to the next slot.

Enter means "accept this row and move on". On a row the user has already
confirmed there is nothing left to accept, and `_advance_to_next_unconfirmed`
only looks for *unconfirmed* rows — so on a fully reviewed screenshot the
selection did not move at all. `_on_enter` routes that case to the same
one-row step the Down arrow performs, from the canvas, the review list or
the name field alike. A row whose editors no longer match what is stored is
a real edit and still goes through accept.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip('PySide6')

from warp.trainer.trainer_window import WarpCoreWindow  # noqa: E402

_enter  = WarpCoreWindow._on_enter
_editor = WarpCoreWindow._current_editor_value


class _FakeCombo:
    def __init__(self, text=''):
        self._text = text

    def currentText(self):
        return self._text

    # _name_edit / ship-type line edit share this shape
    def text(self):
        return self._text


def _win(items, current=0, slot='Fore Weapon', typed=None):
    """Enough of `WarpCoreWindow` for `_on_enter` + `_current_editor_value`.

    `typed` overrides the name field; by default it mirrors the current
    row's stored name, i.e. the user has not touched anything.
    """
    if typed is None:
        typed = items[current].get('name', '') if 0 <= current < len(items) else ''
    ns = SimpleNamespace(
        _recognition_items=items,
        _review_list=SimpleNamespace(currentRow=lambda: current),
        _slot_combo=_FakeCombo(slot),
        _tier_combo=_FakeCombo(),
        _ship_type_combo=_FakeCombo(),
        _name_edit=_FakeCombo(typed),
        accepted=0,
        stepped=[],
    )
    ns._current_editor_value = lambda: _editor(ns)
    ns._slot_for_combo = lambda s: WarpCoreWindow._slot_for_combo(ns, s)
    ns._sets = None
    ns._on_accept = lambda: setattr(ns, 'accepted', ns.accepted + 1)
    ns._nav_review_row = ns.stepped.append
    return ns


def _row(**kw):
    base = {'slot': 'Fore Weapon', 'name': 'Phaser Beam Array',
            'state': 'confirmed'}
    base.update(kw)
    return base


# ── Enter on a confirmed row navigates ────────────────────────────────

def test_enter_on_a_confirmed_row_steps_down():
    win = _win([_row(), _row(name='Phaser Cannon')])
    _enter(win)
    assert win.stepped == [1] and win.accepted == 0


def test_the_step_is_the_only_thing_that_happens():
    # No re-save of an unchanged annotation, no contribute, no auto-sync:
    # _on_accept must not be reached at all.
    win = _win([_row()])
    _enter(win)
    assert win.accepted == 0


# ── …but a real edit still accepts ────────────────────────────────────

def test_a_retyped_name_still_accepts():
    win = _win([_row()], typed='Phaser Dual Cannons')
    _enter(win)
    assert win.accepted == 1 and win.stepped == []


def test_a_changed_slot_still_accepts():
    win = _win([_row()], slot='Aft Weapon')
    _enter(win)
    assert win.accepted == 1 and win.stepped == []


def test_a_pending_row_still_accepts():
    win = _win([_row(state='pending')])
    _enter(win)
    assert win.accepted == 1 and win.stepped == []


def test_a_community_conflict_row_still_accepts():
    win = _win([_row(state='community_conflict')])
    _enter(win)
    assert win.accepted == 1 and win.stepped == []


def test_an_auto_confirmed_row_still_accepts():
    # Yellow rows are the program's decision awaiting human review — Enter
    # is how the user signs off on them.
    win = _win([_row(auto_confirmed=True)])
    _enter(win)
    assert win.accepted == 1 and win.stepped == []


def test_no_selection_falls_through_to_accept():
    win = _win([_row()], current=-1)
    _enter(win)
    assert win.accepted == 1 and win.stepped == []


# ── BOFF seat keys ────────────────────────────────────────────────────

def test_a_seat_keyed_boff_row_compares_against_the_combo_label():
    # The combo can never show 'Boff Seat L[T]_478'; _slot_for_combo maps it
    # to the profession label, which is what the comparison must use.
    win = _win([_row(slot='Boff Seat L[T]_478', name='Attack Pattern Beta I')],
               slot='Boff Tactical')
    _enter(win)
    assert win.stepped == [1] and win.accepted == 0


# ── Editor readout ────────────────────────────────────────────────────

def test_ship_tier_is_read_from_the_tier_combo():
    win = _win([_row()], slot='Ship Tier')
    win._tier_combo = _FakeCombo('T6')
    assert _editor(win) == ('Ship Tier', 'T6')


def test_ship_type_is_read_from_the_type_combo():
    win = _win([_row()], slot='Ship Type')
    win._ship_type_combo = _FakeCombo('  Escort  ')
    assert _editor(win) == ('Ship Type', 'Escort')
