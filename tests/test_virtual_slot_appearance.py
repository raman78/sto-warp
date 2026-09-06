"""Empty and inactive look different, and say the same thing in both views.

They are two states — a slot the player has not filled against one the ship or
the character has not unlocked — and the review list drew both in one grey, so
a glance could not tell them apart. The canvas tooltip did not colour them at
all and showed the internal marker.

The two hues live in `warp.gui.VIRTUAL_COLOURS` and both views read them from
there, so the list and the tooltip cannot drift apart on what an empty slot
looks like.

Offline: colour lookup and HTML composition, no Qt window and no icons.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip('PySide6')

from warp.gui import (VIRTUAL_COLOURS, VIRTUAL_LABELS, slot_tooltip_html,
                      virtual_colour)
from warp.trainer.trainer_window import WarpCoreWindow

_visuals = WarpCoreWindow._review_row_visuals
_STUB = SimpleNamespace(_AUTO_COLOR='#c0ffee', _CONFIRMED_COLOR='#000000',
                        _CONFLICT_COLOR='#ff0000')


def _row(name, **kw):
    kw.setdefault('confirmed', True)
    kw.setdefault('cross_check_failed', False)
    kw.setdefault('auto_confirmed', False)
    kw.setdefault('conflict_disk_name', '')
    return _visuals(_STUB, name, 1.0, **kw)


# ── The two states are told apart ─────────────────────────────────────────

def test_empty_and_inactive_have_different_colours():
    assert virtual_colour('__empty__') != virtual_colour('__inactive__')


def test_they_differ_when_pending_too():
    assert (virtual_colour('__empty__', confirmed=False)
            != virtual_colour('__inactive__', confirmed=False))


def test_confirmed_and_pending_differ_within_a_state():
    """A reviewed row still has to read differently from an unreviewed one."""
    for name in VIRTUAL_COLOURS:
        assert (virtual_colour(name, confirmed=True)
                != virtual_colour(name, confirmed=False))


def test_a_real_item_has_no_virtual_colour():
    assert virtual_colour('Phaser Beam Array') is None


def test_none_of_them_collides_with_a_state_colour():
    """Blue, mint, orange, gold and red already mean something else."""
    taken = {'#5cbfff', '#7effc8', '#ff9a3c', '#ffcc00', '#ff5555', '#ffaaaa'}
    for entry in VIRTUAL_COLOURS.values():
        assert not (set(entry.values()) & taken)


# ── The review list uses them ─────────────────────────────────────────────

def test_the_list_colours_empty_and_inactive_apart():
    _t, _c, _s, empty_col = _row('__empty__')
    _t, _c, _s, inactive_col = _row('__inactive__')
    assert empty_col != inactive_col


def test_the_list_still_names_the_state_in_words():
    text, _c, _s, _col = _row('__empty__')
    assert text == '[empty slot]'
    text, _c, _s, _col = _row('__inactive__')
    assert text == '[inactive slot]'


def test_a_pending_virtual_row_takes_the_pending_shade():
    _t, _c, _s, col = _row('__inactive__', confirmed=False)
    assert col == virtual_colour('__inactive__', confirmed=False)


# ── The tooltip agrees with the list ──────────────────────────────────────

def test_the_tooltip_colours_the_name():
    html = slot_tooltip_html('Devices', '__empty__', 0.99, confirmed=True)
    assert virtual_colour('__empty__') in html


def test_the_tooltip_shows_words_not_the_internal_marker():
    html = slot_tooltip_html('Devices', '__inactive__', 0.99, confirmed=True)
    assert VIRTUAL_LABELS['__inactive__'] in html
    assert '__inactive__' not in html


def test_the_two_states_get_different_tooltip_colours():
    a = slot_tooltip_html('Devices', '__empty__', 0.9, confirmed=True)
    b = slot_tooltip_html('Devices', '__inactive__', 0.9, confirmed=True)
    assert virtual_colour('__empty__') in a
    assert virtual_colour('__empty__') not in b


# ── Typography ────────────────────────────────────────────────────────────

def test_the_item_name_is_bold():
    html = slot_tooltip_html('Fore Weapons', 'Phaser Beam Array', 0.9)
    assert '<b>Phaser Beam Array</b>' in html


def test_the_slot_is_not_bold():
    """It is context — which row this is — and the reader already knows it
    from where they are hovering."""
    html = slot_tooltip_html('Fore Weapons', 'Phaser Beam Array', 0.9)
    assert '<b>Fore Weapons</b>' not in html
    assert 'Fore Weapons' in html


def test_both_hold_on_a_confirmed_card():
    html = slot_tooltip_html('Fore Weapons', 'Phaser Beam Array', 0.9,
                             confirmed=True)
    assert '<b>Phaser Beam Array</b>' in html
    assert '<b>Fore Weapons</b>' not in html


def test_an_unmatched_row_is_not_bolded_into_a_name():
    html = slot_tooltip_html('Fore Weapons', '', 0.4)
    assert '— unmatched —' in html
    assert '<b>— unmatched —</b>' not in html


# ── Width ─────────────────────────────────────────────────────────────────

def test_a_long_name_does_not_wrap():
    """`Console - Advanced Engineering - Isomagnetic Plasma Distribution
    Manifold` is 71 characters and folded over three lines."""
    long_name = ('Console - Advanced Engineering - Isomagnetic Plasma '
                 'Distribution Manifold')
    html = slot_tooltip_html('Engineering Consoles', long_name, 0.97,
                             confirmed=True)
    assert 'nowrap' in html
    assert long_name in html
