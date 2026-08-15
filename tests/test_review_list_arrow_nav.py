"""Up / Down step the review list from anywhere, one slot at a time.

Focus lands on the canvas after almost every click there, so the arrows are
routed by the window's application-level event filter instead of the focused
widget. Group headers are stepped over — the rows worth visiting are the
slots.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip('PySide6')

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QKeyEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QLineEdit  # noqa: E402

from warp.trainer.trainer_window import WarpCoreWindow  # noqa: E402

_steps = WarpCoreWindow._arrow_steps_review_list
_nav   = WarpCoreWindow._nav_review_row


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


class _FakeItem:
    def __init__(self, parent=None):
        self._parent = parent
        self.children: list[_FakeItem] = []
        self.expanded = True

    def parent(self):
        return self._parent

    def childCount(self):
        return len(self.children)

    def child(self, i):
        return self.children[i]

    def setExpanded(self, value):
        self.expanded = value


class _FakeList:
    """The slice of `_ReviewListAdapter` the navigation touches."""

    def __init__(self, rows: int, groups: int = 1, current: int = 0):
        self._parents = [_FakeItem() for _ in range(groups)]
        self._flat = [_FakeItem(self._parents[i % groups]) for i in range(rows)]
        for i, it in enumerate(self._flat):
            self._parents[i % groups].children.append(it)
        self._current = current

    # adapter API
    def count(self):
        return len(self._flat)

    def item(self, row):
        return self._flat[row] if 0 <= row < len(self._flat) else None

    def currentRow(self):
        return self._current

    def setCurrentRow(self, row):
        self._current = row

    def currentItem(self):
        return self._flat[self._current] if self._current >= 0 else self._header

    def row(self, item):
        return self._flat.index(item) if item in self._flat else -1

    def isVisible(self):
        return True

    # test helpers
    _header = None

    def select_header(self, group: int = 0):
        self._current = -1
        self._header = self._parents[group]


def _win(rl):
    """Enough of `WarpCoreWindow` for the two navigation methods."""
    ns = SimpleNamespace(
        _review_list=rl,
        _ARROW_OWNERS=WarpCoreWindow._ARROW_OWNERS,
        isActiveWindow=lambda: True,
    )
    ns._set_review_row = lambda row: WarpCoreWindow._set_review_row(ns, row)
    return ns


def _key(key, mods=Qt.KeyboardModifier.NoModifier):
    return QKeyEvent(QKeyEvent.Type.KeyPress, key, mods)


# ── Which key presses get routed ──────────────────────────────────────

def test_a_plain_down_arrow_steps_the_list(app):
    assert _steps(_win(_FakeList(4)), _key(Qt.Key.Key_Down)) is True


def test_alt_down_is_left_to_screenshot_navigation(app):
    ev = _key(Qt.Key.Key_Down, Qt.KeyboardModifier.AltModifier)
    assert _steps(_win(_FakeList(4)), ev) is False


def test_the_numpad_bit_does_not_block_the_arrows(app):
    ev = _key(Qt.Key.Key_Up, Qt.KeyboardModifier.KeypadModifier)
    assert _steps(_win(_FakeList(4)), ev) is True


def test_other_keys_are_not_touched(app):
    assert _steps(_win(_FakeList(4)), _key(Qt.Key.Key_Left)) is False


def test_an_empty_list_takes_no_arrows(app):
    assert _steps(_win(_FakeList(0)), _key(Qt.Key.Key_Down)) is False


def test_an_inactive_window_takes_no_arrows(app):
    win = _win(_FakeList(4))
    win.isActiveWindow = lambda: False
    assert _steps(win, _key(Qt.Key.Key_Down)) is False


def test_a_text_field_keeps_its_own_arrows():
    # Membership in _ARROW_OWNERS is what the predicate tests; asserting it
    # here avoids depending on offscreen focus delivery.
    assert isinstance(QLineEdit(), WarpCoreWindow._ARROW_OWNERS)


# ── Where the selection lands ─────────────────────────────────────────

def test_down_moves_to_the_next_item_row():
    rl = _FakeList(5, current=2)
    _nav(_win(rl), 1)
    assert rl.currentRow() == 3


def test_up_moves_to_the_previous_item_row():
    rl = _FakeList(5, current=2)
    _nav(_win(rl), -1)
    assert rl.currentRow() == 1


def test_the_last_row_does_not_wrap():
    rl = _FakeList(3, current=2)
    _nav(_win(rl), 1)
    assert rl.currentRow() == 2


def test_the_first_row_does_not_wrap():
    rl = _FakeList(3, current=0)
    _nav(_win(rl), -1)
    assert rl.currentRow() == 0


def test_down_with_nothing_selected_starts_at_the_top():
    rl = _FakeList(3, current=-1)
    _nav(_win(rl), 1)
    assert rl.currentRow() == 0


def test_up_with_nothing_selected_starts_at_the_bottom():
    rl = _FakeList(3, current=-1)
    _nav(_win(rl), -1)
    assert rl.currentRow() == 2


def test_down_on_a_group_header_enters_the_group():
    # Two groups, rows alternating between them: group 0 holds rows 0 and 2.
    rl = _FakeList(4, groups=2)
    rl.select_header(1)          # first child is row 1
    _nav(_win(rl), 1)
    assert rl.currentRow() == 1


def test_up_on_a_group_header_leaves_it_upwards():
    rl = _FakeList(4, groups=2)
    rl.select_header(1)
    _nav(_win(rl), -1)
    assert rl.currentRow() == 0


def test_landing_on_a_row_expands_its_collapsed_group():
    rl = _FakeList(4, current=0)
    rl.item(1).parent().expanded = False
    _nav(_win(rl), 1)
    assert rl.item(1).parent().expanded is True
