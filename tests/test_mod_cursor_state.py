"""The canvas cursor follows the keyboard, not the key events it was sent.

Alt, Ctrl and Shift each change the canvas cursor, and the widget used to
track them by their press/release pairs. That only works while every event is
delivered. A release taken by the window manager (Alt+Tab, Alt+drag to move a
window, a desktop-wide shortcut) never arrives, and Qt's cached modifier state
is no help either: a KeyRelease carries the state from *before* the key was
let go, so `QApplication.keyboardModifiers()` reports Alt from the moment Alt
is released until some other event replaces it (measured on a throwaway X
display, `dev/probe_live_modifiers.py`).

Either way the override cursor latched on and nothing could clear it — the
canvas behaved as though Alt were held down for ever.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtWidgets import QApplication

import warp.trainer.annotation_widget as aw

ALT = Qt.KeyboardModifier.AltModifier
NONE = Qt.KeyboardModifier.NoModifier


@pytest.fixture
def widget():
    QApplication.instance() or QApplication([])

    class _StubDataMgr:
        def get_annotations(self, path):
            return []

    w = aw.AnnotationWidget(_StubDataMgr())
    w.resize(400, 300)
    yield w
    # An override cursor left standing would follow the rest of the suite.
    w._clear_mod_cursor()
    w.close()


def _holding(widget, modifiers):
    """Pretend the keyboard is in this state, whatever events were delivered."""
    widget._live_modifiers = lambda: modifiers


def _arm(widget):
    """Put the modifier-override cursor up, the way holding Alt does."""
    from PySide6.QtGui import QCursor

    widget._set_mod_cursor(QCursor(Qt.CursorShape.CrossCursor))
    assert widget._mod_cursor_active


def _mouse_move(widget):
    from PySide6.QtGui import QMouseEvent

    return QMouseEvent(
        QEvent.Type.MouseMove,
        QPointF(5, 5),
        QPointF(5, 5),
        Qt.MouseButton.NoButton,
        Qt.MouseButton.NoButton,
        NONE,
    )


def test_the_cursor_is_set_from_the_keys_that_are_held(widget):
    _holding(widget, ALT)
    widget._refresh_mod_cursor()
    assert widget._mod_cursor_active, "Alt held — the draw cursor is shown"

    _holding(widget, NONE)
    widget._refresh_mod_cursor()
    assert not widget._mod_cursor_active, "Alt let go — the cursor goes back"


def test_a_release_that_never_arrived_is_cleared_by_the_next_mouse_move(widget):
    """The stuck-Alt case itself: no release event is ever delivered, so
    nothing announces the change — only the keyboard knows, and moving the
    mouse over the canvas is when we next ask it."""
    _arm(widget)                    # Alt was pressed, the crosshair is up
    _holding(widget, NONE)          # released while another window had focus
    widget.mouseMoveEvent(_mouse_move(widget))

    assert not widget._mod_cursor_active, "the crosshair does not outlive the key"


def test_a_drag_interrupted_by_a_window_switch_does_not_latch_the_cursor(widget):
    """A drag that loses the window never gets its mouse release, and while it
    counted as in progress the cursor could not be cleared at all."""
    _arm(widget)
    widget._drawing = True
    widget._draw_start = QPointF(1, 1).toPoint()
    widget._draw_current = QPointF(9, 9).toPoint()

    widget.eventFilter(widget, QEvent(QEvent.Type.WindowDeactivate))

    assert not widget._drawing, "the unfinished box is dropped, not left open"
    assert not widget._mod_cursor_active


def test_entering_the_canvas_asks_the_keyboard_not_the_last_event(widget, monkeypatch):
    """Re-entering the canvas straight after letting Alt go is exactly when
    the cached state still says Alt: the KeyRelease that delivered the news
    carried the state from before it."""
    monkeypatch.setattr(QApplication, "keyboardModifiers", staticmethod(lambda: ALT))
    monkeypatch.setattr(QApplication, "queryKeyboardModifiers", staticmethod(lambda: NONE))

    widget.enterEvent(QEvent(QEvent.Type.Enter))

    assert not widget._mod_cursor_active, "nothing is held, so no draw cursor"


def test_the_widget_cursor_goes_back_too_not_just_the_override(widget):
    """Two cursors, one modifier. Moving the mouse while Alt is held sets a
    widget-level cursor as well (`mouseMoveEvent`), and the override cursor
    coming off merely reveals it — so the crosshair survived the release and
    sat there until the next mouse move happened to call `unsetCursor`.

    Asserted on the cursor's shape, not on `WA_SetCursor`: Qt 6.11 leaves that
    attribute set after `unsetCursor()` and only puts the shape back, so the
    attribute answers a different question from the one being asked here."""
    arrow = Qt.CursorShape.ArrowCursor

    _holding(widget, ALT)
    widget._refresh_mod_cursor()
    widget.mouseMoveEvent(_mouse_move(widget))      # ← the mouse move matters
    assert widget.cursor().shape() != arrow, \
        "holding Alt over the canvas gives the widget a cursor of its own"

    _holding(widget, NONE)
    widget._refresh_mod_cursor()

    assert not widget._mod_cursor_active
    assert widget.cursor().shape() == arrow, \
        "nothing is held, so the canvas keeps no cursor of its own either"


def test_a_lost_release_is_cleared_with_the_pointer_over_a_box_too(widget):
    """The mouse move that rescues a lost release has to rescue it wherever
    the pointer happens to be. The tail of `mouseMoveEvent` only reaches
    `unsetCursor` when the pointer is over nothing at all, so a box under the
    cursor used to keep the crosshair alive."""
    widget.set_review_items([
        {"bbox": (0, 0, 50, 50), "state": "pending", "name": "A", "slot": "Fore Weapons"},
    ])
    _holding(widget, ALT)
    widget._refresh_mod_cursor()
    widget.mouseMoveEvent(_mouse_move(widget))
    assert widget.cursor().shape() != Qt.CursorShape.ArrowCursor

    _holding(widget, NONE)                          # release taken by the WM
    widget.mouseMoveEvent(_mouse_move(widget))      # (5, 5) — inside the box

    assert not widget._mod_cursor_active
    assert widget.cursor().shape() == Qt.CursorShape.ArrowCursor
