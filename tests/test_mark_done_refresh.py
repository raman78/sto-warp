"""Mark Done is derived from the review list, so it must be recomputed.

The button is enabled only when every review row is user-confirmed. That is a
pure function of `_recognition_items`, but Qt does not recompute it on its own,
so any path that adds, removes or re-confirms a row has to say so.

`_on_remove_item` did not. Deleting the last unconfirmed row left the button
greyed until the screenshot was reopened — `_populate_review_panel` refreshes,
which is why the stale state looked like it fixed itself on navigating away and
back.

Run standalone:
    python -m pytest tests/test_mark_done_refresh.py -v
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip('PySide6')

from PySide6.QtWidgets import (  # noqa: E402
    QApplication, QLabel, QPushButton,
)


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


class _Stub:
    """Only what `_refresh_mark_done_btn` reads."""

    def __init__(self, items, done=False):
        from warp.trainer.trainer_window import WarpCoreWindow

        self._MARK_DONE_TAIL_SEP = WarpCoreWindow._MARK_DONE_TAIL_SEP
        self._recognition_items = items
        self._screenshots = [Path('shot.png')]
        self._current_idx = 0
        self._screenshots_done = {'shot.png'} if done else set()
        self._btn_done = QPushButton()
        self._review_summary = QLabel('12 items')


def _refresh(stub):
    from warp.trainer.trainer_window import WarpCoreWindow

    WarpCoreWindow._refresh_mark_done_btn(stub)


def _row(state='confirmed', auto=False):
    return {'state': state, 'auto_confirmed': auto}


def test_enabled_when_every_row_is_user_confirmed(app):
    stub = _Stub([_row(), _row(), _row()])

    _refresh(stub)

    assert stub._btn_done.isEnabled()


def test_disabled_while_a_row_is_pending(app):
    stub = _Stub([_row(), _row(state='pending')])

    _refresh(stub)

    assert not stub._btn_done.isEnabled()


def test_auto_confirmed_does_not_count_as_confirmed(app):
    """Auto-confirmed is the detector's opinion, still awaiting review."""
    stub = _Stub([_row(), _row(auto=True)])

    _refresh(stub)

    assert not stub._btn_done.isEnabled()


def test_removing_the_last_pending_row_enables_it(app):
    """The reported bug, at the level the button actually reads.

    Before the fix nothing recomputed this after a delete, so the button kept
    the answer from before the row was removed.
    """
    items = [_row(), _row(state='pending')]
    stub = _Stub(items)
    _refresh(stub)
    assert not stub._btn_done.isEnabled()

    items.pop()                 # what `_on_remove_item` does
    _refresh(stub)

    assert stub._btn_done.isEnabled()


def test_an_already_done_screenshot_stays_toggleable(app):
    """Done rows keep the button live so Back to Edit is reachable."""
    stub = _Stub([_row(state='pending')], done=True)

    _refresh(stub)

    assert stub._btn_done.isEnabled()


def test_the_remove_path_calls_the_refresh():
    """Guards the wiring, not just the logic: the delete path must ask for a
    recompute. Checked on the source because driving `_on_remove_item` needs
    the whole window — data manager, canvas, review tree and file list."""
    import inspect

    from warp.trainer.trainer_window import WarpCoreWindow

    body = inspect.getsource(WarpCoreWindow._on_remove_item)

    assert '_refresh_mark_done_btn()' in body
