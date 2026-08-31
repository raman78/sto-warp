"""Screen-type detection must not yank the file list back to the top.

Opening a folder starts screen-type detection in the background. When it
finished, the trainer selected the first screenshot so the user would have
something loaded — guarded by `_current_idx < 0`, meaning "the user has not
picked anything yet".

That guard answers the wrong question. Scrolling a list selects nothing, so
someone reading further down while detection ran still looked untouched, and
`setCurrentRow(0)` — which scrolls as well as selects — pulled the view back
to the top under them, mid-scroll.

Run standalone:
    python -m pytest tests/test_file_list_no_jump.py -v
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip('PySide6')

from PySide6.QtWidgets import QApplication, QListWidget  # noqa: E402


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


class _Stub:
    """Only what `_select_first_unless_user_moved` touches."""

    def __init__(self, n_files: int):
        self._screenshots = [Path(f'shot_{i}.png') for i in range(n_files)]
        self._current_idx = -1
        self._file_list = QListWidget()
        for p in self._screenshots:
            self._file_list.addItem(p.name)
        # Shown (offscreen, via conftest's QT_QPA_PLATFORM) and small, so Qt
        # actually computes a scroll range — an unshown widget reports 0 and a
        # scroll test against it would pass without testing anything.
        self._file_list.resize(120, 60)
        self._file_list.show()
        self.loaded: list[int] = []

    def _load_screenshot(self, idx):
        self.loaded.append(idx)


def _settle(stub):
    """Run the real method against the stub."""
    from warp.trainer.trainer_window import WarpCoreWindow

    WarpCoreWindow._select_first_unless_user_moved(stub)


def test_first_screenshot_is_selected_when_nothing_was_touched(app):
    stub = _Stub(40)

    _settle(stub)

    assert stub._file_list.currentRow() == 0


def test_a_scrolled_list_is_left_alone(app, request):
    """The reported behaviour. Scrolling selects nothing, so the old guard
    saw an untouched list and jumped."""
    stub = _Stub(200)
    request.addfinalizer(stub._file_list.close)
    bar = stub._file_list.verticalScrollBar()
    bar.setValue(bar.maximum())
    assert bar.value() > 0, 'the list must actually be scrollable for this test'
    where = bar.value()

    _settle(stub)

    assert stub._file_list.currentRow() == -1
    assert bar.value() == where


def test_an_open_screenshot_is_reloaded_not_replaced(app):
    """Detection can change a screen type, so whatever is open is re-read —
    but the selection stays where the user put it."""
    stub = _Stub(40)
    stub._current_idx = 7

    _settle(stub)

    assert stub.loaded == [7]
    assert stub._file_list.currentRow() != 0


def test_an_empty_list_is_harmless(app):
    stub = _Stub(0)

    _settle(stub)

    assert stub._file_list.currentRow() == -1
    assert stub.loaded == []
