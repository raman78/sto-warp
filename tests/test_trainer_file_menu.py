"""Tests for the WARP CORE file-list menus.

Right-click → Copy filename / Copy full path (mirrors WARP tabs).
Double-click → pick screen type (moved off right-click).
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint
from PySide6.QtWidgets import (
    QApplication, QListWidget, QListWidgetItem, QMainWindow, QMenu,
)

import warp.trainer.trainer_window as tw
from warp.trainer.trainer_window import (
    WarpCoreWindow, SCREEN_TYPE_LABELS,
)


def _bare_window(paths: list[Path]) -> WarpCoreWindow:
    """A WarpCoreWindow with only the attributes the menu methods touch."""
    QApplication.instance() or QApplication([])
    w = WarpCoreWindow.__new__(WarpCoreWindow)
    # Initialise only the QMainWindow base (skip the heavy WarpCoreWindow
    # __init__) so `QMenu(self)` has a valid QWidget parent.
    QMainWindow.__init__(w)
    lst = QListWidget()
    for p in paths:
        lst.addItem(QListWidgetItem(p.name))
    w._file_list = lst
    w._screenshots = paths
    w._screen_types = {p.name: "UNKNOWN" for p in paths}
    return w


def _exec_picks(monkeypatch, *, text=None, data=None):
    """Swap the trainer's QMenu for a subclass whose exec() returns the
    action matching text/data — no real (blocking) popup.

    A QMenu.exec override must live on a Python subclass; assigning to the
    Shiboken QMenu type directly is ignored and the modal loop still runs.
    """
    class _FakeMenu(QMenu):
        def exec(self, *_a, **_k):
            for act in self.actions():
                if text is not None and act.text() == text:
                    return act
                if data is not None and act.data() == data:
                    return act
            return None

    monkeypatch.setattr(tw, "QMenu", _FakeMenu)


def test_right_click_copy_filename(monkeypatch):
    p = Path("/tmp/shots/overview.png")
    w = _bare_window([p])
    _exec_picks(monkeypatch, text="Copy filename")

    w._show_file_list_context_menu(QPoint(1, 1))

    assert QApplication.clipboard().text() == "overview.png"


def test_right_click_copy_full_path(monkeypatch):
    p = Path("/tmp/shots/overview.png")
    w = _bare_window([p])
    _exec_picks(monkeypatch, text="Copy full path")

    w._show_file_list_context_menu(QPoint(1, 1))

    assert QApplication.clipboard().text() == str(p)


def test_double_click_sets_screen_type(monkeypatch):
    p = Path("/tmp/shots/overview.png")
    w = _bare_window([p])
    w._on_type_override_changed = MagicMock()
    target_key = next(iter(SCREEN_TYPE_LABELS))
    _exec_picks(monkeypatch, data=target_key)

    w._show_screen_type_menu(w._file_list.item(0))

    w._on_type_override_changed.assert_called_once_with(target_key)
