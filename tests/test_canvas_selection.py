"""Selecting on the canvas and selecting in the review list are one action.

The canvas carries two kinds of highlight — one row (a slot) and a set of rows
(a group header) — and they are mutually exclusive: one selection at a time.
Nothing may leave both standing, whichever side started it.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

import warp.trainer.annotation_widget as aw


@pytest.fixture
def widget():
    QApplication.instance() or QApplication([])

    class _StubDataMgr:
        def get_annotations(self, path):
            return []

    w = aw.AnnotationWidget(_StubDataMgr())
    w.resize(400, 300)
    w.set_review_items([
        {"bbox": (0, 0, 10, 10), "state": "pending", "name": "A", "slot": "Fore Weapons"},
        {"bbox": (20, 20, 10, 10), "state": "pending", "name": "B", "slot": "Fore Weapons"},
        {"bbox": (40, 40, 10, 10), "state": "pending", "name": "C", "slot": "Aft Weapons"},
    ])
    yield w
    w.close()


def _highlighted(w) -> set[int]:
    """Every row the canvas would draw as highlighted."""
    rows = set(w._highlighted_rows)
    if w._highlighted_row >= 0:
        rows.add(w._highlighted_row)
    return rows


def test_selecting_a_slot_drops_a_group_highlight(widget):
    """Regression: a canvas click syncs the list with signals blocked, so the
    adapter never got to clear the group — both highlights stayed on screen."""
    widget.set_highlighted_rows({0, 1})     # group header clicked in the list
    widget.set_highlighted_row(2)           # then a bbox clicked on the canvas

    assert _highlighted(widget) == {2}


def test_selecting_a_group_drops_a_slot_highlight(widget):
    widget.set_highlighted_row(2)
    widget.set_highlighted_rows({0, 1})

    assert _highlighted(widget) == {0, 1}


def test_clearing_leaves_nothing_highlighted(widget):
    widget.set_highlighted_rows({0, 1})
    widget.clear_highlight()

    assert _highlighted(widget) == set()


def test_empty_group_clears_the_group_highlight(widget):
    widget.set_highlighted_rows({0, 1})
    widget.set_highlighted_rows(())

    assert _highlighted(widget) == set()


def test_full_edit_selection_supersedes_a_group(widget):
    """Manual bbox edit takes over the canvas — nothing else stays lit."""
    widget.set_highlighted_rows({0, 1})
    widget.set_selected_row(2)

    assert _highlighted(widget) == set()
    assert widget._selected_row == 2
