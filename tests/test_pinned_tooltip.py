"""Tests for the pinned (sticky) tooltip on the annotation canvas.

The pin shows the *selected* slot's card beside its bbox and keeps it there.
Hover tooltips are untouched — except on the pinned slot itself, where a hover
copy would only repeat what is already on screen.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QRect
from PySide6.QtWidgets import QApplication, QWidget

import warp.trainer.annotation_widget as aw
from warp.gui.pinned_tooltip import PinnedTooltip


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


# ── placement ──────────────────────────────────────────────────────────────


@pytest.fixture
def pin(app):
    parent = QWidget()
    parent.resize(400, 300)
    p = PinnedTooltip(parent)
    p.resize(100, 40)
    yield p
    parent.close()


def test_card_lands_where_a_hover_tooltip_would(pin):
    """Qt offsets a hover tooltip from the cursor by (2, 16); the pinned card
    uses the same offset from the bbox centre, so pinning does not move it."""
    anchor = QRect(50, 60, 30, 30)
    dx, dy = PinnedTooltip._OFFSET
    pos = pin._place(anchor)
    assert pos.x() == anchor.center().x() + dx
    assert pos.y() == anchor.center().y() + dy


def test_card_is_clamped_at_the_right_edge(pin):
    """A bbox near the right edge cannot push the card off-screen."""
    pos = pin._place(QRect(350, 60, 30, 30))
    assert pos.x() == pin.parentWidget().rect().right() - pin.width()
    assert pos.x() + pin.width() <= pin.parentWidget().width()


def test_card_is_clamped_at_the_bottom_edge(pin):
    pos = pin._place(QRect(50, 290, 30, 30))
    assert pos.y() == pin.parentWidget().rect().bottom() - pin.height()
    assert pos.y() + pin.height() <= pin.parentWidget().height()


# ── AnnotationWidget integration ───────────────────────────────────────────


@pytest.fixture
def widget(app):
    class _StubDataMgr:
        def get_annotations(self, path):
            return []

    w = aw.AnnotationWidget(_StubDataMgr())
    w.resize(400, 300)
    w.set_review_items([
        {"bbox": (0, 0, 10, 10), "state": "confirmed",
         "name": "Item A", "slot": "fore_weapon"},
        {"bbox": (20, 20, 10, 10), "state": "pending",
         "name": "Item B", "slot": "fore_weapon", "conf": 0.9},
    ])
    yield w
    w.close()


def test_no_card_while_the_toggle_is_off(widget):
    widget.set_highlighted_row(0)
    assert widget._pinned_row == -1
    assert widget._pin is None


def test_selecting_a_row_pins_its_card(widget):
    widget.set_pin_enabled(True)
    widget.set_highlighted_row(1)

    assert widget._pinned_row == 1
    assert not widget._pin.isHidden()
    assert "Item B" in widget._pin.text()


def test_moving_the_selection_moves_the_card(widget):
    widget.set_pin_enabled(True)
    widget.set_highlighted_row(1)
    widget.set_highlighted_row(0)

    assert widget._pinned_row == 0
    assert "Item A" in widget._pin.text()


def test_clearing_the_selection_hides_the_card(widget):
    widget.set_pin_enabled(True)
    widget.set_highlighted_row(0)
    widget.clear_highlight()

    assert widget._pinned_row == -1
    assert widget._pin.isHidden()


def test_group_selection_hides_the_card(widget):
    """A group header selects many rows — there is no single slot to pin."""
    widget.set_pin_enabled(True)
    widget.set_highlighted_row(0)
    widget.set_highlighted_rows({0, 1})

    assert widget._pinned_row == -1
    assert widget._pin.isHidden()


def test_turning_the_toggle_off_hides_the_card(widget):
    widget.set_pin_enabled(True)
    widget.set_highlighted_row(0)
    widget.set_pin_enabled(False)

    assert widget._pinned_row == -1
    assert widget._pin.isHidden()


def test_row_without_bbox_is_not_pinned(widget):
    widget.set_review_items([{"state": "pending", "name": "No box",
                              "slot": "fore_weapon", "conf": 0.5}])
    widget.set_pin_enabled(True)
    widget.set_highlighted_row(0)

    assert widget._pinned_row == -1


# ── hover interaction ──────────────────────────────────────────────────────


def test_hover_is_suppressed_on_the_pinned_row(widget, monkeypatch):
    """The pinned slot must not also tooltip under the cursor."""
    widget.set_pin_enabled(True)
    widget.set_highlighted_row(0)
    # Patch after pinning — the pin composes the same card, and we only want
    # to observe what the *hover* path asks for.
    calls = []
    monkeypatch.setattr(widget, '_tooltip_html_for_row',
                        lambda row: calls.append(row) or '')

    widget._show_hover_tooltip(0)

    assert calls == []


def test_hover_still_works_on_every_other_row(widget, monkeypatch):
    widget.set_pin_enabled(True)
    widget.set_highlighted_row(0)
    calls = []
    monkeypatch.setattr(widget, '_tooltip_html_for_row',
                        lambda row: calls.append(row) or '')

    widget._show_hover_tooltip(1)

    assert calls == [1]


# ── WARP results canvas ────────────────────────────────────────────────────
#
# Same feature, second canvas: _InteractiveCanvas keys off the highlight set
# instead of a row index, so its pin gate is worth pinning down separately.


@pytest.fixture
def canvas(app):
    from warp.gui.results_view import _InteractiveCanvas
    from warp.warp_importer import RecognisedItem

    c = _InteractiveCanvas()
    c.resize(400, 300)
    items = [
        RecognisedItem(slot='Fore Weapons', slot_index=0, name='Item A',
                       confidence=0.9, bbox=(0, 0, 10, 10)),
        RecognisedItem(slot='Fore Weapons', slot_index=1, name='Item B',
                       confidence=0.8, bbox=(40, 40, 10, 10)),
    ]
    c._items = items
    c._gidx  = [0, 1]
    yield c
    c.close()


def test_results_canvas_pins_the_highlighted_item(canvas):
    canvas.set_pin_enabled(True)
    canvas.set_highlight(1)

    assert canvas._pinned_gidx == 1
    assert 'Item B' in canvas._pin.text()


def test_results_canvas_group_highlight_hides_the_card(canvas):
    canvas.set_pin_enabled(True)
    canvas.set_highlight(0)
    canvas.set_highlight_set([0, 1])

    assert canvas._pinned_gidx == -1
    assert canvas._pin.isHidden()


def test_results_canvas_hover_suppressed_on_pinned_item(canvas, monkeypatch):
    canvas.set_pin_enabled(True)
    canvas.set_highlight(0)
    calls = []
    monkeypatch.setattr(canvas, '_tooltip_html_for_gidx',
                        lambda g: calls.append(g) or '')

    canvas._show_hover_tooltip(0)
    canvas._show_hover_tooltip(1)

    assert calls == [1]
