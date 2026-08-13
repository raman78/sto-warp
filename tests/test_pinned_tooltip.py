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


def test_card_clears_the_bbox_below_it(pin):
    """Anchored on the bbox's bottom edge, plus Qt's (2, 16) tooltip offset —
    so the card sits below the slot with a gap instead of covering it."""
    anchor = QRect(50, 60, 30, 30)
    dx, dy = PinnedTooltip._OFFSET
    pos = pin.place_for(anchor)
    assert pos.x() == anchor.center().x() + dx
    assert pos.y() == anchor.bottom() + dy
    assert pos.y() > anchor.bottom()      # never over the slot itself


# ── identity with the real hover tooltip ───────────────────────────────────


def _live_tip(app):
    """The QTipLabel Qt is currently showing, or None."""
    for w in app.topLevelWidgets():
        if 'Tip' in w.metaObject().className():
            return w
    return None


@pytest.mark.parametrize('bbox,row', [
    ((40, 40, 64, 64), {'state': 'confirmed', 'auto_confirmed': False,
                        'slot': 'Fore Weapons',
                        'name': 'Quantum Torpedo Launcher', 'conf': 0.71}),
    ((200, 150, 48, 48), {'state': 'pending', 'slot': 'Fore Weapons',
                          'name': 'Phaser Beam Array Mk XII', 'conf': 0.92}),
])
def test_card_is_pixel_identical_to_the_hover_tooltip(app, tmp_path, bbox, row):
    """Same text ⇒ same box, same spot.

    Drives the real hover path (`_show_hover_tooltip`) and diffs the resulting
    QTipLabel against the pinned card, so this catches drift in either — font,
    margin, indent, wrap, Qt's 1 px of slack, or the anchor rule.
    """
    from PySide6.QtGui import QPixmap
    from PySide6.QtWidgets import QToolTip
    from warp.style import apply_dark_style

    apply_dark_style(app)                     # the QSS font rule matters here
    shot = tmp_path / 'shot.png'
    QPixmap(600, 320).save(str(shot))

    class _StubDataMgr:
        def get_annotations(self, path):
            return []

    w = aw.AnnotationWidget(_StubDataMgr())
    w.resize(600, 320)
    w.load_image(shot)
    w.show()
    app.processEvents()
    w.set_review_items([{'bbox': bbox, **row}])

    # 1. hover the row with the pin off — this is the production hover path
    w._show_hover_tooltip(0)
    app.processEvents()
    tip = _live_tip(app)
    if tip is None:                            # no tooltip machinery here
        pytest.skip('platform did not materialise a QTipLabel')
    hover_size, hover_pos = tip.size(), w.mapFromGlobal(tip.pos())
    QToolTip.hideText()
    app.processEvents()

    # 2. pin the same row
    w.set_pin_enabled(True)
    w.set_highlighted_row(0)
    app.processEvents()

    try:
        assert w._pin.size() == hover_size
        assert w._pin.pos() == hover_pos
    finally:
        w.close()


def test_card_is_clamped_at_the_right_edge(pin):
    """A bbox near the right edge cannot push the card off-screen."""
    pos = pin.place_for(QRect(350, 60, 30, 30))
    assert pos.x() == pin.parentWidget().rect().right() - pin.width()
    assert pos.x() + pin.width() <= pin.parentWidget().width()


def test_card_flips_above_when_there_is_no_room_below(pin):
    """With no room under the bbox the card goes above it — squeezing it back
    over the slot would defeat the point of anchoring on the bottom edge."""
    anchor = QRect(50, 250, 30, 30)
    _, dy = PinnedTooltip._OFFSET
    pos = pin.place_for(anchor)
    assert pos.y() + pin.height() < anchor.top()      # clear of the slot
    assert pos.y() == anchor.top() - dy - pin.height()
    assert pos.y() >= 0


def test_hover_anchor_back_solves_the_cards_position(pin):
    """QToolTip adds _OFFSET to what it is handed, so the hover tooltip must be
    anchored that much before the card's own top-left to land on it."""
    anchor = QRect(50, 60, 30, 30)
    dx, dy = PinnedTooltip._OFFSET
    pos   = pin.place_for(anchor)
    hover = pin.hover_anchor(anchor)
    assert (hover.x() + dx, hover.y() + dy) == (pos.x(), pos.y())


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
