"""Where a pinned tooltip is allowed to go.

Regression: the card was fitted into the *image* rather than into the visible
canvas. A screenshot smaller than the viewport therefore pushed cards back
inside the picture — shifted sideways, or flipped above their bbox — while
most of the canvas sat empty around it. The working area is what the user can
see, not what the image happens to cover.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtWidgets import QApplication, QScrollArea, QWidget

from warp.gui.pinned_tooltip import PinnedTooltip

VIEWPORT = (900, 700)
IMAGE = (300, 200)          # a screenshot far smaller than the canvas


@pytest.fixture
def canvas(request):
    """A small image-sized canvas centred in a large scroll area."""
    QApplication.instance() or QApplication([])
    area = QScrollArea()
    area.setWidgetResizable(False)
    area.setAlignment(Qt.AlignmentFlag.AlignCenter)
    area.resize(*VIEWPORT)
    inner = QWidget()
    inner.resize(*IMAGE)
    area.setWidget(inner)
    area.show()
    QApplication.processEvents()
    request.addfinalizer(area.close)
    inner._area = area                      # keep the area alive and reachable
    return inner


@pytest.fixture
def tip(canvas):
    t = PinnedTooltip(canvas)
    t.prepare('<b>Adaptive Offense</b><br>Personal Ground Traits — 0.96')
    return t


def _origin(canvas) -> QPoint:
    return canvas.mapTo(canvas._area.viewport(), QPoint(0, 0))


def test_card_lives_in_the_viewport_not_the_image(tip, canvas):
    assert tip.parentWidget() is canvas._area.viewport()


def test_card_may_extend_past_the_image_edge(tip, canvas):
    """A bbox at the image's right edge must not drag the card back inside."""
    anchor = QRect(IMAGE[0] - 30, 90, 20, 20)
    got = tip.place_for(anchor)

    expected_x = _origin(canvas).x() + anchor.center().x() + 2
    assert got.x() == expected_x
    assert got.x() + tip.width() > _origin(canvas).x() + IMAGE[0]


def test_card_does_not_flip_above_while_the_viewport_has_room(tip, canvas):
    """Flipping is for a real edge, not for the bottom of a small picture."""
    anchor = QRect(140, IMAGE[1] - 30, 20, 20)
    got = tip.place_for(anchor)

    assert got.y() > _origin(canvas).y() + anchor.bottom()


def test_card_is_still_clamped_by_the_visible_edge(tip, canvas):
    """The viewport is a real boundary — the card is a child of it."""
    viewport = canvas._area.viewport()
    far_right = QRect(IMAGE[0] + 10_000, 90, 20, 20)      # way off to the side
    got = tip.place_for(far_right)

    assert got.x() + tip.width() <= viewport.width()
    assert got.x() >= 0


def test_card_follows_the_canvas_when_it_scrolls():
    """Parented to the viewport, the card no longer rides along for free.

    Needs a canvas *larger* than the viewport: a smaller one is re-centred by
    the scroll area, so there is nothing to scroll and nudging it by hand is
    simply undone on the next layout pass.
    """
    QApplication.instance() or QApplication([])
    area = QScrollArea()
    area.setWidgetResizable(False)
    area.resize(*VIEWPORT)
    big = QWidget()
    big.resize(2000, 1500)
    area.setWidget(big)
    area.show()
    QApplication.processEvents()

    tip = PinnedTooltip(big)
    tip.show_for('<b>Adaptive Offense</b>', QRect(400, 300, 20, 20))
    before = tip.pos()

    area.verticalScrollBar().setValue(area.verticalScrollBar().value() + 200)
    QApplication.processEvents()

    assert tip.pos().y() == before.y() - 200
    assert tip.pos().x() == before.x()
    area.close()


def test_a_canvas_without_a_scroll_area_keeps_the_old_parenting():
    QApplication.instance() or QApplication([])
    lone = QWidget()
    lone.resize(*IMAGE)
    t = PinnedTooltip(lone)
    assert t.parentWidget() is lone
