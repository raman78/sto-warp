# warp/gui/pinned_tooltip.py
# A tooltip that stays put: shows the selected slot's card next to its bbox
# and keeps it there until the selection changes.
#
# Why not QToolTip: QToolTip is hover-scoped by contract — it hides on mouse
# move outside its rect, on any mouse press, and after a display timeout. None
# of that is configurable away, so a pinned card has to be a real widget.
#
# It is transparent to mouse events, so it never steals a click meant for a
# bbox underneath it. Both canvases use it: AnnotationWidget (WARP CORE
# training + Fast Correction) and _InteractiveCanvas (WARP results view).
#
# It parents itself to the scroll area's viewport rather than to the canvas.
# A child cannot paint outside its parent, and the canvas widget is sized to
# the *image* — so a card anchored near the edge of a small screenshot was
# being squeezed back inside the picture even with most of the viewport empty
# around it. The working area is the visible canvas, not the image. Anchors
# still arrive in canvas coordinates and are mapped here, and the card follows
# the canvas as it scrolls (see `eventFilter`).

from __future__ import annotations

from PySide6.QtCore    import QEvent, Qt, QPoint, QRect, QSize
from PySide6.QtGui     import QFontMetrics, QPalette
from PySide6.QtGui     import Qt as _GuiQt   # hosts mightBeRichText in PySide6
from PySide6.QtWidgets import (QAbstractScrollArea, QFrame, QLabel, QStyle,
                               QToolTip)

from warp.style import TOOLTIP_QSS_BODY


class PinnedTooltip(QLabel):
    """Frameless rich-text card anchored on a bbox on a canvas.

    Every property below is copied from Qt's own ``QTipLabel`` — font, palette,
    margin, indent, frame, word wrap — because the card stands in for a hover
    tooltip and any difference shows up immediately as a differently sized box
    for identical text.
    """

    # Gap from the anchor point to the card's top-left. Ours, deliberately:
    # QTipLabel derives its own offset from the cursor size, so it is (2, 16)
    # under the offscreen platform but (2, 24) on xcb and wayland — matching a
    # moving target is how the card ended up 8 px off on a real desktop. Both
    # hover and pin now draw this same widget, so nothing needs to be matched.
    _OFFSET = (2, 16)

    @staticmethod
    def anchor_point(rect: QRect) -> QPoint:
        """The point a tooltip for *rect* is shown from.

        Bottom edge, horizontal centre: Qt's own (2, 16) offset then drops the
        card clear of the slot instead of over it, leaving a visible gap. Both
        the hover tooltip and the pinned card go through here, so the two
        cannot end up in different places — and unlike the cursor, this anchor
        does not drift with every pass of the mouse over the same bbox.
        """
        return QPoint(rect.center().x(), rect.bottom())

    @staticmethod
    def _host_for(canvas):
        """The widget the card should live in: the viewport, when there is one.

        `QAbstractScrollArea.setWidget` reparents the canvas into the viewport,
        so the viewport is simply the canvas's parent. Falls back to the canvas
        itself for a canvas that is not inside a scroll area.
        """
        parent = canvas.parentWidget()
        grandparent = parent.parentWidget() if parent is not None else None
        if isinstance(grandparent, QAbstractScrollArea) and \
                parent is grandparent.viewport():
            return parent
        return canvas

    def __init__(self, canvas):
        self._canvas = canvas
        self._anchor: QRect | None = None
        host = self._host_for(canvas)
        super().__init__(host)
        if host is not canvas:
            # The canvas moves inside the viewport as the user scrolls, and its
            # own geometry changes on zoom. Either way the anchor has travelled
            # and the card has to follow it.
            canvas.installEventFilter(self)
        self.setTextFormat(Qt.TextFormat.RichText)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        # QTipLabel's own construction, in the same order.
        self.setFont(QToolTip.font())
        self.setPalette(QToolTip.palette())
        self.setForegroundRole(QPalette.ColorRole.ToolTipText)
        self.setBackgroundRole(QPalette.ColorRole.ToolTipBase)
        self.setFrameStyle(QFrame.Shape.NoFrame)
        self.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.setIndent(1)
        # Shared with the global QToolTip rule so the two cannot drift apart.
        # The font has to go through the stylesheet too: WARP_QSS sets
        # `QWidget { font-size: 11px }`, which a plain setFont() loses to —
        # and a bigger font is exactly what made the card the wrong size.
        # QTipLabel escapes that rule, so it keeps QToolTip.font() (9 pt).
        self.setStyleSheet(TOOLTIP_QSS_BODY + self._font_qss())
        self.ensurePolished()
        self.setMargin(1 + self.style().pixelMetric(
            QStyle.PixelMetric.PM_ToolTipLabelFrameWidth, None, self))
        self.hide()

    @staticmethod
    def _font_qss() -> str:
        """`font-*` declarations for whatever font Qt gives hover tooltips."""
        f = QToolTip.font()
        size = (f'{f.pointSizeF():g}pt' if f.pointSizeF() > 0
                else f'{f.pixelSize()}px')
        return f'font-family: "{f.family()}"; font-size: {size};'

    # ── public API ────────────────────────────────────────────────────────

    def prepare(self, html: str) -> None:
        """Load *html* and take the size Qt's hover tooltip would take.

        Mirrors QTipLabel::reuseTip, including its 1 px of slack (and the
        extra row Qt adds for fonts with a small descent) — without it the
        card comes out a pixel narrower than the hover tooltip showing the
        very same text. Does not show: `place_for` needs a sized widget, and
        the hover path measures with a hidden one.
        """
        self.setWordWrap(_GuiQt.mightBeRichText(html))
        self.setText(html)
        fm = QFontMetrics(self.font())
        extra = QSize(1, 0)
        if fm.descent() == 2 and fm.ascent() >= 11:
            extra.setHeight(extra.height() + 1)
        self.resize(self.sizeHint() + extra)

    def show_for(self, html: str, anchor: QRect) -> None:
        """Show *html* under *anchor* (canvas coordinates)."""
        if not html:
            self.hide()
            return
        self._anchor = QRect(anchor)
        self.prepare(html)
        self.move(self.place_for(anchor))
        self.show()
        self.raise_()

    def eventFilter(self, watched, event):
        """Keep the card on its bbox while the canvas moves under it.

        Parented to the viewport, the card no longer rides along with the image
        for free, so a scroll or a zoom has to be answered by re-placing it.
        """
        if (watched is self._canvas and self.isVisible()
                and self._anchor is not None
                and event.type() in (QEvent.Type.Move, QEvent.Type.Resize)):
            self.move(self.place_for(self._anchor))
        return super().eventFilter(watched, event)

    # ── placement ─────────────────────────────────────────────────────────

    def place_for(self, anchor: QRect) -> QPoint:
        """Top-left for a card of the current size against bbox *anchor*.

        The one authority on where a card for a bbox goes; hover and pin are
        two instances of this widget, both placed through here.

        Below the bbox by default. With no room below, it flips *above* rather
        than being squeezed back over the slot, which is the whole point of
        anchoring on the bottom edge. Qt does the same for its own tooltips at
        a screen edge; the difference is the box being fitted into — the
        visible canvas here, since the card is a child of it and would
        otherwise just be clipped.

        *anchor* is in canvas coordinates. The card lives in the viewport, so
        the anchor is mapped across first: the box to fit into is the visible
        area, never the image, which may be far smaller than it.
        """
        host = self.parentWidget()
        if host is not self._canvas:
            anchor = QRect(self._canvas.mapTo(host, anchor.topLeft()),
                           anchor.size())
            area = host.rect()
        else:
            area = host.visibleRegion().boundingRect()
            if area.isEmpty():
                area = host.rect()
        w, h = self.width(), self.height()
        dx, dy = self._OFFSET
        point = self.anchor_point(anchor)

        x = point.x() + dx
        x = max(area.left(), min(x, max(area.left(), area.right() - w)))

        y = point.y() + dy
        if y + h > area.bottom():
            y = anchor.top() - dy - h          # flip above, same visual gap
        y = max(area.top(), min(y, max(area.top(), area.bottom() - h)))
        return QPoint(int(x), int(y))

