# warp/gui/pinned_tooltip.py
# A tooltip that stays put: shows the selected slot's card next to its bbox
# and keeps it there until the selection changes.
#
# Why not QToolTip: QToolTip is hover-scoped by contract — it hides on mouse
# move outside its rect, on any mouse press, and after a display timeout. None
# of that is configurable away, so a pinned card has to be a real widget.
#
# It is a child of the canvas, so it scrolls and clips with the image, and it
# is transparent to mouse events, so it never steals a click meant for a bbox
# underneath it. Both canvases use it: AnnotationWidget (WARP CORE training +
# Fast Correction) and _InteractiveCanvas (WARP results view).

from __future__ import annotations

from PySide6.QtCore    import Qt, QPoint, QRect, QSize
from PySide6.QtGui     import QFontMetrics, QPalette
from PySide6.QtGui     import Qt as _GuiQt   # hosts mightBeRichText in PySide6
from PySide6.QtWidgets import QFrame, QLabel, QStyle, QToolTip

from warp.style import TOOLTIP_QSS_BODY


class PinnedTooltip(QLabel):
    """Frameless rich-text card anchored on a bbox on a canvas.

    Every property below is copied from Qt's own ``QTipLabel`` — font, palette,
    margin, indent, frame, word wrap — because the card stands in for a hover
    tooltip and any difference shows up immediately as a differently sized box
    for identical text.
    """

    # Qt offsets a hover tooltip from the cursor by exactly this much
    # (QTipLabel: pos + (2, 16), measured). The pinned card applies it to the
    # bbox centre: a selection has no cursor, and the centre is the one point
    # that does not depend on how the user got there.
    _OFFSET = (2, 16)

    def __init__(self, parent):
        super().__init__(parent)
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

    def show_for(self, html: str, anchor: QRect) -> None:
        """Show *html* beside *anchor* (canvas coordinates)."""
        if not html:
            self.hide()
            return
        # The rest of this mirrors QTipLabel::reuseTip, including its 1 px of
        # slack (and the extra row Qt adds for fonts with a small descent) —
        # without it the card comes out a pixel narrower than the hover
        # tooltip showing the very same text.
        self.setWordWrap(_GuiQt.mightBeRichText(html))
        self.setText(html)
        fm = QFontMetrics(self.font())
        extra = QSize(1, 0)
        if fm.descent() == 2 and fm.ascent() >= 11:
            extra.setHeight(extra.height() + 1)
        self.resize(self.sizeHint() + extra)
        self.move(self._place(anchor))
        self.show()
        self.raise_()

    # ── placement ─────────────────────────────────────────────────────────

    def _place(self, anchor: QRect) -> QPoint:
        """Sit below-right of the bbox centre, clamped to stay visible.

        Same geometry a hover tooltip would get with the cursor in the middle
        of the bbox. The clamp mirrors what Qt does near a screen edge; the
        visible region is what the scroll area actually shows, so the card
        lands on screen even when the canvas dwarfs the viewport.
        """
        area = self.parentWidget().visibleRegion().boundingRect()
        if area.isEmpty():
            area = self.parentWidget().rect()
        w, h = self.width(), self.height()
        dx, dy = self._OFFSET

        x = anchor.center().x() + dx
        x = max(area.left(), min(x, max(area.left(), area.right() - w)))

        y = anchor.center().y() + dy
        y = max(area.top(), min(y, max(area.top(), area.bottom() - h)))
        return QPoint(int(x), int(y))
