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

from PySide6.QtCore    import Qt, QPoint, QRect
from PySide6.QtWidgets import QLabel

from warp.style import MBG, FG, LBG, ACCENT


class PinnedTooltip(QLabel):
    """Frameless rich-text card anchored on a bbox on a canvas."""

    # Qt offsets a hover tooltip from the cursor by exactly this much
    # (QTipLabel: pos + (2, 16), measured). The pinned card reuses it against
    # the bbox centre — the point the cursor would be on while hovering — so
    # pinning does not make the card jump somewhere else.
    _OFFSET = (2, 16)

    def __init__(self, parent):
        super().__init__(parent)
        self.setTextFormat(Qt.TextFormat.RichText)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        # Mirrors the QToolTip rule in style.py, with the accent border marking
        # it as pinned rather than hovered.
        self.setStyleSheet(
            f'background-color: {MBG}; color: {FG};'
            f'border: 1px solid {LBG}; border-left: 2px solid {ACCENT};'
            f'padding: 4px;'
        )
        self.hide()

    # ── public API ────────────────────────────────────────────────────────

    def show_for(self, html: str, anchor: QRect) -> None:
        """Show *html* beside *anchor* (canvas coordinates)."""
        if not html:
            self.hide()
            return
        self.setText(html)
        self.adjustSize()
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
