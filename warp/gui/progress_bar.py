"""Shared status-bar progress widget with built-in Cancel button.

Used by both WARP and WARP CORE so the two tools surface their detection
progress identically — a QProgressBar embedded in the QMainWindow's status
bar plus a `Cancel` button on its right edge. The per-stage status text
(e.g. "[1/3] image.png  ·  OCR…") stays on the status-bar message label,
exactly as WARP already did before this refactor; this widget owns only
the bar + cancel chrome so callers keep using `statusBar().showMessage(…)`
for the text breakdown.

Cancellation is cooperative: clicking `Cancel` emits `cancel_requested`,
the caller is expected to flip an interruption flag / `QThread.requestInterruption()`
on its worker. The worker must poll that flag from its progress callback —
`QThread.terminate()` is intentionally not used (corrupts OpenCV/torch state).
"""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QProgressBar, QPushButton, QWidget

from warp.style import secondary_btn_style


class StatusProgressBar(QWidget):
    """Compact progress bar + Cancel button, sized for a QStatusBar."""

    cancel_requested = Signal()

    def __init__(self, parent=None, bar_min_width: int = 320):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        # The bar keeps its natural sizeHint (matches the pre-refactor
        # WARP look) but won't shrink below `bar_min_width` so the Cancel
        # button can't squish it. Cancel sits at the far right of this
        # widget which, because the widget itself is the status bar's
        # right-most permanent widget, lands at the window corner.
        self._bar = QProgressBar(self)
        self._bar.setMinimumWidth(bar_min_width)
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._bar.setTextVisible(True)
        lay.addWidget(self._bar, stretch=1)

        self._cancel = QPushButton('Cancel', self)
        self._cancel.setFixedWidth(70)
        self._cancel.setStyleSheet(secondary_btn_style())
        self._cancel.setToolTip(
            'Stop the running detection at the next progress checkpoint.'
        )
        self._cancel.clicked.connect(self.cancel_requested.emit)
        lay.addWidget(self._cancel, stretch=0)

        self.setVisible(False)

    # ── Lifecycle ───────────────────────────────────────────────────

    def start(self, determinate: bool = True, maximum: int = 100) -> None:
        """Show the widget and reset state for a fresh run.

        `determinate=False` switches the bar into the marquee animation
        used while we have no measurable progress (e.g. icon matcher's
        opaque inner loop)."""
        if determinate:
            self._bar.setRange(0, maximum)
            self._bar.setValue(0)
        else:
            self._bar.setRange(0, 0)
        self._cancel.setEnabled(True)
        self._cancel.setText('Cancel')
        self.setVisible(True)

    def set_progress(self, value: int) -> None:
        """Set absolute value in determinate mode; no-op for marquee."""
        if self._bar.maximum() == 0:
            return
        self._bar.setValue(max(0, min(self._bar.maximum(), value)))

    def set_cancel_enabled(self, enabled: bool) -> None:
        self._cancel.setEnabled(enabled)

    def finish(self) -> None:
        """Hide the widget. Caller is responsible for any "Done." text on
        the status-bar message label."""
        self._bar.setRange(0, 100)
        self._bar.setValue(100)
        self.setVisible(False)
