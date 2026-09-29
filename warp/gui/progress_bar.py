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
    """Progress bar + Cancel button, sized for a QStatusBar."""

    cancel_requested = Signal()
    visibility_changed = Signal()

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

    def setVisible(self, visible: bool) -> None:           # noqa: N802 (Qt)
        super().setVisible(visible)
        self.visibility_changed.emit()

    def mirror_messages(self, status_bar) -> None:
        """Write the status bar's message into the bar.

        The bar is laid out across the whole status line (`ProgressStrip`),
        which covers the message area, so what a run says about itself
        ("[1/3] image.png · OCR…") has to be inside the bar to be seen.
        The callers keep using `showMessage` as before."""
        status_bar.messageChanged.connect(self._on_message)

    def _on_message(self, text: str) -> None:
        self._bar.setFormat(f'{text}  %p%' if text else '%p%')

    def finish(self) -> None:
        """Hide the widget. Caller is responsible for any "Done." text on
        the status-bar message label."""
        self._bar.setRange(0, 100)
        self._bar.setValue(100)
        self.setVisible(False)


class RecognitionPrepBar(QProgressBar):
    """The recognition warm-up's progress, shown while Auto-Detect waits for it.

    Its own bar rather than `StatusProgressBar`: screen-type classification
    uses that one, and it can run while the warm-up does (opening a folder
    starts it), so sharing a bar would let each overwrite the other. The text
    sits inside the bar, and the bar is a permanent status-bar widget, so a
    status message cannot hide it. Hidden whenever no warm-up runs.
    """

    # Warm-up part names (`SETSIconMatcher.warm_up`) as the user reads them.
    _PARTS = {
        'ocr':             'text reader',
        'icon index':      'icon library',
        'models':          'models',
        'own crops':       'confirmed crops',
        'community crops': 'community crops',
        'session stack':   'crop index',
    }

    visibility_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTextVisible(True)
        f = self.font()
        f.setBold(True)
        self.setFont(f)
        self.setVisible(False)

    def setVisible(self, visible: bool) -> None:           # noqa: N802 (Qt)
        super().setVisible(visible)
        self.visibility_changed.emit()

    def report(self, done: int, total: int, part: str) -> bool:
        """Show one progress report; returns True while the warm-up runs."""
        if total <= 0 or done >= total:
            self.setVisible(False)
            return False
        self.setRange(0, total)
        self.setValue(done)
        name = self._PARTS.get(part, part)
        self.setFormat(f'Preparing recognition — {name} ({done + 1}/{total})  %p%')
        self.setVisible(True)
        return True


class ProgressStrip(QWidget):
    """The status line's progress area: its bars side by side, full width.

    Added to the status bar as a permanent widget with stretch, so while it
    is shown it takes the whole line and the status message area has no
    room — measured offscreen: a message set under a stretched permanent
    widget is not painted. That is the point: the line shows progress or
    the message, never a bar squeezed into half of it beside some text.

    Standalone, the strip shows only while one of its bars does, so an idle
    window still shows its message. Inside the launcher the messages go to
    the launcher's own status bar instead, and `set_keep_visible(True)`
    keeps the strip (empty when idle) so the tool's line holds only progress
    and does not change height between runs.
    """

    def __init__(self, parent=None, *bars):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self._bars = bars
        self.run_bar = bars[0]      # the one a detection run drives
        self._keep = False
        for b in bars:
            lay.addWidget(b, stretch=1)
            b.visibility_changed.connect(self._sync)
        # Kept on while idle, the strip must be as tall as with a bar in it,
        # or the whole tab shifts by a few pixels at every run's start/end.
        self.setMinimumHeight(max(b.sizeHint().height() for b in bars))
        self._sync()

    def set_keep_visible(self, keep: bool) -> None:
        self._keep = bool(keep)
        self._sync()

    def _sync(self) -> None:
        self.setVisible(self._keep or any(not b.isHidden() for b in self._bars))
