"""Live + previous-session log viewer for the launcher's 3rd tab.

Subscribes to `warp.debug` records via the pub/sub hook on that module
(thread-safe — the callback fires on whatever thread emitted the log;
we marshal to the GUI thread with a Qt signal). A combo switches between
the running session and the prior session loaded from `warp_debug.log.bak`.
"""
from __future__ import annotations

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QFont, QTextCursor
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton,
    QVBoxLayout, QWidget,
)

from warp import debug as _warp_debug


_LEVEL_LABEL = {
    'CURRENT':  'Current session (live)',
    'PREVIOUS': 'Previous session',
}


class LogViewWidget(QWidget):
    """Read-only log viewer with current/previous session toggle.

    `_new_line` decouples logger callbacks (any thread) from widget I/O —
    Qt routes the signal to the GUI thread via auto/queued connection.
    """

    _new_line = Signal(str, str, str)   # channel, level, line

    def __init__(self, channel: str = 'detection', parent: QObject | None = None):
        super().__init__(parent)
        self._channel = channel
        self._mode = 'CURRENT'
        self._build_ui()
        self._new_line.connect(self._on_new_line, Qt.ConnectionType.QueuedConnection)
        _warp_debug.subscribe(self._on_log_record)
        self._show_current_initial()

    # ── UI ──────────────────────────────────────────────────────────

    def _build_ui(self):
        v = QVBoxLayout(self)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(6)

        bar = QHBoxLayout()
        bar.setSpacing(8)
        bar.addWidget(QLabel('Source:', self))
        self._mode_combo = QComboBox(self)
        for key, lbl in _LEVEL_LABEL.items():
            self._mode_combo.addItem(lbl, key)
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        bar.addWidget(self._mode_combo)

        self._autoscroll = QCheckBox('Auto-scroll', self)
        self._autoscroll.setChecked(True)
        bar.addWidget(self._autoscroll)

        bar.addStretch(1)

        self._clear_btn = QPushButton('Clear view', self)
        self._clear_btn.setToolTip(
            'Clears the display only. The log file on disk is untouched.'
        )
        self._clear_btn.clicked.connect(lambda: self._view.clear())
        bar.addWidget(self._clear_btn)

        self._reload_btn = QPushButton('Reload', self)
        self._reload_btn.clicked.connect(self._reload_current_mode)
        bar.addWidget(self._reload_btn)

        v.addLayout(bar)

        self._view = QPlainTextEdit(self)
        self._view.setReadOnly(True)
        self._view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self._view.setMaximumBlockCount(20000)   # cap memory
        f = QFont('Monospace')
        f.setStyleHint(QFont.StyleHint.TypeWriter)
        f.setPointSize(9)
        self._view.setFont(f)
        v.addWidget(self._view, stretch=1)

        cur_path, bak_path = _warp_debug.log_paths(self._channel)
        self._path_lbl = QLabel(
            f'Current: {cur_path}    Previous: {bak_path}', self)
        self._path_lbl.setStyleSheet('color: #888;')
        v.addWidget(self._path_lbl)

    # ── Mode switching / loading ────────────────────────────────────

    def _on_mode_changed(self, _idx: int):
        self._mode = self._mode_combo.currentData()
        self._reload_current_mode()

    def _reload_current_mode(self):
        if self._mode == 'CURRENT':
            self._show_current_initial()
        else:
            self._show_previous()

    def _show_current_initial(self):
        # Load whatever's already been written this session, then live-tail.
        cur_path, _ = _warp_debug.log_paths(self._channel)
        text = self._read_tail(cur_path, max_lines=5000)
        self._view.setPlainText(text)
        self._scroll_to_end()

    def _show_previous(self):
        _, bak_path = _warp_debug.log_paths(self._channel)
        if not bak_path.exists():
            self._view.setPlainText(f'(no previous session — {bak_path} not found)')
            return
        text = self._read_tail(bak_path, max_lines=20000)
        self._view.setPlainText(text)
        self._scroll_to_end()

    @staticmethod
    def _read_tail(path, max_lines: int) -> str:
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
        except Exception as e:
            return f'(failed to read {path}: {e})'
        if len(lines) > max_lines:
            lines = lines[-max_lines:]
        return ''.join(lines)

    # ── Live tail ───────────────────────────────────────────────────

    def _on_log_record(self, channel: str, level: str, line: str):
        # Called from arbitrary threads — bounce through signal.
        if channel != self._channel:
            return
        self._new_line.emit(channel, level, line)

    def _on_new_line(self, _channel: str, _level: str, line: str):
        if self._mode != 'CURRENT':
            return
        self._view.appendPlainText(line)
        if self._autoscroll.isChecked():
            self._scroll_to_end()

    def _scroll_to_end(self):
        c = self._view.textCursor()
        c.movePosition(QTextCursor.MoveOperation.End)
        self._view.setTextCursor(c)

    # ── Lifecycle ───────────────────────────────────────────────────

    def closeEvent(self, event):
        _warp_debug.unsubscribe(self._on_log_record)
        super().closeEvent(event)
