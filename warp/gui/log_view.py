"""Live + previous-session log viewer for the launcher's 3rd tab.

Subscribes to `warp.debug` records via the pub/sub hook on that module
(thread-safe — the callback fires on whatever thread emitted the log;
we marshal to the GUI thread with a Qt signal). A combo switches between
the running session and the prior session loaded from `warp_debug.log.bak`.
"""
from __future__ import annotations

import datetime
from pathlib import Path
from typing import Callable

import os

from PySide6.QtCore import QEvent, QObject, Qt, QUrl, Signal
from PySide6.QtGui import QCursor, QDesktopServices, QFont, QGuiApplication
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel,
    QPlainTextEdit, QPushButton, QVBoxLayout, QWidget,
)

from warp import debug as _warp_debug
from warp.debug import syslog


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
        # Optional caller-supplied hook returning the proposed file name
        # (no path, no extension) for the Save As dialog. WARP and WARP
        # CORE both inject one so the dialog opens with a name derived
        # from the currently-selected screenshot.
        self._default_save_name_cb: Callable[[], str] | None = None
        self._build_ui()
        self._new_line.connect(self._on_new_line, Qt.ConnectionType.QueuedConnection)
        _warp_debug.subscribe(self._on_log_record)
        self._show_current_initial()
        # Wheel-event probe — opt-in diagnostic for the "dead zone" where
        # mouse wheel scrolling stops working over a vertical band of the
        # log viewer. When WARP_WHEEL_PROBE=1, install an app-wide filter
        # that prints which widget actually receives each QWheelEvent we
        # see over this widget's subtree (and whether the event is accepted).
        # Output goes to the system log channel so the detection log we're
        # debugging stays uncluttered.
        self._wheel_probe = None
        if os.environ.get('WARP_WHEEL_PROBE') == '1':
            self._wheel_probe = _WheelProbe(self)
            QApplication.instance().installEventFilter(self._wheel_probe)
            syslog.info(f'log_view: wheel probe ENABLED on {self._channel!r}')

    def set_default_save_name_cb(self, cb: Callable[[], str] | None) -> None:
        """Set the hook that supplies the Save As dialog's default file
        stem (no extension). When unset, falls back to a timestamped
        `detection_<channel>_<YYYYMMDD-HHMMSS>` name."""
        self._default_save_name_cb = cb

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

        self._open_dir_btn = QPushButton('Open folder', self)
        self._open_dir_btn.setToolTip(
            'Open the log directory in the system file manager.'
        )
        self._open_dir_btn.clicked.connect(self._open_log_dir)
        bar.addWidget(self._open_dir_btn)

        self._copy_all_btn = QPushButton('Copy All', self)
        self._copy_all_btn.setToolTip(
            'Copy the visible log buffer to the system clipboard.'
        )
        self._copy_all_btn.clicked.connect(self._copy_all_to_clipboard)
        bar.addWidget(self._copy_all_btn)

        self._save_as_btn = QPushButton('Save As…', self)
        self._save_as_btn.setToolTip(
            'Save the visible log buffer to a .log file.'
        )
        self._save_as_btn.clicked.connect(self._save_as)
        bar.addWidget(self._save_as_btn)

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
        self._set_text_preserving_hscroll(text)
        self._scroll_to_end()

    def _show_previous(self):
        _, bak_path = _warp_debug.log_paths(self._channel)
        if not bak_path.exists():
            self._set_text_preserving_hscroll(
                f'(no previous session — {bak_path} not found)')
            return
        text = self._read_tail(bak_path, max_lines=20000)
        self._set_text_preserving_hscroll(text)
        self._scroll_to_end()

    def _set_text_preserving_hscroll(self, text: str):
        # setPlainText leaves the horizontal scrollbar at 0, which is what
        # we want for a fresh load — but new lines arriving via
        # appendPlainText would later drag it right. We anchor h=0 here
        # explicitly so the initial view always opens left-aligned.
        self._view.setPlainText(text)
        self._view.horizontalScrollBar().setValue(0)

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
        if _level == 'CLEAR':
            self.clear_live()
            return
        # appendPlainText calls ensureCursorVisible() under the hood, which
        # snaps the horizontal scrollbar to the end of the inserted line.
        # Save the user's horizontal position before, restore after — so
        # they stay wherever they parked the view (left edge by default).
        hsb = self._view.horizontalScrollBar()
        h_before = hsb.value()
        self._view.appendPlainText(line)
        hsb.setValue(h_before)
        if self._autoscroll.isChecked():
            self._scroll_to_end()

    def _open_log_dir(self):
        cur_path, _ = _warp_debug.log_paths(self._channel)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(cur_path.parent)))

    def _copy_all_to_clipboard(self):
        cb = QGuiApplication.clipboard()
        if cb is not None:
            cb.setText(self._view.toPlainText())

    def _default_save_stem(self) -> str:
        ts = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        # Caller-supplied stem wins (e.g. screenshot name + space/ground).
        if self._default_save_name_cb is not None:
            try:
                stem = self._default_save_name_cb()
            except Exception:
                stem = ''
            if stem:
                # Caller may or may not include the timestamp themselves.
                # Don't second-guess — use the stem verbatim.
                return stem
        return f'detection_{self._channel}_{ts}'

    def _save_as(self):
        stem = self._default_save_stem()
        # Last used directory: log file's directory by default.
        cur_path, _ = _warp_debug.log_paths(self._channel)
        suggested = str(Path(cur_path).parent / f'{stem}.log')
        path, _flt = QFileDialog.getSaveFileName(
            self, 'Save detection log',
            suggested, 'Log files (*.log);;Text files (*.txt);;All files (*)',
        )
        if not path:
            return
        try:
            Path(path).write_text(self._view.toPlainText(), encoding='utf-8')
        except OSError as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, 'Save failed', f'{type(e).__name__}: {e}')

    def clear_live(self):
        """Wipe the view only when tailing the live session — leaves the
        Previous-session buffer alone so reviewing old logs is undisturbed."""
        if self._mode == 'CURRENT':
            self._view.clear()

    def _scroll_to_end(self):
        # Terminal-style: jump to the newest line vertically, but leave the
        # horizontal scrollbar wherever the user parked it. The cursor-based
        # variant called ensureCursorVisible() which yanked the viewport
        # right on every long line. Newline arrivals therefore no longer
        # cause horizontal jumps in the Detection logs tab.
        sb = self._view.verticalScrollBar()
        sb.setValue(sb.maximum())

    # ── Lifecycle ───────────────────────────────────────────────────

    def closeEvent(self, event):
        _warp_debug.unsubscribe(self._on_log_record)
        if self._wheel_probe is not None:
            QApplication.instance().removeEventFilter(self._wheel_probe)
            self._wheel_probe = None
        super().closeEvent(event)


class _WheelProbe(QObject):
    """Diagnostic event filter that logs every QWheelEvent whose receiver
    (or the widget under the cursor) lives inside the host LogViewWidget.

    For each such event we record:
      - the Qt event recipient (the widget Qt is delivering to),
      - the widget actually under the mouse cursor at that instant,
      - the local x within the LogViewWidget (helps spot the "1/4 strip"),
      - whether the event has been .accepted() so far.

    Enable with `WARP_WHEEL_PROBE=1`. Output goes to the system log. The
    probe never marks events accepted — it's pure observation."""

    def __init__(self, host: 'LogViewWidget'):
        super().__init__(host)
        self._host = host

    def eventFilter(self, obj, event):
        if event.type() != QEvent.Type.Wheel:
            return False
        host = self._host
        if host is None:
            return False
        try:
            gpos = QCursor.pos()
            under = QApplication.widgetAt(gpos)

            def _in_host(w):
                cur = w
                while cur is not None:
                    if cur is host:
                        return True
                    cur = cur.parent() if hasattr(cur, 'parent') else None
                return False

            obj_in_host = hasattr(obj, 'parent') and _in_host(obj)
            under_in_host = under is not None and _in_host(under)
            if not (obj_in_host or under_in_host):
                return False

            host_local = host.mapFromGlobal(gpos)
            obj_name = type(obj).__name__
            obj_objname = obj.objectName() if hasattr(obj, 'objectName') else ''
            under_name = type(under).__name__ if under is not None else 'None'
            under_objname = under.objectName() if under is not None and hasattr(under, 'objectName') else ''
            accepted = event.isAccepted()
            syslog.info(
                f'[WHEEL] host=({host.width()}x{host.height()}) '
                f'local=({host_local.x()},{host_local.y()}) '
                f'recipient={obj_name}({obj_objname!r}) '
                f'underCursor={under_name}({under_objname!r}) '
                f'accepted={accepted}'
            )
        except Exception as e:
            try:
                syslog.warning(f'wheel probe error: {e}')
            except Exception:
                pass
        return False
