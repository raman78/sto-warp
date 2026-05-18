"""sto-warp launcher window.

Single-window host for both the WARP recognition GUI and the WARP CORE
trainer GUI. Saves the user from juggling separate processes / windows
and gives us one place to drive app-wide concerns:

  - Auto-sync at startup (community knowledge + central model + crop upload).
  - Periodic re-sync every 5 minutes.
  - Manual "Refresh" button on the toolbar, guarded by a mutex so it can
    never run concurrently with the auto-sync that's already in flight.
  - Status labels showing the current sync stage.

Both child windows are QMainWindow instances embedded as tabs. Qt allows
nesting QMainWindow inside a QTabWidget — toolbars/statusbars belong to
each child and keep working.
"""
from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QByteArray, QSettings
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QTabWidget,
)

from warp.debug import log
from warp.resources import resource_path


def _icon(name: str) -> QIcon:
    """Best-effort load of a packaged icon. Returns a null QIcon on miss."""
    try:
        p = resource_path(name)
        if p.is_file():
            return QIcon(str(p))
    except Exception as e:
        log.debug(f'Launcher: icon {name!r} unavailable: {e}')
    return QIcon()


def _find_sets_root() -> Path:
    p = Path(__file__).resolve()
    for _ in range(8):
        if (p / 'pyproject.toml').exists():
            return p
        p = p.parent
    return Path('.')


_SETTINGS_GEOMETRY = 'launcher/geometry'
_SETTINGS_STATE    = 'launcher/window_state'


class LauncherWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('sto-warp')
        _app_icon = _icon('SETS_icon_small.png')
        if not _app_icon.isNull():
            self.setWindowIcon(_app_icon)

        # Shared SETS-app shim — both windows reference the same cargo
        # cache and the same `_warp_core_window` pointer used by sync.
        from warp.data.cargo import app_view
        self._sets_app  = app_view()
        self._sets_root = _find_sets_root()

        self._build_ui()
        self._init_sync()
        self._install_desktop_entry()
        self._restore_window_state()

    # ── UI ────────────────────────────────────────────────────────────

    def _build_ui(self):
        from warp.gui.warp_window import WarpWindow
        from warp.gui.log_view import LogViewWidget
        from warp.trainer.trainer_window import WarpCoreWindow

        self._tabs = QTabWidget(self)
        self._tabs.setDocumentMode(True)
        self.setCentralWidget(self._tabs)

        self._warp_win = WarpWindow()
        warp_idx = self._tabs.addTab(self._warp_win, 'WARP — Recognition')
        _warp_icon = _icon('warp.jpg')
        if not _warp_icon.isNull():
            self._tabs.setTabIcon(warp_idx, _warp_icon)

        self._core_win = WarpCoreWindow(sets_app=self._sets_app, embed=True)
        core_idx = self._tabs.addTab(self._core_win, 'WARP CORE — Trainer')
        _core_icon = _icon('warp_core_icon.png')
        if not _core_icon.isNull():
            self._tabs.setTabIcon(core_idx, _core_icon)

        self._log_view = LogViewWidget(channel='detection')
        self._tabs.addTab(self._log_view, 'Detection logs')
        # Wipe the live log every time WARP starts a fresh run — keeps
        # noise from earlier runs out of the new run's summary.
        self._warp_win.detection_started.connect(self._log_view.clear_live)

        self._syslog_view = LogViewWidget(channel='system')
        self._tabs.addTab(self._syslog_view, 'System logs')

        # Refresh button lives in the status bar (bottom-right corner).
        self._refresh_btn = QPushButton('🔄 Refresh', self)
        self._refresh_btn.setToolTip(
            'Re-download community knowledge, check for a newer central '
            'model, and upload pending confirmed crops.'
        )
        self._refresh_btn.clicked.connect(self._on_refresh_clicked)
        self.statusBar().addWidget(self._refresh_btn)
        self.statusBar().showMessage('Starting sync…')

    # ── Sync orchestration ───────────────────────────────────────────

    def _init_sync(self):
        from warp.gui.sync_coordinator import SyncCoordinator
        self._coord = SyncCoordinator(self._sets_app, self._sets_root, parent=self)

        # Embedded trainer relies on `_sync_client` for the contribute()
        # path triggered from "Accept". Hand it the launcher's instance
        # so we don't run two parallel knowledge downloaders.
        if self._coord.sync_client is not None:
            self._core_win._sync_client = self._coord.sync_client

        self._coord.busy_changed.connect(self._on_busy_changed)
        self._coord.status.connect(self._on_status)

        # Kick off the initial cycle on the next event loop tick so the
        # window is visible before sync prints to the status bar.
        from PySide6.QtCore import QTimer
        QTimer.singleShot(500, self._coord.start)

    # ── Desktop integration ─────────────────────────────────────────

    def _install_desktop_entry(self):
        try:
            from warp.gui.desktop_install import install_desktop_entry
            install_desktop_entry()
        except Exception as e:
            log.debug(f'Launcher: desktop install skipped: {e}')

    def _on_refresh_clicked(self):
        self._coord.request_refresh(force=True)

    def _on_busy_changed(self, busy: bool):
        self._refresh_btn.setEnabled(not busy)

    def _on_status(self, text: str):
        self.statusBar().showMessage(text)

    # ── Lifecycle ────────────────────────────────────────────────────

    def _restore_window_state(self):
        s = QSettings()
        geom = s.value(_SETTINGS_GEOMETRY)
        if isinstance(geom, QByteArray) and not geom.isEmpty():
            self.restoreGeometry(geom)
        else:
            self.resize(1320, 800)
        state = s.value(_SETTINGS_STATE)
        if isinstance(state, QByteArray) and not state.isEmpty():
            self.restoreState(state)

    def closeEvent(self, event):
        try:
            s = QSettings()
            s.setValue(_SETTINGS_GEOMETRY, self.saveGeometry())
            s.setValue(_SETTINGS_STATE, self.saveState())
        except Exception as e:
            log.debug(f'Launcher: geometry save failed: {e}')
        try:
            self._coord.stop()
        except Exception as e:
            log.debug(f'Launcher: coordinator stop failed: {e}')
        super().closeEvent(event)


def main(argv: list[str] | None = None) -> int:
    app = QApplication.instance() or QApplication(argv or sys.argv)
    QApplication.setOrganizationName('sto-warp')
    QApplication.setApplicationName('sto-warp')
    win = LauncherWindow()
    win.show()
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
