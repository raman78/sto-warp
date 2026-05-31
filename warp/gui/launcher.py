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
    QApplication, QMainWindow, QMessageBox, QPushButton, QTabWidget,
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
        self._gc_fast_correction_sessions()
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

        # WARP Results "Open in WARP CORE" context-menu action — switch
        # to the trainer tab and select the matching file. Wiring it here
        # (rather than in WarpWindow) keeps WARP free of any direct
        # reference to the trainer module.
        self._warp_win.open_in_warp_core.connect(self._on_open_in_warp_core)
        self._warp_win.set_warp_core_handler(True)

        # "Open in WARP Fast Correction Mode" — batch handoff of every
        # screenshot in the current WARP result. The launcher flips the
        # trainer tab into Fast Correction Mode and loads them all.
        self._warp_win.open_in_warp_fast_correction.connect(
            self._on_open_in_warp_fast_correction)
        self._warp_win.set_fast_correction_handler(True)

        # WARP CORE "↗ Send to WARP" — install the corrected ImportResult
        # into WARP and switch tabs so the user can run JSON export
        # without re-detecting.
        self._core_win.send_to_warp.connect(self._on_send_to_warp)
        # When the trainer leaves Fast Correction Mode, restore the
        # default tab title and swing the user back to the WARP tab so
        # the workflow loop closes naturally.
        self._core_win.fast_correction_exited.connect(
            self._on_fast_correction_exited)

        # Default tab titles — captured here so `set_fast_correction_mode`
        # can swap the trainer's tab label and restore it on exit.
        self._core_tab_default_title = self._tabs.tabText(core_idx)

        # Main-thread detection-log routing follows the active tab so any
        # synchronous `log.info(...)` from a UI callback lands in the
        # tool the user is currently looking at. Worker threads override
        # this per-thread via `use_detection_channel`, so concurrent
        # background runs in the other tool still keep their own scope.
        self._warp_idx = warp_idx
        self._core_idx = core_idx
        self._tabs.currentChanged.connect(self._on_main_tab_changed)
        self._on_main_tab_changed(self._tabs.currentIndex())

        # Detection logs live in each tool's own window (WARP / WARP CORE)
        # now — the launcher only hosts cross-tool concerns.
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

    def _gc_fast_correction_sessions(self):
        """Drop Fast Correction staging dirs older than 14 days.

        Runs once at launch — Fast Mode sessions are content-hashed
        snapshots of WARP batches and can accumulate indefinitely if the
        user keeps starting new batches without exiting cleanly.
        """
        try:
            from warp.trainer import fast_session
            fast_session.gc_old_sessions(max_age_days=14)
        except Exception as e:
            log.debug(f'Launcher: fast-correction GC skipped: {e}')

    def _on_main_tab_changed(self, idx: int):
        from warp.debug import set_main_detection_channel
        if idx == self._core_idx:
            set_main_detection_channel('detection_core')
        else:
            set_main_detection_channel('detection')

    def _on_open_in_warp_core(self, path: str, items: object):
        self._tabs.setCurrentIndex(self._core_idx)
        try:
            self._core_win.open_screenshot(path, preload_items=items or None)
        except Exception as e:
            log.warning(f'Launcher: open_screenshot({path!r}) failed: {e}')

    def _on_open_in_warp_fast_correction(self, items_by_file: dict, stype_by_file: dict):
        """Enter the trainer's Fast Correction Mode with WARP's batch.

        Re-entry while the trainer is already in Fast Mode shows a
        confirmation dialog: the running batch's annotation state is
        discarded when the user accepts. This is intentional — the
        ephemeral nature of Fast Mode is core to its design.
        """
        if not items_by_file:
            return
        if getattr(self._core_win, '_mode', 'training') == 'fast_correction':
            choice = QMessageBox.question(
                self,
                'Fast Correction already active',
                'A Fast Correction batch is already loaded.\n'
                'Replace it with the new batch from WARP?\n\n'
                '(Any unsaved Done marks on the current batch will be kept '
                'in the trainer\'s annotation store; the new batch starts fresh.)',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if choice != QMessageBox.StandardButton.Yes:
                return
        files = list(items_by_file.keys())
        self._tabs.setCurrentIndex(self._core_idx)
        self._tabs.setTabText(self._core_idx, 'WARP CORE — Fast Correction')
        try:
            self._core_win.set_fast_correction_mode(files, items_by_file, stype_by_file)
        except Exception as e:
            log.warning(
                f'Launcher: set_fast_correction_mode({len(files)} file(s)) failed: {e}')
            # Restore tab title on failure so the user isn't stranded.
            self._tabs.setTabText(self._core_idx, self._core_tab_default_title)

    def _on_fast_correction_exited(self):
        self._tabs.setTabText(self._core_idx, self._core_tab_default_title)
        self._tabs.setCurrentIndex(self._warp_idx)

    def _on_send_to_warp(self, result: object):
        try:
            self._warp_win.set_external_result(result)
        except Exception as e:
            log.warning(f'Launcher: set_external_result failed: {e}')
            return
        # Send to WARP is the terminal action of the Fast Correction loop —
        # close the loop by exiting Fast Mode so the trainer tab reverts to
        # its standing training view. Safe no-op when WARP CORE is already
        # in normal training mode. `fast_correction_exited` then drives the
        # tab title restore + tab switch via `_on_fast_correction_exited`;
        # the explicit switch below covers the non-FC path.
        try:
            self._core_win.exit_fast_correction_mode()
        except Exception as e:
            log.debug(f'Launcher: exit_fast_correction_mode failed: {e}')
        self._tabs.setCurrentIndex(self._warp_idx)

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
    # Apply the active theme (warp.themes.get_active) before any window
    # is built — Qt cascades the QApplication-level palette + stylesheet
    # to every subsequently constructed widget, so WARP and WARP CORE
    # share the same look without each having to call apply_dark_style.
    from warp.style import apply_dark_style
    apply_dark_style(app)
    win = LauncherWindow()
    win.show()
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
