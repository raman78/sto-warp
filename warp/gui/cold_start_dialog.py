"""Blocking splash for the first-run download.

Shown by `maybe_run_cold_start()` between `QApplication()` and
`LauncherWindow().show()`. Runs `SyncCoordinator`'s full cycle in the
foreground so nothing slow leaks into the background after the
launcher opens:

  1. CARGO     — equipment/ship/trait JSONs from GitHub raw  (~2 s)
  2. Assets    — item icons + ship images mirror             (the long one)
  3. Knowledge — community pHash overrides                   (~1 s)
  4. Model     — central model version check                 (~1 s)
  5. Crops     — community icon library (tarball or snapshot)
  6. Seed      — icon matcher template index from crops      (~5 s)
  7. Equiv     — admin-curated icon equivalence classes      (~1 s)

Detection (`is_cold_start()`): the marker file at
`config_dir()/startup_sync_done` is missing. The marker is written
only after every phase has completed in a single run, so an
interrupted or cancelled splash leaves the marker absent and the
splash reappears on the next launch. Once written it stays written —
subsequent launches skip the splash entirely and let `SyncCoordinator`
keep mirrors fresh in the background.

Buttons:
  - Close   → `QApplication.quit()` (clean exit, nothing started yet)
  - Cancel  → warn + dismiss; LauncherWindow starts in degraded mode
              and the splash reappears on the next launch.
"""
from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QObject, QThread, Qt, Signal
from PySide6.QtWidgets import (
    QApplication, QDialog, QHBoxLayout, QLabel, QMessageBox, QProgressBar,
    QPushButton, QVBoxLayout, QWidget,
)

from warp.debug import syslog as log
from warp.userdata import config_dir


_MARKER_FILENAME = 'startup_sync_done'


def _marker_path():
    return config_dir() / _MARKER_FILENAME


# ── Cold-start detection ──────────────────────────────────────────────────

def is_cold_start() -> bool:
    """True until the splash has run every phase to completion once.

    Marker-file driven: presence of `startup_sync_done` in `config_dir()`
    means a previous launch finished the full cycle without being
    cancelled. A half-populated mirror (interrupted download, kill -9)
    leaves the marker absent, so the splash reappears next launch
    instead of letting the background sync resume invisibly.
    """
    return not _marker_path().exists()


# ── Worker ────────────────────────────────────────────────────────────────

class _ColdStartWorker(QThread):
    """Runs the four phases sequentially, emitting signals for the UI.

    Cross-thread emit lands on the dialog via Qt.QueuedConnection by
    default (auto-selected when sender/receiver live on different
    threads). The worker checks `_cancelled` between phases — a Cancel
    click can't interrupt huggingface_hub mid-download, but it'll stop
    the next phase from starting.
    """

    phase_started  = Signal(str)              # phase id
    phase_progress = Signal(str, int, int)    # phase id, done, total (0/0 = indeterminate)
    phase_done     = Signal(str)              # phase id
    phase_failed   = Signal(str, str)         # phase id, error message
    all_done       = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._cancelled = False

    def cancel(self) -> None:
        """Request graceful stop. Cannot interrupt an in-flight download
        (huggingface_hub doesn't expose a stop signal); checked between
        phases."""
        self._cancelled = True

    def run(self) -> None:
        steps: list[tuple[str, Callable[[], None]]] = [
            ('cargo',     self._do_cargo),
            ('assets',    self._do_assets),
            ('knowledge', self._do_knowledge),
            ('model',     self._do_model),
            ('crops',     self._do_crops),
            ('seed',      self._do_seed),
            ('equiv',     self._do_equiv),
        ]
        for phase, fn in steps:
            if self._cancelled:
                log.info(f'cold-start: cancelled before phase={phase}')
                return
            self.phase_started.emit(phase)
            try:
                fn()
                self.phase_done.emit(phase)
            except Exception as e:
                log.warning(f'cold-start: phase={phase} failed: {e}')
                self.phase_failed.emit(phase, str(e))
                # Continue: a knowledge.json 503 must not block crops.
        self.all_done.emit()

    # ── Phase implementations ────────────────────────────────────────────

    def _do_cargo(self) -> None:
        from warp.data import cargo
        # Indeterminate — refresh_all is fast (~2 s for 5 files) and
        # doesn't expose per-file callbacks. Worth showing the phase even
        # though there's no % to display.
        self.phase_progress.emit('cargo', 0, 0)
        cargo.refresh_all(force=False)

    def _do_assets(self) -> None:
        from warp.data.asset_sync import AssetSyncManager

        # AssetSync's on_progress fires once per file across two groups
        # (item icons, ship images). We surface raw counts — totals jump
        # at group boundaries, which is fine because the user mostly
        # cares about seeing forward motion.
        def cb(label: str, current: int, total: int) -> None:
            if total > 0:
                self.phase_progress.emit('assets', current, total)
            else:
                self.phase_progress.emit('assets', 0, 0)

        self.phase_progress.emit('assets', 0, 0)
        AssetSyncManager().run(on_progress=cb)

    def _do_knowledge(self) -> None:
        from warp.knowledge.sync_client import WARPSyncClient
        self.phase_progress.emit('knowledge', 0, 0)
        client = WARPSyncClient()
        # Block on the synchronous worker (constructor spawns background
        # threads; we explicitly call the worker to wait for completion).
        client._download_knowledge_bg(force=False)

    def _do_model(self) -> None:
        from warp.trainer.model_updater import ModelUpdater
        self.phase_progress.emit('model', 0, 0)
        ModelUpdater()._bg_check(on_updated=None)

    def _do_crops(self) -> None:
        from warp.knowledge.community_crops import CommunityCropsClient

        def cb(done: int, total: int, _label: str) -> None:
            self.phase_progress.emit('crops', done, total)

        # Initial signal so the bar appears in "preparing" state during
        # the ~23 s huggingface_hub first-import + dataset_info() probe,
        # before any byte-level progress kicks in.
        self.phase_progress.emit('crops', 0, 0)
        CommunityCropsClient().fetch(progress_cb=cb)

    def _do_seed(self) -> None:
        from warp.recognition.icon_matcher import SETSIconMatcher
        self.phase_progress.emit('seed', 0, 0)
        SETSIconMatcher.seed_from_community_crops()

    def _do_equiv(self) -> None:
        from warp.knowledge.sync_client import WARPSyncClient
        self.phase_progress.emit('equiv', 0, 0)
        WARPSyncClient()._download_icon_equivalence_bg(force=False)


# ── Dialog ────────────────────────────────────────────────────────────────

_PHASE_LABELS = {
    'cargo':     'CARGO data',
    'assets':    'Item & ship icons',
    'knowledge': 'Community knowledge',
    'model':     'Recognition model',
    'crops':     'Community icon library',
    'seed':      'Matcher template index',
    'equiv':     'Icon equivalence',
}


class _PhaseRow(QWidget):
    """One row: label + status + progress bar. Mutually exclusive state
    via setStatus('pending'|'active'|'done'|'failed')."""

    def __init__(self, phase_id: str, label: str, parent: QWidget | None = None):
        super().__init__(parent)
        self._phase_id = phase_id

        self._label = QLabel(label, self)
        self._label.setMinimumWidth(180)
        self._status = QLabel('— waiting', self)
        self._status.setMinimumWidth(120)
        self._bar = QProgressBar(self)
        self._bar.setRange(0, 0)   # indeterminate by default
        self._bar.setVisible(False)
        self._bar.setTextVisible(True)
        self._bar.setMinimumWidth(280)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._label)
        lay.addWidget(self._status)
        lay.addWidget(self._bar, 1)

        self.setStatus('pending')

    def setStatus(self, kind: str, detail: str = '') -> None:
        if kind == 'pending':
            self._status.setText('— waiting')
            self._bar.setVisible(False)
        elif kind == 'active':
            self._status.setText('downloading…')
            self._bar.setVisible(True)
        elif kind == 'done':
            self._status.setText('✓ done')
            self._bar.setVisible(False)
        elif kind == 'failed':
            self._status.setText(f'✗ {detail[:40]}' if detail else '✗ failed')
            self._bar.setVisible(False)

    def setProgress(self, done: int, total: int) -> None:
        if total > 0:
            self._bar.setRange(0, total)
            self._bar.setValue(done)
            pct = 100 * done // total
            # Heuristic: crops streams a single ~hundreds-of-MB tarball so
            # totals are byte-sized; everything else (assets) counts files.
            if total >= 1_000_000:
                self._bar.setFormat(
                    f'{done/1e6:.1f} / {total/1e6:.1f} MB ({pct}%)')
            else:
                self._bar.setFormat(f'{done} / {total} files ({pct}%)')
        else:
            self._bar.setRange(0, 0)
            self._bar.setFormat('')


class ColdStartDialog(QDialog):
    """Modal splash blocking LauncherWindow until cold-start downloads
    finish or the user opts out.

    Two exit paths:
      - accept()  → finished successfully; caller proceeds normally
      - reject()  → user opted out (Cancel) or closed window (Close);
                    caller decides based on the `closed_via_quit` flag
                    whether to quit the app or proceed degraded.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle('sto-warp — first-run setup')
        self.setModal(True)
        self.setMinimumWidth(640)
        # Strip the system close button; we want users to use Close or
        # Cancel explicitly so the intent is recorded (quit vs. degrade).
        self.setWindowFlags(
            self.windowFlags() & ~Qt.WindowType.WindowCloseButtonHint
        )

        try:
            from warp.style import apply_dark_style
            apply_dark_style(self)
        except Exception:
            pass

        # `closed_via_quit` lets `maybe_run_cold_start()` distinguish
        # "user pressed Close → kill the app" from "user pressed Cancel →
        # start LauncherWindow degraded".
        self.closed_via_quit = False
        # Set by _on_all_done — used by main() to skip the launcher's
        # immediate SyncCoordinator cycle (we just did it all).
        self.completed_cleanly = False

        intro = QLabel(
            'First run: downloading reference data needed by the recognizer.\n'
            'This takes several minutes once; subsequent launches reuse the cache.',
            self,
        )
        intro.setWordWrap(True)

        self._rows: dict[str, _PhaseRow] = {}
        rows_box = QVBoxLayout()
        for phase_id, label in _PHASE_LABELS.items():
            row = _PhaseRow(phase_id, label, self)
            self._rows[phase_id] = row
            rows_box.addWidget(row)

        self._close_btn  = QPushButton('Close (exit)', self)
        self._cancel_btn = QPushButton('Cancel (start without full data)', self)
        self._close_btn.clicked.connect(self._on_close)
        self._cancel_btn.clicked.connect(self._on_cancel)

        btns = QHBoxLayout()
        btns.addStretch(1)
        btns.addWidget(self._cancel_btn)
        btns.addWidget(self._close_btn)

        outer = QVBoxLayout(self)
        outer.addWidget(intro)
        outer.addLayout(rows_box)
        outer.addStretch(1)
        outer.addLayout(btns)

        self._worker = _ColdStartWorker()
        self._worker.phase_started.connect(self._on_phase_started)
        self._worker.phase_progress.connect(self._on_phase_progress)
        self._worker.phase_done.connect(self._on_phase_done)
        self._worker.phase_failed.connect(self._on_phase_failed)
        self._worker.all_done.connect(self._on_all_done)

    # ── Lifecycle ────────────────────────────────────────────────────────

    def showEvent(self, event) -> None:  # noqa: N802 (Qt name)
        super().showEvent(event)
        if not self._worker.isRunning():
            self._worker.start()

    def closeEvent(self, event) -> None:  # noqa: N802
        # Window-system close (Alt-F4 etc.) treated as Close (exit).
        self._on_close()
        event.ignore()

    # ── Worker signal handlers ───────────────────────────────────────────

    def _on_phase_started(self, phase: str) -> None:
        row = self._rows.get(phase)
        if row:
            row.setStatus('active')

    def _on_phase_progress(self, phase: str, done: int, total: int) -> None:
        row = self._rows.get(phase)
        if row:
            row.setProgress(done, total)

    def _on_phase_done(self, phase: str) -> None:
        row = self._rows.get(phase)
        if row:
            row.setStatus('done')

    def _on_phase_failed(self, phase: str, err: str) -> None:
        row = self._rows.get(phase)
        if row:
            row.setStatus('failed', err)

    def _on_all_done(self) -> None:
        self.completed_cleanly = True
        # Persist the marker only after every phase ran — a cancelled
        # or partial cycle leaves no marker so we re-prompt next launch.
        try:
            _marker_path().write_text('')
        except OSError as e:
            log.warning(f'cold-start: failed to write completion marker: {e}')
        self.accept()

    # ── Button handlers ──────────────────────────────────────────────────

    def _on_close(self) -> None:
        self.closed_via_quit = True
        self._worker.cancel()
        # Don't wait — the worker holds the GIL during downloads; let the
        # interpreter reap the daemon thread on exit.
        self.reject()

    def _on_cancel(self) -> None:
        resp = QMessageBox.warning(
            self,
            'Continue without full data?',
            'Recognition quality will be reduced this session — the community '
            'icon library is not yet downloaded.\n\n'
            'The splash will reappear on next launch so the download can finish.\n\n'
            'Continue anyway?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if resp == QMessageBox.StandardButton.Yes:
            self._worker.cancel()
            self.reject()


# ── Top-level entry ───────────────────────────────────────────────────────

def maybe_run_cold_start() -> tuple[bool, bool]:
    """Show the splash if cold start is detected. Blocks until dismissal.

    Returns:
        (should_launch, skip_initial_sync)

        should_launch     — False when the user pressed Close (caller
                            should not open LauncherWindow); True
                            otherwise.
        skip_initial_sync — True when the splash ran every phase to
                            completion, so the launcher's
                            SyncCoordinator should arm its periodic
                            timer without re-running the cycle that
                            just finished. False for warm starts
                            (nothing happened — let the background
                            timer handle refresh on schedule) and for
                            cancelled splashes.
    """
    if not is_cold_start():
        return (True, False)

    log.info('cold-start: completion marker missing — showing splash')
    dialog = ColdStartDialog()
    dialog.exec()

    if dialog.closed_via_quit:
        log.info('cold-start: user pressed Close — quitting app')
        app = QApplication.instance()
        if app is not None:
            app.quit()
        return (False, False)

    if dialog.completed_cleanly:
        log.info('cold-start: splash finished all phases — '
                 'skipping launcher initial sync cycle')
        return (True, True)

    log.info('cold-start: splash dismissed early — proceeding degraded')
    return (True, False)
