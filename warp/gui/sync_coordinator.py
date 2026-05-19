"""App-level sync orchestration for the sto-warp launcher.

Owns four independent sync paths:

  1. AssetSyncManager — GitHub-backed binary assets (item icons,
     ship images) mirrored from STOCD/SETS-Data. Runs first so the
     icon matcher's template index sees populated files on cold start.
  2. WARPSyncClient   — community pHash knowledge.json download.
  3. ModelUpdater     — central EfficientNet / ArcFace model refresh.
  4. SyncManager      — confirmed-crop upload to HuggingFace.

Goals:
  - Run all three at app startup (after the UI is shown).
  - Run them again every `PERIOD_MIN` minutes.
  - Expose a single "Refresh" entry point for the launcher toolbar.
  - Guarantee at most one refresh runs at a time. The button is disabled
    via `busy_changed` while the worker thread is alive.

The worker calls the underlying clients' private synchronous methods
(`_download_knowledge_bg`, `_bg_check`) so the busy flag actually
reflects work-in-progress instead of just queuing daemons.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QThread, QTimer, Signal

from warp.debug import syslog as log

PERIOD_MIN = 5  # match legacy WARP CORE behaviour


class _RefreshWorker(QThread):
    """Runs one full sync cycle: assets → knowledge → model → upload.

    Each step is wrapped in try/except so a network failure in one path
    never aborts the others. Errors are logged, not propagated."""

    step = Signal(str)

    def __init__(self, sync_client, sets_root: Path, sync_manager, force: bool):
        super().__init__()
        self._sync_client  = sync_client
        self._sets_root    = sets_root
        self._sync_manager = sync_manager
        self._force        = force

    def run(self):
        log.info(f'SyncCoordinator: cycle start (force={self._force})')

        self.step.emit('assets')
        log.info('SyncCoordinator: step=assets — GitHub icon/ship asset mirror')
        try:
            from warp.data.asset_sync import AssetSyncManager
            AssetSyncManager().run()
        except Exception as e:
            log.warning(f'SyncCoordinator: asset sync failed: {e}')

        self.step.emit('knowledge')
        log.info('SyncCoordinator: step=knowledge — community pHash download')
        try:
            if self._sync_client is not None:
                self._sync_client._download_knowledge_bg(force=self._force)
        except Exception as e:
            log.warning(f'SyncCoordinator: knowledge refresh failed: {e}')

        self.step.emit('model')
        log.info('SyncCoordinator: step=model — central model version check')
        try:
            from warp.trainer.model_updater import ModelUpdater
            updater = ModelUpdater()
            updater._bg_check(on_updated=None)
        except Exception as e:
            log.warning(f'SyncCoordinator: model update check failed: {e}')

        self.step.emit('upload')
        log.info('SyncCoordinator: step=upload — confirmed-crop HuggingFace upload')
        try:
            if self._sync_manager is not None:
                self._sync_manager.check_and_upload()
                worker = getattr(self._sync_manager, '_worker', None)
                if worker is not None and worker.isRunning():
                    worker.wait()
        except Exception as e:
            log.warning(f'SyncCoordinator: crop upload failed: {e}')

        self.step.emit('done')
        log.info('SyncCoordinator: cycle done')


class SyncCoordinator(QObject):
    busy_changed = Signal(bool)
    status       = Signal(str)

    def __init__(self, sets_app, sets_root: Path, parent=None):
        super().__init__(parent)
        self._sets_app  = sets_app
        self._sets_root = sets_root
        self._worker: _RefreshWorker | None = None

        # WARPSyncClient kicks off a background knowledge download in its
        # constructor — that's fine, we want it warm.
        from warp.knowledge.sync_client import WARPSyncClient
        try:
            self.sync_client = WARPSyncClient()
        except Exception as e:
            log.warning(f'SyncCoordinator: WARPSyncClient init failed: {e}')
            self.sync_client = None

        from warp.trainer.sync import SyncManager
        try:
            self.sync_manager = SyncManager(sets_app)
        except Exception as e:
            log.warning(f'SyncCoordinator: SyncManager init failed: {e}')
            self.sync_manager = None

        self._timer = QTimer(self)
        self._timer.setInterval(PERIOD_MIN * 60 * 1000)
        self._timer.timeout.connect(self._periodic_tick)

    # ── lifecycle ──────────────────────────────────────────────────────

    def start(self):
        """Kick off the initial sync cycle and arm the periodic timer."""
        self.request_refresh(force=False)
        self._timer.start()

    def stop(self):
        self._timer.stop()
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait(5000)

    # ── public API ─────────────────────────────────────────────────────

    @property
    def is_busy(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    def request_refresh(self, force: bool = True):
        """Start a manual or scheduled refresh. No-op if already running."""
        if self.is_busy:
            self.status.emit('Refresh already in progress…')
            return
        self._worker = _RefreshWorker(
            self.sync_client, self._sets_root, self.sync_manager, force=force,
        )
        self._worker.step.connect(self._on_step)
        self._worker.finished.connect(self._on_finished)
        self.busy_changed.emit(True)
        self.status.emit('Syncing…')
        self._worker.start()

    # ── internals ──────────────────────────────────────────────────────

    def _periodic_tick(self):
        self.request_refresh(force=False)

    def _on_step(self, step: str):
        labels = {
            'assets':    'Syncing icons and ship images…',
            'knowledge': 'Refreshing community knowledge…',
            'model':     'Checking for newer model…',
            'upload':    'Uploading confirmed crops…',
            'done':      'Sync complete.',
        }
        self.status.emit(labels.get(step, step))

    def _on_finished(self):
        self._worker = None
        self.busy_changed.emit(False)
