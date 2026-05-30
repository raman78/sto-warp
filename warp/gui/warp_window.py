"""Standalone WARP recognition window.

Lightweight QMainWindow used as the entry point for `sto-warp gui`.
Replaces the SETS-coupled `warp/warp_dialog.py` — does not write into a
SETS build; instead it surfaces the `ImportResult` for inspection and
SETS-compatible export.

Flow:
  1. User picks one or more screenshots (or a folder).
  2. `RecognitionWorker` runs `WarpImporter.process_folder` off the UI
     thread; progress is forwarded to the status bar.
  3. Result populates the slot tree (one top-level row per slot, child
     rows per slot_index).
  4. Export to SETS JSON writes a SETS v3.0.0-compatible build file.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from PySide6.QtCore import QObject, QSettings, QThread, Signal
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QLabel,
    QMainWindow, QMessageBox, QPushButton, QTabWidget,
    QToolBar, QWidget, QVBoxLayout,
)

from warp.gui.log_view import LogViewWidget
from warp.gui.progress_bar import StatusProgressBar
from warp.gui.results_view import ResultsView
from warp.style import primary_btn_style, secondary_btn_style

from warp.debug import log
from warp.warp_importer import (
    SCREENSHOT_EXTENSIONS, ImportResult, WarpImporter,
)

_SETTINGS_LAST_FILES_DIR  = 'warp/last_files_dir'
_SETTINGS_LAST_FOLDER_DIR = 'warp/last_folder_dir'
_SETTINGS_FORCE_BT_ON     = 'warp/force_build_type_on'
_SETTINGS_FORCE_BT_VALUE  = 'warp/force_build_type_value'


def _restore_dir(key: str) -> str:
    s = QSettings()
    v = s.value(key)
    if isinstance(v, str) and v and Path(v).is_dir():
        return v
    return str(Path.home())


def _remember_dir(key: str, path: Path):
    QSettings().setValue(key, str(path))


BUILD_TYPES = (
    'SPACE_MIXED',
    'GROUND_MIXED',
    'SPACE',
    'GROUND',
    'BOFFS',
    'SPACE_BOFFS',
    'GROUND_BOFFS',
    'SPACE_TRAITS',
    'GROUND_TRAITS',
    'SPEC',
)


class RecognitionWorker(QObject):
    """Runs `WarpImporter.process_folder` on a worker thread.

    The importer's per-image callback is wired into `progress` for the
    status bar. Final `ImportResult` is delivered via `finished`; any
    fatal exception via `failed`.
    """

    progress = Signal(int, int, str)   # done, total, current file name
    stage    = Signal(int, str)        # sub-stage pct (0-100) + label
    finished = Signal(object)          # ImportResult
    failed   = Signal(str)

    def __init__(self, folder: Path, build_type: str,
                 overrides: dict[str, str] | None = None):
        super().__init__()
        self._folder     = folder
        self._build_type = build_type
        self._overrides  = dict(overrides or {})
        self._cancelled  = False

    def cancel(self):
        """Mark the worker for cooperative cancellation. The next progress
        callback the importer fires will raise InterruptedError which run()
        translates into a `failed('Cancelled')` signal."""
        self._cancelled = True

    def _stage_cb(self, pct: int, label: str) -> None:
        if self._cancelled:
            raise InterruptedError('cancelled')
        self.stage.emit(pct, label)

    def run(self):
        from warp.debug import use_detection_channel
        # Pin this worker's detection writes to the WARP-side channel so
        # WARP CORE's Detection Logs tab stays clean.
        with use_detection_channel('detection'):
            try:
                importer = WarpImporter(
                    build_type=self._build_type,
                    progress_callback=self._stage_cb,
                    per_file_overrides=self._overrides or None,
                )
                result = importer.process_folder(
                    self._folder,
                    progress_cb=lambda done, total, name: self.progress.emit(done, total, name),
                )
            except InterruptedError:
                log.info('RecognitionWorker: cancelled by user')
                self.failed.emit('Cancelled')
                return
            except Exception as e:
                log.exception('RecognitionWorker failed')
                self.failed.emit(f'{type(e).__name__}: {e}')
                return
            self.finished.emit(result)


class WarpWindow(QMainWindow):
    # Emitted when the user kicks off a fresh detection run (single
    # screenshot or folder). Launcher uses it to wipe the live Detection
    # logs tab so each run starts on a clean slate.
    detection_started = Signal()

    # Emitted when the user picks "Open in WARP CORE" from a Results-row
    # context menu. Payload: (absolute screenshot path, list[RecognisedItem]).
    # The trainer uses the second arg to skip its own Auto-Detect and load
    # WARP's pending results directly for review/correction. Empty list ⇒
    # fall back to the trainer's normal (annotations-only) load path.
    open_in_warp_core = Signal(str, object)

    # Emitted when the user picks "Open in WARP Fast Correction Mode" —
    # batch handoff of every screenshot in the current result. Payload:
    # dict[resolved_abs_path → list[RecognisedItem]]. The trainer enters
    # Fast Correction Mode and loads them all at once.
    open_in_warp_fast_correction = Signal(dict)

    def __init__(self):
        super().__init__()
        self.setWindowTitle('sto-warp — Star Trek Online screenshot recognition')
        self.resize(1100, 700)

        self._result: ImportResult | None = None
        self._worker: RecognitionWorker | None = None
        self._thread: QThread | None = None
        # Launcher flips these via set_warp_core_handler / set_fast_correction_handler
        # when it wires the respective signals to the trainer. Standalone
        # `sto-warp gui` runs leave them False and the Results context menu
        # hides the corresponding handoff entries.
        self._has_warp_core_handler = False
        self._has_fast_correction_handler = False
        # Temp dir holds copies of single-file selections so we can reuse
        # the folder-oriented `process_folder` entry point cleanly.
        self._tmp_dir: tempfile.TemporaryDirectory | None = None
        # Last folder processed — remembered so the Preview "Rerun
        # Recognition" action can re-run against the same input with
        # per-file overrides applied.
        self._last_folder: Path | None = None

        self._setup_ui()

    # ── UI scaffolding ──────────────────────────────────────────────

    def _setup_ui(self):
        tb = QToolBar('Main', self)
        tb.setMovable(False)
        self.addToolBar(tb)

        self._open_files_btn = QPushButton('Open Screenshot…', self)
        self._open_files_btn.setStyleSheet(secondary_btn_style())
        self._open_files_btn.clicked.connect(self._on_open_files)
        tb.addWidget(self._open_files_btn)

        self._open_folder_btn = QPushButton('Open Folder…', self)
        self._open_folder_btn.setStyleSheet(secondary_btn_style())
        self._open_folder_btn.clicked.connect(self._on_open_folder)
        tb.addWidget(self._open_folder_btn)

        # Re-runs the last detection without re-opening the file picker.
        # Disabled until the user opens at least one screenshot/folder.
        self._rerun_btn = QPushButton('Auto-Detect Slots', self)
        self._rerun_btn.setStyleSheet(secondary_btn_style())
        self._rerun_btn.setToolTip(
            'Re-run detection on the most recently opened screenshot(s).')
        self._rerun_btn.setEnabled(False)
        self._rerun_btn.clicked.connect(self._on_rerun_detection)
        tb.addWidget(self._rerun_btn)

        tb.addSeparator()

        # Force build type: when unchecked, every image in the folder is
        # classified independently by ML+OCR (AUTO mode). When checked, the
        # combo's value is forced on every image — same behavior as before.
        s = QSettings()
        force_on    = s.value(_SETTINGS_FORCE_BT_ON, False, type=bool)
        force_value = s.value(_SETTINGS_FORCE_BT_VALUE, 'SPACE_MIXED', type=str)

        self._force_bt_check = QCheckBox(' Force build type: ', self)
        self._force_bt_check.setToolTip(
            'Unchecked (default): each screenshot in the folder is classified '
            'independently by ML+OCR.\n'
            'Checked: every screenshot is processed as the selected type.'
        )
        self._force_bt_check.setChecked(force_on)
        self._force_bt_check.toggled.connect(self._on_force_bt_toggled)
        tb.addWidget(self._force_bt_check)

        self._build_combo = QComboBox(self)
        self._build_combo.addItems(BUILD_TYPES)
        if force_value in BUILD_TYPES:
            self._build_combo.setCurrentText(force_value)
        self._build_combo.setEnabled(force_on)
        self._build_combo.currentTextChanged.connect(
            lambda v: QSettings().setValue(_SETTINGS_FORCE_BT_VALUE, v))
        tb.addWidget(self._build_combo)

        tb.addSeparator()
        self._export_sets_btn = QPushButton('Export to SETS JSON…', self)
        self._export_sets_btn.setStyleSheet(primary_btn_style())
        self._export_sets_btn.setToolTip(
            'SETS v3.0.0-compatible build JSON — loadable via SETS '
            'File → Load Build.'
        )
        self._export_sets_btn.clicked.connect(self._on_export_sets_json)
        self._export_sets_btn.setEnabled(False)
        tb.addWidget(self._export_sets_btn)

        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)

        # Ship banner — surfaces the three signals SETS needs (name, type,
        # tier) above everything else so the user can sanity-check the OCR
        # result at a glance before exporting. Kept permanently visible
        # (text cleared between runs) so the panel above the tabs always
        # reserves two text rows; otherwise toggling visibility makes the
        # tabs jump up and down between detections.
        self._ship_banner = QLabel('', central)
        f = self._ship_banner.font()
        f.setPointSizeF(f.pointSizeF() + 2.0)
        f.setBold(True)
        self._ship_banner.setFont(f)
        from PySide6.QtGui import QFontMetrics
        self._ship_banner.setMinimumHeight(QFontMetrics(f).height())
        layout.addWidget(self._ship_banner)

        self._summary_lbl = QLabel('Open a screenshot to begin.', central)
        self._summary_lbl.setMinimumHeight(
            QFontMetrics(self._summary_lbl.font()).height())
        layout.addWidget(self._summary_lbl)

        self._tabs = QTabWidget(central)

        self._results = ResultsView(self._tabs)
        self._results.rerun_requested.connect(self._on_rerun_requested)
        self._results.open_in_warp_core.connect(self.open_in_warp_core.emit)
        self._results.open_in_warp_fast_corr.connect(
            self.open_in_warp_fast_correction.emit)
        self._tabs.addTab(self._results, 'Results')

        # Detection logs are scoped to WARP's own runs — live-tails the
        # 'detection' channel and gets wiped at the start of every new run
        # (`detection_started` → clear_live) so the buffer reflects just
        # the current detection rather than accumulating across runs.
        self._log_view = LogViewWidget(channel='detection', parent=self._tabs)
        self._log_view.set_default_save_name_cb(self._suggest_log_save_stem)
        self._tabs.addTab(self._log_view, 'Detection Logs')
        self.detection_started.connect(self._log_view.clear_live)

        layout.addWidget(self._tabs, stretch=1)

        self.setCentralWidget(central)

        self._progress = StatusProgressBar(self)
        self._progress.cancel_requested.connect(self._on_cancel_requested)
        self.statusBar().addPermanentWidget(self._progress)
        self.statusBar().showMessage('Ready.')

    # ── File picking ────────────────────────────────────────────────

    def _on_open_files(self):
        from warp.folder_picker import pick_files
        filters = tuple(f'*{e}' for e in sorted(SCREENSHOT_EXTENSIONS))
        files = pick_files(
            self,
            title='Open screenshot',
            start_dir=_restore_dir(_SETTINGS_LAST_FILES_DIR),
            image_filters=filters,
            multi=False,
        )
        if not files:
            return
        _remember_dir(_SETTINGS_LAST_FILES_DIR, files[0].parent)
        # Stage selections into a temp dir so the importer's folder
        # pipeline picks them up unchanged. Retire any previous staged
        # dir first — _cleanup_thread no longer touches it so a Rerun
        # can find it alive.
        self._release_tmp_dir()
        self._tmp_dir = tempfile.TemporaryDirectory(prefix='warp-gui-')
        staged = Path(self._tmp_dir.name)
        for f in files:
            try:
                (staged / Path(f).name).symlink_to(Path(f).resolve())
            except OSError:
                # Symlinks not supported (e.g. Windows w/o privilege) →
                # fall back to copy.
                import shutil
                shutil.copy2(f, staged / Path(f).name)
        self._stage_folder(staged)

    def _on_open_folder(self):
        from warp.folder_picker import pick_folder
        folder = pick_folder(
            self,
            title='Open screenshot folder',
            start_dir=_restore_dir(_SETTINGS_LAST_FOLDER_DIR),
        )
        if folder is None:
            return
        _remember_dir(_SETTINGS_LAST_FOLDER_DIR, folder)
        # No staged tempdir for direct folder open — but if a previous
        # single-file run left one around, retire it now.
        self._release_tmp_dir()
        self._stage_folder(folder)

    def _stage_folder(self, folder: Path):
        """Remember the folder and prime the UI but don't run detection.

        WARP used to auto-fire a detection pass at the end of every Open
        Files / Open Folder. The toolbar's "Auto-Detect Slots" button
        now owns that step, so the user can preview which folder they
        picked (and adjust the Force-build-type combo) before paying
        for a full pipeline run.
        """
        self._last_folder = folder
        self._result = None
        self._results.clear()
        self._ship_banner.clear()
        self._export_sets_btn.setEnabled(False)
        n = sum(1 for p in folder.iterdir()
                if p.is_file() and p.suffix.lower() in SCREENSHOT_EXTENSIONS) \
            if folder.is_dir() else 0
        self._summary_lbl.setText(
            f'{n} screenshot{"" if n == 1 else "s"} loaded — '
            f'press "Auto-Detect Slots" to run recognition.')
        self.statusBar().showMessage(
            f'Loaded {folder} — press Auto-Detect Slots to start.')
        self._rerun_btn.setEnabled(True)

    # ── Pipeline plumbing ───────────────────────────────────────────

    def _run_against(self, folder: Path,
                     overrides: dict[str, str] | None = None,
                     focus_preview: bool = True):
        if self._thread is not None:
            QMessageBox.information(self, 'Busy',
                                    'A recognition run is already in progress.')
            return

        self.detection_started.emit()
        # Keep the previous Results visible during the run — the tree is
        # cleared and rebuilt by `_populate_tree` only when the new result
        # arrives. Mirrors the WARP CORE review-panel fix (commit 513d87a)
        # so the user doesn't see an empty pane flash mid-detection.
        # Preserve Results' per-file overrides across re-runs so the user
        # doesn't have to re-pick them; the dropdown state is what drives
        # the next Rerun. Only clear on a fresh folder open.
        if overrides is None:
            self._results.clear()
        elif focus_preview:
            self._tabs.setCurrentWidget(self._results)
        self._result = None
        self._ship_banner.clear()
        self._export_sets_btn.setEnabled(False)
        self._summary_lbl.clear()
        self._progress.start(determinate=True, maximum=100)
        self._set_controls_enabled(False)
        self._last_folder = folder

        self._thread = QThread(self)
        # AUTO mode = empty string; importer derives per-image build_type
        # from ML+OCR. Forced mode = combo's current text.
        forced_bt = (self._build_combo.currentText()
                     if self._force_bt_check.isChecked() else '')
        self._worker = RecognitionWorker(folder, forced_bt, overrides)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.stage.connect(self._on_stage)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_thread)
        self._thread.start()

    def _on_progress(self, done: int, total: int, name: str):
        # Per-image notification — only refresh the status text. The actual
        # progress bar is driven by the sub-stage signal below, which moves
        # smoothly through OCR / classify / layout / per-slot matching
        # within the per-image window so single-image runs don't jump 0→100%.
        self.statusBar().showMessage(f'[{done + 1}/{total}] {name}')

    def _on_stage(self, pct: int, label: str):
        self._progress.set_progress(pct)
        self.statusBar().showMessage(label)

    def _on_cancel_requested(self):
        if self._worker is not None:
            self._worker.cancel()
            self._progress.set_cancel_enabled(False)
            self.statusBar().showMessage('Cancelling…')

    def _on_finished(self, result: ImportResult):
        self._result = result
        self._results.set_result(result)
        self._progress.finish()
        self._set_ship_banner(result)
        msg = f'{len(result.items)} items recognised'
        if result.errors:
            msg += f'  ·  {len(result.errors)} error(s)'
        self._summary_lbl.setText(msg)
        self.statusBar().showMessage('Done.')
        self._export_sets_btn.setEnabled(True)
        self._set_controls_enabled(True)

    def set_external_result(self, result: ImportResult) -> None:
        """Install an ImportResult produced outside WARP (e.g. WARP CORE
        "Send to WARP" handoff) into the WARP UI.

        Mirrors `_on_finished` minus the progress/worker bookkeeping:
        populates the Results view, ship banner, summary, and enables the
        SETS-build export so the user can hit "Export" without re-running
        detection.
        """
        self._result = result
        self._results.set_result(result)
        self._set_ship_banner(result)
        msg = f'{len(result.items)} items recognised (from WARP CORE)'
        if result.errors:
            msg += f'  ·  {len(result.errors)} error(s)'
        self._summary_lbl.setText(msg)
        self.statusBar().showMessage('Loaded from WARP CORE.')
        self._export_sets_btn.setEnabled(True)
        self._set_controls_enabled(True)

    def _set_ship_banner(self, result: ImportResult):
        bits = []
        if result.ship_name:
            bits.append(result.ship_name)
        if result.ship_type:
            bits.append(result.ship_type)
        if result.ship_tier:
            bits.append(result.ship_tier)
        if bits:
            self._ship_banner.setText('   ·   '.join(bits))
            self._ship_banner.setVisible(True)
        else:
            self._ship_banner.setText('(no ship info recognised)')
            self._ship_banner.setVisible(True)

    def _on_failed(self, err: str):
        self._progress.finish()
        self._ship_banner.clear()
        if err == 'Cancelled':
            self._summary_lbl.setText('Recognition cancelled.')
            self.statusBar().showMessage('Cancelled.')
        else:
            self._summary_lbl.setText(f'Recognition failed: {err}')
            self.statusBar().showMessage('Failed.')
        self._set_controls_enabled(True)
        if err != 'Cancelled':
            QMessageBox.critical(self, 'Recognition failed', err)

    def _cleanup_thread(self):
        if self._thread is not None:
            self._thread.deleteLater()
        self._thread = None
        self._worker = None
        # Note: we do NOT clean up self._tmp_dir here. The Preview tab can
        # request a Rerun against the same staged folder; the tempdir is
        # retired the next time the user opens a new screenshot/folder
        # (see _release_tmp_dir).

    def _release_tmp_dir(self):
        if self._tmp_dir is not None:
            try:
                self._tmp_dir.cleanup()
            except OSError:
                pass
            self._tmp_dir = None

    def _on_rerun_requested(self, overrides: dict):
        # Preview asked for a re-run with per-file build_type overrides.
        # We point the importer at the same folder we used last time. For
        # single-file selections that folder is the staged tempdir, which
        # is still alive because we delay its cleanup until the next open
        # action (see _on_open_files).
        if not self._last_folder or not self._last_folder.is_dir():
            QMessageBox.warning(
                self, 'Cannot rerun',
                'The original screenshot folder is no longer available. '
                'Re-open the screenshots and try again.')
            return
        self._run_against(self._last_folder, overrides=overrides)

    def _on_rerun_detection(self):
        # Toolbar "Auto-Detect Slots" — re-run the last detection without
        # opening the picker again and without jumping the user to a
        # different tab. No per-file overrides; the AUTO mode /
        # force-build-type toggle still applies.
        if not self._last_folder or not self._last_folder.is_dir():
            QMessageBox.warning(
                self, 'Cannot rerun',
                'The original screenshot folder is no longer available. '
                'Re-open the screenshots and try again.')
            return
        self._run_against(self._last_folder, overrides={},
                          focus_preview=False)

    def _on_force_bt_toggled(self, checked: bool):
        self._build_combo.setEnabled(checked)
        QSettings().setValue(_SETTINGS_FORCE_BT_ON, checked)

    def _set_controls_enabled(self, enabled: bool):
        self._open_files_btn.setEnabled(enabled)
        self._open_folder_btn.setEnabled(enabled)
        # Rerun only makes sense once a folder has been opened at least
        # once — keep it gated on `_last_folder` even when controls are
        # re-enabled after a run completes.
        self._rerun_btn.setEnabled(
            enabled and self._last_folder is not None
            and self._last_folder.is_dir())
        self._force_bt_check.setEnabled(enabled)
        # The combo only takes input when the checkbox is on AND we're idle.
        self._build_combo.setEnabled(enabled and self._force_bt_check.isChecked())

    # ── Handler advertisement (launcher wires these on init) ────────

    def set_warp_core_handler(self, has: bool) -> None:
        self._has_warp_core_handler = bool(has)
        self._results.set_warp_core_handler(has)

    def set_fast_correction_handler(self, has: bool) -> None:
        self._has_fast_correction_handler = bool(has)
        self._results.set_fast_correction_handler(has)

    # ── Detection Logs save-name suggestion ─────────────────────────

    def _suggest_log_save_stem(self) -> str:
        """Default 'Save As' name for the Detection Logs tab.

        Uses the currently-displayed file in the Results view plus its
        detected build family (SPACE / GROUND) when known, suffixed with
        the date — e.g. `Screenshot_2024-01-15_120000_12345_space_20260521`.
        Falls back to a plain timestamped stem when nothing is loaded.
        """
        import datetime
        date = datetime.datetime.now().strftime('%Y%m%d')
        try:
            src = self._results.current_file()
            if src:
                stem = Path(src).stem
                bt = (self._results.current_build_type() or '').upper()
                fam = ''
                if bt.startswith('SPACE'):
                    fam = '_space'
                elif bt.startswith('GROUND'):
                    fam = '_ground'
                return f'{stem}{fam}_{date}'
        except Exception:
            pass
        return f'detection_{date}'

    # ── Export ──────────────────────────────────────────────────────

    def _on_export_sets_json(self):
        if self._result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, 'Export SETS build JSON',
            str(Path.home() / 'warp_sets_build.json'),
            'JSON (*.json);;All files (*)',
        )
        if not path:
            return
        try:
            from warp.build_writer import build_from_result
            from warp.sets_export import write_sets_build
            from warp.data.cargo import cache_view
            cache = cache_view()
            build, report = build_from_result(self._result, cache=cache)
            write_sets_build(build, path, cache=cache)
        except Exception as e:
            log.exception('SETS export failed')
            QMessageBox.critical(self, 'Export failed', f'{type(e).__name__}: {e}')
            return
        msg = (f'SETS build → {path}  ·  '
               f'ship={report.ship or "—"}  '
               f'eq={report.n_equipment}  traits={report.n_traits}  '
               f'boff_ab={report.n_boff_abilities}')
        if report.unmatched_items:
            msg += f'  ·  {report.unmatched_items} unmatched'
        self.statusBar().showMessage(msg)


def main(argv: list[str] | None = None) -> int:
    app = QApplication.instance() or QApplication(argv or sys.argv)
    win = WarpWindow()
    win.show()
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
