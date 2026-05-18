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

from PySide6.QtCore import QObject, QSettings, QThread, Qt, Signal
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QHeaderView, QLabel, QMainWindow,
    QMessageBox, QProgressBar, QPushButton, QTabWidget, QToolBar,
    QTreeWidget, QTreeWidgetItem, QWidget, QVBoxLayout,
)

from warp.gui.preview_view import PreviewView

from warp.debug import log
from warp.warp_importer import (
    SCREENSHOT_EXTENSIONS, SLOT_ORDER, ImportResult, RecognisedItem,
    WarpImporter,
)

_SETTINGS_LAST_FILES_DIR  = 'warp/last_files_dir'
_SETTINGS_LAST_FOLDER_DIR = 'warp/last_folder_dir'


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

    def __init__(self, folder: Path, build_type: str):
        super().__init__()
        self._folder     = folder
        self._build_type = build_type

    def run(self):
        try:
            importer = WarpImporter(
                build_type=self._build_type,
                progress_callback=lambda pct, label: self.stage.emit(pct, label),
            )
            result = importer.process_folder(
                self._folder,
                progress_cb=lambda done, total, name: self.progress.emit(done, total, name),
            )
        except Exception as e:
            log.exception('RecognitionWorker failed')
            self.failed.emit(f'{type(e).__name__}: {e}')
            return
        self.finished.emit(result)


class WarpWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('sto-warp — Star Trek Online screenshot recognition')
        self.resize(1100, 700)

        self._result: ImportResult | None = None
        self._worker: RecognitionWorker | None = None
        self._thread: QThread | None = None
        # Temp dir holds copies of single-file selections so we can reuse
        # the folder-oriented `process_folder` entry point cleanly.
        self._tmp_dir: tempfile.TemporaryDirectory | None = None

        self._setup_ui()

    # ── UI scaffolding ──────────────────────────────────────────────

    def _setup_ui(self):
        tb = QToolBar('Main', self)
        tb.setMovable(False)
        self.addToolBar(tb)

        self._open_files_btn = QPushButton('Open Screenshot(s)…', self)
        self._open_files_btn.clicked.connect(self._on_open_files)
        tb.addWidget(self._open_files_btn)

        self._open_folder_btn = QPushButton('Open Folder…', self)
        self._open_folder_btn.clicked.connect(self._on_open_folder)
        tb.addWidget(self._open_folder_btn)

        tb.addSeparator()
        tb.addWidget(QLabel(' Build type: ', self))

        self._build_combo = QComboBox(self)
        self._build_combo.addItems(BUILD_TYPES)
        self._build_combo.setCurrentText('SPACE_MIXED')
        tb.addWidget(self._build_combo)

        tb.addSeparator()
        self._export_sets_btn = QPushButton('Export to SETS JSON…', self)
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
        # result at a glance before exporting.
        self._ship_banner = QLabel('', central)
        f = self._ship_banner.font()
        f.setPointSizeF(f.pointSizeF() + 2.0)
        f.setBold(True)
        self._ship_banner.setFont(f)
        self._ship_banner.setVisible(False)
        layout.addWidget(self._ship_banner)

        self._summary_lbl = QLabel('Open a screenshot to begin.', central)
        layout.addWidget(self._summary_lbl)

        self._tabs = QTabWidget(central)

        self._tree = QTreeWidget(self._tabs)
        self._tree.setColumnCount(5)
        self._tree.setHeaderLabels(['Slot', 'Idx', 'Item', 'Conf', 'Source'])
        self._tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        self._tree.header().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self._tree.setAlternatingRowColors(True)
        self._tree.setRootIsDecorated(True)
        self._tree.setUniformRowHeights(True)
        self._tree.setColumnWidth(0, 220)
        self._tree.setColumnWidth(1, 50)
        self._tree.setColumnWidth(3, 70)
        self._tabs.addTab(self._tree, 'Results')

        self._preview = PreviewView(self._tabs)
        self._tabs.addTab(self._preview, 'Preview')

        layout.addWidget(self._tabs, stretch=1)

        self.setCentralWidget(central)

        self._progress = QProgressBar(self)
        self._progress.setRange(0, 100)
        self._progress.setVisible(False)
        self.statusBar().addPermanentWidget(self._progress)
        self.statusBar().showMessage('Ready.')

    # ── File picking ────────────────────────────────────────────────

    def _on_open_files(self):
        from warp.folder_picker import pick_files
        filters = tuple(f'*{e}' for e in sorted(SCREENSHOT_EXTENSIONS))
        files = pick_files(
            self,
            title='Open screenshot(s)',
            start_dir=_restore_dir(_SETTINGS_LAST_FILES_DIR),
            image_filters=filters,
        )
        if not files:
            return
        _remember_dir(_SETTINGS_LAST_FILES_DIR, files[0].parent)
        # Stage selections into a temp dir so the importer's folder
        # pipeline picks them up unchanged.
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
        self._run_against(staged)

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
        self._run_against(folder)

    # ── Pipeline plumbing ───────────────────────────────────────────

    def _run_against(self, folder: Path):
        if self._thread is not None:
            QMessageBox.information(self, 'Busy',
                                    'A recognition run is already in progress.')
            return

        self._tree.clear()
        self._preview.clear()
        self._result = None
        self._ship_banner.setVisible(False)
        self._export_sets_btn.setEnabled(False)
        self._summary_lbl.setText(f'Recognising {folder}…')
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._set_controls_enabled(False)

        self._thread = QThread(self)
        self._worker = RecognitionWorker(folder, self._build_combo.currentText())
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
        self._progress.setValue(max(0, min(100, pct)))
        self.statusBar().showMessage(label)

    def _on_finished(self, result: ImportResult):
        self._result = result
        self._populate_tree(result)
        self._preview.set_result(result)
        self._progress.setValue(100)
        self._progress.setVisible(False)
        self._set_ship_banner(result)
        msg = f'{len(result.items)} items recognised'
        if result.errors:
            msg += f'  ·  {len(result.errors)} error(s)'
        self._summary_lbl.setText(msg)
        self.statusBar().showMessage('Done.')
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
        self._progress.setVisible(False)
        self._ship_banner.setVisible(False)
        self._summary_lbl.setText(f'Recognition failed: {err}')
        self.statusBar().showMessage('Failed.')
        self._set_controls_enabled(True)
        QMessageBox.critical(self, 'Recognition failed', err)

    def _cleanup_thread(self):
        if self._thread is not None:
            self._thread.deleteLater()
        self._thread = None
        self._worker = None
        if self._tmp_dir is not None:
            self._tmp_dir.cleanup()
            self._tmp_dir = None

    def _set_controls_enabled(self, enabled: bool):
        self._open_files_btn.setEnabled(enabled)
        self._open_folder_btn.setEnabled(enabled)
        self._build_combo.setEnabled(enabled)

    # ── Result rendering ────────────────────────────────────────────

    def _populate_tree(self, result: ImportResult):
        by_slot: dict[str, list[RecognisedItem]] = {}
        for it in result.items:
            by_slot.setdefault(it.slot, []).append(it)

        # Display order: ship metadata first (the three OCR signals SETS
        # actually needs), then the canonical pipeline order from SLOT_ORDER
        # (equipment → BOFFs → traits → spec). Anything left over — BOFF
        # seat keys like `Boff Seat L[T]_392`, ad-hoc labels — is appended
        # sorted so the output stays deterministic.
        meta_slots = ['Ship Name', 'Ship Type', 'Ship Tier']
        canonical = [sd['name'] for sd in SLOT_ORDER.get(result.build_type, [])]
        seen: set[str] = set()
        ordered_slots: list[str] = []
        for s in meta_slots + canonical:
            if s in by_slot and s not in seen:
                ordered_slots.append(s)
                seen.add(s)
        for s in sorted(by_slot):
            if s not in seen:
                ordered_slots.append(s)
                seen.add(s)

        for slot in ordered_slots:
            entries = sorted(by_slot[slot], key=lambda it: (it.slot_index, it.name))
            parent = QTreeWidgetItem(self._tree)
            parent.setText(0, slot)
            parent.setText(1, str(len(entries)))
            parent.setFirstColumnSpanned(False)
            for it in entries:
                child = QTreeWidgetItem(parent)
                child.setText(0, '')
                child.setText(1, str(it.slot_index + 1))
                child.setText(2, it.name or '—')
                child.setText(3, f'{it.confidence:.2f}')
                child.setText(4, Path(it.source_file).name if it.source_file else '')
            parent.setExpanded(True)

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
