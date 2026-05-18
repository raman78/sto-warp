"""Standalone WARP recognition window.

Lightweight QMainWindow used as the entry point for `sto-warp gui`.
Replaces the SETS-coupled `warp/warp_dialog.py` — does not write into a
SETS build; instead it surfaces the `ImportResult` for inspection and
JSON export.

Flow:
  1. User picks one or more screenshots (or a folder).
  2. `RecognitionWorker` runs `WarpImporter.process_folder` off the UI
     thread; progress is forwarded to the status bar.
  3. Result populates the slot tree (one top-level row per slot, child
     rows per slot_index).
  4. File → Export JSON serialises the result for downstream tooling.
"""
from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Qt, Signal
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QHeaderView, QLabel, QMainWindow,
    QMessageBox, QProgressBar, QPushButton, QToolBar, QTreeWidget,
    QTreeWidgetItem, QWidget, QVBoxLayout,
)

from warp.debug import log
from warp.warp_importer import (
    SCREENSHOT_EXTENSIONS, ImportResult, RecognisedItem, WarpImporter,
)

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


def _result_to_dict(result: ImportResult) -> dict:
    """JSON-safe serialisation of `ImportResult` (tuples → lists, drops
    non-serialisable `thumbnail` field)."""
    items = []
    for it in result.items:
        d = asdict(it) if is_dataclass(it) else dict(it.__dict__)
        d.pop('thumbnail', None)
        if isinstance(d.get('bbox'), tuple):
            d['bbox'] = list(d['bbox'])
        items.append(d)
    return {
        'build_type':   result.build_type,
        'ship_name':    result.ship_name,
        'ship_type':    result.ship_type,
        'ship_tier':    result.ship_tier,
        'ship_profile': result.ship_profile,
        'items':        items,
        'errors':       list(result.errors),
        'warnings':     list(result.warnings),
    }


class RecognitionWorker(QObject):
    """Runs `WarpImporter.process_folder` on a worker thread.

    The importer's per-image callback is wired into `progress` for the
    status bar. Final `ImportResult` is delivered via `finished`; any
    fatal exception via `failed`.
    """

    progress = Signal(int, int, str)   # done, total, current file name
    finished = Signal(object)          # ImportResult
    failed   = Signal(str)

    def __init__(self, folder: Path, build_type: str):
        super().__init__()
        self._folder     = folder
        self._build_type = build_type

    def run(self):
        try:
            importer = WarpImporter(build_type=self._build_type)
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
        self._export_btn = QPushButton('Export JSON…', self)
        self._export_btn.clicked.connect(self._on_export_json)
        self._export_btn.setEnabled(False)
        tb.addWidget(self._export_btn)

        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)

        self._summary_lbl = QLabel('Open a screenshot to begin.', central)
        layout.addWidget(self._summary_lbl)

        self._tree = QTreeWidget(central)
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
        layout.addWidget(self._tree, stretch=1)

        self.setCentralWidget(central)

        self._progress = QProgressBar(self)
        self._progress.setRange(0, 100)
        self._progress.setVisible(False)
        self.statusBar().addPermanentWidget(self._progress)
        self.statusBar().showMessage('Ready.')

    # ── File picking ────────────────────────────────────────────────

    def _on_open_files(self):
        exts = ' '.join(f'*{e}' for e in sorted(SCREENSHOT_EXTENSIONS))
        files, _ = QFileDialog.getOpenFileNames(
            self, 'Open screenshot(s)',
            str(Path.home()),
            f'Screenshots ({exts});;All files (*)',
        )
        if not files:
            return
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
        d = QFileDialog.getExistingDirectory(
            self, 'Open screenshot folder', str(Path.home()),
        )
        if not d:
            return
        self._run_against(Path(d))

    # ── Pipeline plumbing ───────────────────────────────────────────

    def _run_against(self, folder: Path):
        if self._thread is not None:
            QMessageBox.information(self, 'Busy',
                                    'A recognition run is already in progress.')
            return

        self._tree.clear()
        self._result = None
        self._export_btn.setEnabled(False)
        self._summary_lbl.setText(f'Recognising {folder}…')
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._set_controls_enabled(False)

        self._thread = QThread(self)
        self._worker = RecognitionWorker(folder, self._build_combo.currentText())
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_thread)
        self._thread.start()

    def _on_progress(self, done: int, total: int, name: str):
        pct = int(done / max(total, 1) * 100)
        self._progress.setValue(pct)
        self.statusBar().showMessage(f'[{done}/{total}] {name}')

    def _on_finished(self, result: ImportResult):
        self._result = result
        self._populate_tree(result)
        self._progress.setValue(100)
        self._progress.setVisible(False)
        msg = f'{len(result.items)} items recognised'
        if result.ship_name or result.ship_type:
            msg = f'{result.ship_name or "?"} ({result.ship_type or "?"}) — ' + msg
        if result.errors:
            msg += f'  ·  {len(result.errors)} error(s)'
        self._summary_lbl.setText(msg)
        self.statusBar().showMessage('Done.')
        self._export_btn.setEnabled(True)
        self._set_controls_enabled(True)

    def _on_failed(self, err: str):
        self._progress.setVisible(False)
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
        for slot in sorted(by_slot):
            entries = sorted(by_slot[slot], key=lambda it: (it.slot_index, it.name))
            parent = QTreeWidgetItem(self._tree)
            parent.setText(0, slot)
            parent.setText(1, str(len(entries)))
            parent.setFirstColumnSpanned(False)
            for it in entries:
                child = QTreeWidgetItem(parent)
                child.setText(0, '')
                child.setText(1, str(it.slot_index))
                child.setText(2, it.name or '—')
                child.setText(3, f'{it.confidence:.2f}')
                child.setText(4, Path(it.source_file).name if it.source_file else '')
            parent.setExpanded(True)

    # ── Export ──────────────────────────────────────────────────────

    def _on_export_json(self):
        if self._result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, 'Export recognition result',
            str(Path.home() / 'warp_result.json'),
            'JSON (*.json);;All files (*)',
        )
        if not path:
            return
        try:
            payload = _result_to_dict(self._result)
            Path(path).write_text(json.dumps(payload, indent=2), encoding='utf-8')
        except Exception as e:
            QMessageBox.critical(self, 'Export failed', f'{type(e).__name__}: {e}')
            return
        self.statusBar().showMessage(f'Exported → {path}')


def main(argv: list[str] | None = None) -> int:
    app = QApplication.instance() or QApplication(argv or sys.argv)
    win = WarpWindow()
    win.show()
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
