"""
warp/folder_picker.py

Three-pane folder/file picker:
  Left   — directory tree (selectable).
  Middle — image files in the selected directory.
  Right  — thumbnail preview of the highlighted file.

In the folder variant the file list is single-selection so the user can
preview without affecting the "Open Folder" outcome — the button always
opens the highlighted directory regardless of which image is shown.

Replaces QFileDialog(FileMode.Directory) which on PySide6 + non-native
mode combined with a NameFilter became unselectable (NoSelection
workaround disabled folder clicks too).
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QDir, QModelIndex
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QSplitter, QTreeView, QListView,
    QAbstractItemView, QPushButton, QLabel, QFileSystemModel,
)

from warp.style import (
    apply_dark_style, primary_btn_style, secondary_btn_style,
    FG, MBG, BC,
)

DEFAULT_IMAGE_FILTERS = ('*.png', '*.jpg', '*.jpeg', '*.webp', '*.bmp')
PREVIEW_MAX = 320


class _PreviewPane(QLabel):
    """Right-pane thumbnail. Scales the source image to fit while
    preserving aspect ratio; shows a placeholder when empty."""

    PLACEHOLDER = '(select an image)'

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumWidth(PREVIEW_MAX)
        self.setStyleSheet(
            f'color:{FG};background:{MBG};border:1px solid {BC};'
            f'border-radius:3px;padding:6px;'
        )
        self._path: Path | None = None
        self.clear()

    def clear(self):
        self._path = None
        super().setPixmap(QPixmap())
        self.setText(self.PLACEHOLDER)

    def show_path(self, path: Path):
        self._path = path
        pm = QPixmap(str(path))
        if pm.isNull():
            self.setText(f'(cannot preview: {path.name})')
            return
        self._render(pm)

    def _render(self, pm: QPixmap):
        side = max(64, min(self.width(), self.height()) - 12)
        scaled = pm.scaled(
            side, side,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        super().setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._path is not None:
            pm = QPixmap(str(self._path))
            if not pm.isNull():
                self._render(pm)


class FolderPickerDialog(QDialog):
    """Split-pane folder picker with image preview."""

    def __init__(
        self,
        parent=None,
        title: str = 'Select Folder',
        start_dir: str | None = None,
        image_filters: tuple[str, ...] = DEFAULT_IMAGE_FILTERS,
    ):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1100, 640)
        apply_dark_style(self)

        self._selected: Path | None = None

        # ── Models ───────────────────────────────────────────────────────────
        self._dir_model = QFileSystemModel(self)
        self._dir_model.setFilter(
            QDir.Filter.Dirs | QDir.Filter.NoDotAndDotDot | QDir.Filter.NoDot
        )
        self._dir_model.setRootPath('')

        self._file_model = QFileSystemModel(self)
        self._file_model.setFilter(QDir.Filter.Files | QDir.Filter.NoDotAndDotDot)
        self._file_model.setNameFilters(list(image_filters))
        self._file_model.setNameFilterDisables(False)  # hide non-matching

        # ── Views ────────────────────────────────────────────────────────────
        self._tree = QTreeView(self)
        self._tree.setModel(self._dir_model)
        for col in range(1, self._dir_model.columnCount()):
            self._tree.hideColumn(col)
        self._tree.setHeaderHidden(True)
        self._tree.selectionModel().currentChanged.connect(self._on_dir_selected)
        self._tree.doubleClicked.connect(self._on_tree_double_click)

        self._list = QListView(self)
        self._list.setModel(self._file_model)
        # Single-selection so users can preview images without affecting
        # the "Open Folder" semantics — the button stays bound to the
        # selected directory regardless of which file is highlighted.
        self._list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._list.setUniformItemSizes(True)
        self._list.selectionModel().currentChanged.connect(self._on_file_changed)

        self._preview = _PreviewPane(self)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.addWidget(self._tree)
        splitter.addWidget(self._list)
        splitter.addWidget(self._preview)
        splitter.setSizes([300, 320, 360])

        # ── Bottom bar ───────────────────────────────────────────────────────
        self._path_label = QLabel('—', self)
        self._path_label.setStyleSheet(
            f'color:{FG};background:{MBG};border:1px solid {BC};'
            f'border-radius:3px;padding:4px 8px;'
        )

        self._open_btn = QPushButton('Open Folder', self)
        self._open_btn.setStyleSheet(primary_btn_style())
        self._open_btn.setEnabled(False)
        self._open_btn.clicked.connect(self._accept)

        cancel_btn = QPushButton('Cancel', self)
        cancel_btn.setStyleSheet(secondary_btn_style())
        cancel_btn.clicked.connect(self.reject)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self._path_label, 1)
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(self._open_btn)

        root = QVBoxLayout(self)
        root.addWidget(splitter, 1)
        root.addLayout(btn_row)

        # ── Pre-select start directory ───────────────────────────────────────
        if start_dir:
            p = Path(start_dir).expanduser()
            if p.is_dir():
                idx = self._dir_model.index(str(p))
                if idx.isValid():
                    self._tree.expand(idx)
                    self._tree.setCurrentIndex(idx)
                    self._tree.scrollTo(
                        idx, QAbstractItemView.ScrollHint.PositionAtCenter
                    )

    def _on_dir_selected(self, current: QModelIndex, _previous: QModelIndex):
        if not current.isValid():
            return
        path = self._dir_model.filePath(current)
        self._selected = Path(path)
        self._path_label.setText(path)
        self._file_model.setRootPath(path)
        self._list.setRootIndex(self._file_model.index(path))
        self._open_btn.setEnabled(True)
        self._preview.clear()

    def _on_file_changed(self, current: QModelIndex, _previous: QModelIndex):
        if current.isValid() and not self._file_model.isDir(current):
            self._preview.show_path(Path(self._file_model.filePath(current)))
        else:
            self._preview.clear()

    def _on_tree_double_click(self, idx: QModelIndex):
        if idx.isValid() and self._dir_model.isDir(idx):
            self._accept()

    def _accept(self):
        if self._selected and self._selected.is_dir():
            self.accept()

    def selected_folder(self) -> Path | None:
        return self._selected


def pick_folder(
    parent,
    title: str = 'Select Folder',
    start_dir: str | None = None,
    image_filters: tuple[str, ...] = DEFAULT_IMAGE_FILTERS,
) -> Path | None:
    """Show the picker modally; return chosen folder or None on cancel."""
    dlg = FolderPickerDialog(parent, title=title, start_dir=start_dir,
                             image_filters=image_filters)
    if dlg.exec():
        return dlg.selected_folder()
    return None


class FilePickerDialog(QDialog):
    """Same split-pane layout as :class:`FolderPickerDialog`, with optional
    multi-select on the right pane. Open is enabled as soon as at least one
    file is selected. ``multi=False`` restricts to single-file selection.
    """

    def __init__(
        self,
        parent=None,
        title: str = 'Select Files',
        start_dir: str | None = None,
        image_filters: tuple[str, ...] = DEFAULT_IMAGE_FILTERS,
        multi: bool = True,
    ):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1100, 640)
        apply_dark_style(self)

        self._selected_files: list[Path] = []
        self._current_dir: Path | None = None

        # ── Models ───────────────────────────────────────────────────────────
        self._dir_model = QFileSystemModel(self)
        self._dir_model.setFilter(
            QDir.Filter.Dirs | QDir.Filter.NoDotAndDotDot | QDir.Filter.NoDot
        )
        self._dir_model.setRootPath('')

        self._file_model = QFileSystemModel(self)
        self._file_model.setFilter(QDir.Filter.Files | QDir.Filter.NoDotAndDotDot)
        self._file_model.setNameFilters(list(image_filters))
        self._file_model.setNameFilterDisables(False)

        # ── Views ────────────────────────────────────────────────────────────
        self._tree = QTreeView(self)
        self._tree.setModel(self._dir_model)
        for col in range(1, self._dir_model.columnCount()):
            self._tree.hideColumn(col)
        self._tree.setHeaderHidden(True)
        self._tree.selectionModel().currentChanged.connect(self._on_dir_selected)

        self._list = QListView(self)
        self._list.setModel(self._file_model)
        self._list.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection if multi
            else QAbstractItemView.SelectionMode.SingleSelection
        )
        self._list.setUniformItemSizes(True)
        self._list.doubleClicked.connect(self._on_file_double_click)
        self._list.selectionModel().selectionChanged.connect(self._on_files_changed)
        self._list.selectionModel().currentChanged.connect(self._on_file_focused)

        self._preview = _PreviewPane(self)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.addWidget(self._tree)
        splitter.addWidget(self._list)
        splitter.addWidget(self._preview)
        splitter.setSizes([300, 320, 360])

        # ── Bottom bar ───────────────────────────────────────────────────────
        self._path_label = QLabel('—', self)
        self._path_label.setStyleSheet(
            f'color:{FG};background:{MBG};border:1px solid {BC};'
            f'border-radius:3px;padding:4px 8px;'
        )

        self._open_btn = QPushButton('Open', self)
        self._open_btn.setStyleSheet(primary_btn_style())
        self._open_btn.setEnabled(False)
        self._open_btn.clicked.connect(self._accept)

        cancel_btn = QPushButton('Cancel', self)
        cancel_btn.setStyleSheet(secondary_btn_style())
        cancel_btn.clicked.connect(self.reject)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self._path_label, 1)
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(self._open_btn)

        root = QVBoxLayout(self)
        root.addWidget(splitter, 1)
        root.addLayout(btn_row)

        # ── Pre-select start directory ───────────────────────────────────────
        if start_dir:
            p = Path(start_dir).expanduser()
            if p.is_dir():
                idx = self._dir_model.index(str(p))
                if idx.isValid():
                    self._tree.expand(idx)
                    self._tree.setCurrentIndex(idx)
                    self._tree.scrollTo(
                        idx, QAbstractItemView.ScrollHint.PositionAtCenter
                    )

    def _on_dir_selected(self, current: QModelIndex, _previous: QModelIndex):
        if not current.isValid():
            return
        path = self._dir_model.filePath(current)
        self._current_dir = Path(path)
        self._path_label.setText(path)
        self._file_model.setRootPath(path)
        self._list.setRootIndex(self._file_model.index(path))
        self._preview.clear()
        self._refresh_selection()

    def _on_files_changed(self, *_):
        self._refresh_selection()

    def _on_file_focused(self, current: QModelIndex, _previous: QModelIndex):
        if current.isValid() and not self._file_model.isDir(current):
            self._preview.show_path(Path(self._file_model.filePath(current)))
        else:
            self._preview.clear()

    def _refresh_selection(self):
        idxs = self._list.selectionModel().selectedIndexes()
        self._selected_files = [
            Path(self._file_model.filePath(i)) for i in idxs
        ]
        self._open_btn.setEnabled(bool(self._selected_files))

    def _on_file_double_click(self, idx: QModelIndex):
        if idx.isValid() and not self._file_model.isDir(idx):
            self._selected_files = [Path(self._file_model.filePath(idx))]
            self._accept()

    def _accept(self):
        if self._selected_files:
            self.accept()

    def selected_files(self) -> list[Path]:
        return list(self._selected_files)


def pick_files(
    parent,
    title: str = 'Select Files',
    start_dir: str | None = None,
    image_filters: tuple[str, ...] = DEFAULT_IMAGE_FILTERS,
    multi: bool = True,
) -> list[Path]:
    """Show the file picker modally; return chosen files (possibly empty
    on cancel). Set ``multi=False`` for single-file selection."""
    dlg = FilePickerDialog(parent, title=title, start_dir=start_dir,
                           image_filters=image_filters, multi=multi)
    if dlg.exec():
        return dlg.selected_files()
    return []
