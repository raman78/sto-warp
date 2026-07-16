"""Unified WARP Results view — 3-pane interactive replacement for the old
Results tree + Preview tabs.

Layout (left → right inside a horizontal QSplitter):

    file list          interactive screenshot canvas        sorted item tree
    (QListWidget)      (image + bboxes, hover+click)        (QTreeWidget)

Cross-pane wiring:
  - file list row change → loads image in canvas; rows in the right tree
    that belong to that file get a faint background tint
  - click a bbox in the canvas → selects the matching row in the right tree
  - click a row in the right tree → highlights the matching bbox in the
    canvas (dashed, +1 px, same per-slot color)
  - right-click on any pane → unified context menu (Copy filename / Copy
    full path / Open in WARP CORE / Open in WARP Fast Correction Mode)

Read-only — no bbox editing, no annotation writes. Pure consumer of
``ImportResult``.

Also carries the per-file Screen-Type override + Rerun button from the
old Preview tab (UX is unchanged: pick a type → Rerun).
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QSize, QPoint, Signal
from PySide6.QtGui import (
    QBrush, QColor, QFont, QPainter, QPen, QPixmap,
)
from PySide6.QtWidgets import (
    QApplication, QComboBox, QHBoxLayout, QHeaderView, QLabel, QListWidget,
    QListWidgetItem, QMenu, QPushButton, QScrollArea, QSplitter, QStyle,
    QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from warp.recognition.boff_keys import group_items_by_seat, order_items_for_display
from warp.style import (
    BG as _THEME_BG, LBG as _THEME_LBG, ACCENT as _THEME_ACCENT,
    primary_btn_style,
)
from warp.warp_importer import (
    ImportResult, RecognisedItem, SLOT_ORDER, DISPLAY_CANONICAL_ORDER,
)


# Selection accent — gold/amber, matches the Export to SETS JSON button.
_SEL_COLOR = QColor(_THEME_ACCENT)
# Pale, very translucent green for "this row belongs to the file the user
# is currently looking at". Drawn under the whole row, not just one cell.
_FILE_TINT = QColor(120, 220, 120, 38)


# Override choices — mirror the toolbar combo so muscle memory matches.
# Must cover every value produced by SCREEN_TYPE_TO_BUILD_TYPE so the
# combo is selectable for any auto-detected screen.
_OVERRIDE_BUILD_TYPES = (
    'SPACE_MIXED',
    'GROUND_MIXED',
    'SPACE',
    'GROUND',
    'BOFFS',
    'SPACE_BOFFS',
    'GROUND_BOFFS',
    'TRAITS',
    'SPACE_TRAITS',
    'GROUND_TRAITS',
    'SPEC',
    'SKILLS',
    'SPACE_SKILLS',
    'GROUND_SKILLS',
    'DISCARD',
)


def _color_for_slot(slot: str) -> QColor:
    """Stable per-slot-family color. Deterministic so the same slot lands
    on the same hue across screenshots in a batch."""
    s = (slot or '').lower()
    if s.startswith('boff'):
        return QColor(255, 170,  40)
    if 'trait' in s:
        return QColor(120, 220, 120)
    if 'specialization' in s:
        return QColor(220, 120, 220)
    if 'console' in s:
        return QColor(180, 180, 255)
    if s in ('deflector', 'engines', 'warp core', 'shields',
             'singularity core', 'secondary deflector', 'devices'):
        return QColor(100, 200, 255)
    if 'weapon' in s:
        return QColor(255, 100, 100)
    if s in ('body armor', 'ev suit', 'kit', 'kit modules',
             'personal shield', 'ground devices'):
        return QColor(  0, 255, 200)
    h = (sum(ord(c) for c in slot) * 17) % 360
    return QColor.fromHsv(h, 200, 240)


from warp.gui import _tooltip_html  # noqa: E402  (after top-level imports)


class _InteractiveCanvas(QWidget):
    """Paints one screenshot with bbox overlay. Tracks hover, emits clicks.

    Items are passed in two parallel arrays:
      _items   — RecognisedItem list (drives drawing)
      _gidx    — int list, same length as _items, mapping each canvas
                 bbox to a *global* index in ImportResult.items so the
                 owning ResultsView can correlate canvas clicks with rows
                 in the right tree.
    """

    bbox_clicked  = Signal(int)   # global item index, -1 = empty area
    context_menu  = Signal(QPoint, int)  # (global pos, global idx | -1)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
        self._scaled: QPixmap | None = None
        self._items: list[RecognisedItem] = []
        self._gidx:  list[int] = []
        self._build_type: str = ''
        self._scale: float = 1.0
        self._highlight_set: set[int] = set()
        self._hover_gidx:     int = -1
        self._hover_timer: object = None
        self.setMinimumSize(200, 200)
        self.setMouseTracking(True)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._on_context_menu_local)

    # ── Public API ──────────────────────────────────────────────────

    def set_image(self, path: Path,
                  items: list[RecognisedItem],
                  gidx:  list[int],
                  build_type: str = '') -> None:
        pm = QPixmap(str(path)) if path.is_file() else QPixmap()
        self._pixmap = pm if not pm.isNull() else None
        self._items = list(items)
        self._gidx  = list(gidx)
        self._build_type = build_type or ''
        self._scaled = None
        self._highlight_set = set()
        self._hover_gidx = -1
        self._cancel_hover_timer()
        self._compute_fit()
        self.update()

    def clear(self) -> None:
        self._pixmap = None
        self._scaled = None
        self._items = []
        self._gidx  = []
        self._build_type = ''
        self._highlight_set = set()
        self._hover_gidx = -1
        self._cancel_hover_timer()
        self.update()

    def set_highlight(self, gidx: int) -> None:
        new_set: set[int] = {gidx} if gidx >= 0 else set()
        if new_set == self._highlight_set:
            return
        self._highlight_set = new_set
        self.update()

    def set_highlight_set(self, gidxs) -> None:
        new_set = {g for g in gidxs if isinstance(g, int) and g >= 0}
        if new_set == self._highlight_set:
            return
        self._highlight_set = new_set
        self.update()

    # ── Geometry ────────────────────────────────────────────────────

    def _compute_fit(self) -> None:
        if self._pixmap is None:
            self._scale = 1.0
            return
        pw, ph = self._pixmap.width(), self._pixmap.height()
        if pw == 0 or ph == 0:
            self._scale = 1.0
            return
        vp = self.size()
        s = min(vp.width() / pw, vp.height() / ph, 1.0)
        self._scale = max(s, 0.05)

    def resizeEvent(self, e):
        prev = self._scale
        self._compute_fit()
        if self._scale != prev:
            self._scaled = None
        super().resizeEvent(e)

    def sizeHint(self) -> QSize:
        if self._pixmap is None:
            return QSize(640, 400)
        return QSize(int(self._pixmap.width()  * self._scale),
                     int(self._pixmap.height() * self._scale))

    def _image_origin(self) -> tuple[int, int]:
        if self._scaled is None and self._pixmap is not None:
            self._scaled = self._pixmap.scaled(
                int(self._pixmap.width()  * self._scale),
                int(self._pixmap.height() * self._scale),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        if self._scaled is None:
            return (0, 0)
        x0 = (self.width()  - self._scaled.width())  // 2
        y0 = (self.height() - self._scaled.height()) // 2
        return (x0, y0)

    def _bbox_at(self, px: int, py: int) -> int:
        """Return global item index whose bbox contains the widget-space
        point (px, py), or -1 if none. Picks the smallest bbox under the
        cursor so nested boxes (BOFF abilities inside a seat rectangle)
        resolve to the inner one."""
        if self._pixmap is None:
            return -1
        x0, y0 = self._image_origin()
        best_g = -1
        best_area = None
        for it, g in zip(self._items, self._gidx):
            if not it.bbox or len(it.bbox) < 4:
                continue
            x, y, w, h = it.bbox[:4]
            rx = x0 + int(x * self._scale)
            ry = y0 + int(y * self._scale)
            rw = max(1, int(w * self._scale))
            rh = max(1, int(h * self._scale))
            if rx <= px <= rx + rw and ry <= py <= ry + rh:
                a = rw * rh
                if best_area is None or a < best_area:
                    best_area = a
                    best_g = g
        return best_g

    # ── Mouse ───────────────────────────────────────────────────────

    def mouseMoveEvent(self, e):
        g = self._bbox_at(e.position().x(), e.position().y())
        if g != self._hover_gidx:
            self._hover_gidx = g
            self._cancel_hover_timer()
            if g >= 0:
                self._start_hover_timer(g)
            else:
                from PySide6.QtWidgets import QToolTip
                QToolTip.hideText()
            self.update()
        super().mouseMoveEvent(e)

    def _cancel_hover_timer(self):
        if self._hover_timer is not None:
            self._hover_timer.stop()
            self._hover_timer = None

    def _start_hover_timer(self, gidx: int):
        from PySide6.QtCore import QTimer
        self._hover_timer = QTimer(self)
        self._hover_timer.setSingleShot(True)
        self._hover_timer.timeout.connect(lambda: self._show_hover_tooltip(gidx))
        self._hover_timer.start(500)

    def _show_hover_tooltip(self, gidx: int):
        it: RecognisedItem | None = None
        for item, g in zip(self._items, self._gidx):
            if g == gidx:
                it = item
                break
        if it is None or not it.name:
            return
        from warp.recognition.boff_keys import pretty_slot
        slot = pretty_slot(it.slot or '?')
        conf = it.confidence or 0.0
        color = ('#7effc8' if conf >= 0.85 else
                 '#e8c060' if conf >= 0.70 else '#ff9966')
        info_html = (f'<b>{slot}</b><br>{it.name}'
                     f'<br>Confidence: <span style="color:{color}">{conf:.0%}</span>')

        text = _tooltip_html(it.thumbnail, it.name, info_html)

        from PySide6.QtWidgets import QToolTip
        from PySide6.QtGui import QCursor
        QToolTip.showText(QCursor.pos(), text, self)

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            g = self._bbox_at(e.position().x(), e.position().y())
            self.bbox_clicked.emit(g)
        super().mousePressEvent(e)

    def _on_context_menu_local(self, pos: QPoint):
        g = self._bbox_at(pos.x(), pos.y())
        self.context_menu.emit(self.mapToGlobal(pos), g)

    # ── Painting ────────────────────────────────────────────────────

    def paintEvent(self, _e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(_THEME_BG))
        if self._pixmap is None:
            p.setPen(QColor(160, 160, 160))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       '(no preview)')
            return

        x0, y0 = self._image_origin()
        p.drawPixmap(x0, y0, self._scaled)

        # Build-type badge upper-left of the image
        if self._build_type:
            bf = QFont('Monospace')
            bf.setPointSize(10)
            bf.setBold(True)
            p.setFont(bf)
            fm = p.fontMetrics()
            pad_x, pad_y = 8, 4
            tw = fm.horizontalAdvance(self._build_type)
            th = fm.height()
            bx = x0 + 6
            by = y0 + 6
            p.fillRect(bx, by, tw + pad_x * 2, th + pad_y * 2,
                       QColor(0, 0, 0, 180))
            p.setPen(QColor(220, 220, 220))
            p.drawText(bx + pad_x, by + pad_y + fm.ascent(), self._build_type)

        f = QFont('Monospace')
        f.setPointSize(8)
        p.setFont(f)

        # Pass 1 — bbox rectangles + per-bbox confidence. Selected bbox
        # uses a dashed pen, +1 px thicker, same per-slot color.
        for it, g in zip(self._items, self._gidx):
            if not it.bbox or len(it.bbox) < 4:
                continue
            x, y, w, h = it.bbox[:4]
            rx = x0 + int(x * self._scale)
            ry = y0 + int(y * self._scale)
            rw = max(1, int(w * self._scale))
            rh = max(1, int(h * self._scale))
            color = _color_for_slot(it.slot)
            selected = (g in self._highlight_set)
            hovered  = (g == self._hover_gidx) and not selected
            if selected:
                # Selected bbox jumps to the amber accent so it stands out
                # against any slot family colour.
                pen = QPen(_SEL_COLOR, 3)
                pen.setStyle(Qt.PenStyle.DashLine)
            elif hovered:
                pen = QPen(color.lighter(140), 2)
            else:
                pen = QPen(color, 2)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(rx, ry, rw, rh)
            if it.confidence is not None:
                conf_txt = f'{it.confidence:.0%}'
                p.setPen(QPen(QColor(0, 0, 0, 200), 3))
                p.drawText(rx + 2, max(ry - 2, 10), conf_txt)
                p.setPen(color)
                p.drawText(rx + 2, max(ry - 2, 10), conf_txt)

        # Pass 2 — group labels (seat-aware).
        groups: list[dict] = []
        for label, items in group_items_by_seat(self._items):
            bboxed = [it for it in items if it.bbox and len(it.bbox) >= 4]
            if not bboxed:
                continue
            rects = []
            for it in bboxed:
                x, y, w, h = it.bbox[:4]
                rx = x0 + int(x * self._scale)
                ry = y0 + int(y * self._scale)
                rw = max(1, int(w * self._scale))
                rh = max(1, int(h * self._scale))
                rects.append((rx, ry, rx + rw, ry + rh))
            groups.append({
                'label': label,
                'color': _color_for_slot(bboxed[0].slot),
                'x0': min(r[0] for r in rects),
                'y0': min(r[1] for r in rects),
                'x1': max(r[2] for r in rects),
                'y1': max(r[3] for r in rects),
            })
        gf = QFont('Monospace')
        gf.setPointSize(10)
        gf.setBold(True)
        p.setFont(gf)
        fm = p.fontMetrics()
        for g in groups:
            label = g['label']
            s = label.lower()
            tw = fm.horizontalAdvance(label)
            th = fm.height()
            if s.startswith('boff'):
                lx = (g['x0'] + g['x1']) // 2 - tw // 2
                ly = max(g['y0'] - 18, th)
            elif 'trait' in s or 'reputation' in s or 'rep' == s.split()[-1]:
                lx = g['x1'] + 8
                ly = (g['y0'] + g['y1']) // 2 + th // 3
            else:
                lx = max(g['x0'] - tw - 8, 2)
                ly = (g['y0'] + g['y1']) // 2 + th // 3
            p.setPen(QPen(QColor(0, 0, 0, 220), 4))
            p.drawText(lx, ly, label)
            p.setPen(g['color'])
            p.drawText(lx, ly, label)


class ResultsView(QWidget):
    """Replacement for the old Results tree + Preview tabs.

    Public signals:
      rerun_requested            (dict[str_path, build_type])
      open_in_warp_core          (str_path, list[RecognisedItem])
      open_in_warp_fast_corr     (dict[str_path, list[RecognisedItem]],
                                  dict[str_path, str])
    """

    rerun_requested        = Signal(dict)
    open_in_warp_core      = Signal(str, object)
    # Two dicts: items per file, AND screen_type per file (so Fast
    # Correction can render the screen-type WARP just classified, not
    # whatever the user previously confirmed for that filename in TDM).
    open_in_warp_fast_corr = Signal(dict, dict)
    # Fired when the user presses the "Export to SETS JSON…" button at
    # the bottom of the Results view. WarpWindow owns the actual export
    # (file dialog + SETS writer); the view only provides the trigger.
    export_sets_requested  = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._result: ImportResult | None = None
        # Resolved file path → list[RecognisedItem]
        self._items_by_file: dict[str, list[RecognisedItem]] = {}
        # Resolved file path → detected build_type (autodetector pick)
        self._bt_by_file:    dict[str, str] = {}
        # Resolved file path → detected screen_type (full ML label,
        # finer-grained than build_type). Used by the Fast Correction
        # handoff to pass WARP's actual classification through to the
        # trainer.
        self._stype_by_file: dict[str, str] = {}
        # Resolved file path → user-chosen build_type (overrides detected)
        self._overrides:     dict[str, str] = {}
        self._file_keys:     list[str] = []
        # Global index → file-resolved-path (parallel to ImportResult.items)
        self._gidx_to_file:  list[str] = []
        # Currently displayed file (resolved abs path) — drives canvas + tint
        self._current_file:  str = ''
        # Re-entrancy guard for the override combo
        self._suppress_combo = False
        # Flag set when a callback is already updating the highlight so the
        # paired widget doesn't echo the change back into a loop.
        self._syncing_highlight = False
        # Launcher flips these to advertise which handoff entries should
        # appear in the right-click menu.
        self._has_warp_core_handler = False
        self._has_fast_correction_handler = False

        self._build_ui()

    # ── UI ──────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)

        split = QSplitter(Qt.Orientation.Horizontal, self)

        # ── Left pane: file list + Rerun button
        left = QWidget(self)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        self._list = QListWidget(left)
        self._list.setMinimumWidth(220)
        self._list.itemSelectionChanged.connect(self._on_file_selected)
        self._list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._list.customContextMenuRequested.connect(self._on_list_context_menu)
        ll.addWidget(self._list, stretch=1)
        self._rerun_btn = QPushButton('Rerun Recognition', left)
        self._rerun_btn.setStyleSheet(primary_btn_style())
        self._rerun_btn.setToolTip(
            'Re-run recognition on the same folder, applying your per-file '
            'screen-type overrides.')
        self._rerun_btn.clicked.connect(self._on_rerun_clicked)
        self._rerun_btn.setVisible(False)
        ll.addWidget(self._rerun_btn)
        # Export sits in the left pane beneath the file list — mirrors
        # WARP CORE's left panel where Mark Done lives beneath
        # Screenshots. Disabled until a result is loaded; WarpWindow
        # toggles it via `set_export_enabled`.
        self._export_sets_btn = QPushButton('Export to SETS JSON…', left)
        self._export_sets_btn.setStyleSheet(primary_btn_style())
        self._export_sets_btn.setToolTip(
            'SETS v3.0.0-compatible build JSON — loadable via SETS '
            'File → Load Build.')
        self._export_sets_btn.setEnabled(False)
        self._export_sets_btn.clicked.connect(self.export_sets_requested.emit)
        ll.addWidget(self._export_sets_btn)
        split.addWidget(left)

        # ── Middle pane: screen-type combo + interactive canvas
        mid = QWidget(self)
        ml = QVBoxLayout(mid)
        ml.setContentsMargins(0, 0, 0, 0)
        top_row = QHBoxLayout()
        top_row.setContentsMargins(4, 0, 4, 4)
        top_row.addWidget(QLabel('Screen type:', mid))
        self._type_combo = QComboBox(mid)
        self._type_combo.addItems(_OVERRIDE_BUILD_TYPES)
        self._type_combo.setEnabled(False)
        self._type_combo.currentTextChanged.connect(self._on_combo_changed)
        top_row.addWidget(self._type_combo)
        self._override_lbl = QLabel('', mid)
        self._override_lbl.setStyleSheet('color: #d4a017;')
        top_row.addWidget(self._override_lbl)
        top_row.addStretch(1)
        ml.addLayout(top_row)
        self._scroll = QScrollArea(mid)
        self._scroll.setWidgetResizable(True)
        self._canvas = _InteractiveCanvas(self._scroll)
        self._canvas.bbox_clicked.connect(self._on_canvas_bbox_clicked)
        self._canvas.context_menu.connect(self._on_canvas_context_menu)
        self._scroll.setWidget(self._canvas)
        ml.addWidget(self._scroll, stretch=1)
        split.addWidget(mid)

        # ── Right pane: sorted item tree
        self._tree = QTreeWidget(self)
        self._tree.setColumnCount(4)
        self._tree.setHeaderLabels(['Slot', 'Idx', 'Item', 'Conf'])
        # stretchLastSection defaults to True, which silently overrides the
        # explicit width on the last column and steals the divider grab
        # zone on its left. Turn it off so each column has a real divider
        # the user can drag and so Conf actually honours its 56 px width.
        h = self._tree.header()
        h.setStretchLastSection(False)
        # All columns Interactive — Qt's Stretch mode silently fixes the
        # section's width to the calculated stretch-fill and refuses to
        # let the user drag the divider on either side of it, which is
        # what made the Item↔Conf boundary unresponsive. Without any
        # Stretch column the tree may leave trailing whitespace if the
        # panel is wider than the sum of column widths — acceptable in
        # exchange for fully resizable boundaries.
        for c in range(4):
            h.setSectionResizeMode(c, QHeaderView.ResizeMode.Interactive)
        self._tree.setAlternatingRowColors(False)
        self._tree.setRootIsDecorated(True)
        self._tree.setUniformRowHeights(True)
        self._tree.setColumnWidth(0, 170)
        self._tree.setColumnWidth(1, 40)
        self._tree.setColumnWidth(2, 240)
        self._tree.setColumnWidth(3, 56)   # fits '100%'
        self._tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._on_tree_context_menu)
        self._tree.itemSelectionChanged.connect(self._on_tree_selection_changed)
        # Bold the selected leaf row so the active pick stands out even
        # against the ACCENT-tinted selection background.
        self._tree.itemSelectionChanged.connect(self._refresh_bold_selected)
        # Amber-accented selected row + currently-displayed file gets a
        # subtle pale-green wash on its rows (drawn per-cell in
        # _apply_file_tint — this stylesheet handles only selection).
        self._tree.setStyleSheet(
            f'QTreeWidget::item:selected {{ '
            f'background-color: {_THEME_ACCENT}; color: #1a1a1a; }}'
        )
        split.addWidget(self._tree)

        # File list is select-only: a click picks a file, no toggle-off
        # behaviour (clearing the active file via clicking it again was
        # confusing — the canvas would flicker between "loaded" and
        # "empty" depending on click cadence). _on_file_selected handles
        # all selection moves.

        split.setSizes([240, 700, 320])
        root.addWidget(split)

    def set_export_enabled(self, enabled: bool) -> None:
        self._export_sets_btn.setEnabled(bool(enabled))

    # ── Public API ──────────────────────────────────────────────────

    def clear(self) -> None:
        self._result = None
        self._items_by_file = {}
        self._bt_by_file = {}
        self._file_keys = []
        self._gidx_to_file = []
        self._overrides = {}
        self._current_file = ''
        self._list.clear()
        self._tree.clear()
        self._canvas.clear()
        self._type_combo.setEnabled(False)
        self._override_lbl.setText('')
        self._rerun_btn.setVisible(False)
        self._export_sets_btn.setEnabled(False)

    def preload_files(self, paths: list) -> None:
        """Populate the file list with screenshot paths before recognition runs.

        Used by WARP's Open Files / Open Folder flow so the user sees the
        loaded screenshots immediately. Per-file screen types are filled in
        later via ``update_file_screen_type`` as the background classifier
        produces results. A subsequent ``set_result`` call replaces this
        preview with the full recognition output.
        """
        kept_overrides = dict(self._overrides)
        self.clear()
        self._overrides = kept_overrides
        self._result = None
        keys: list[str] = []
        for p in paths:
            try:
                k = str(Path(p).resolve())
            except OSError:
                k = str(p)
            keys.append(k)
            self._items_by_file[k] = []
        self._file_keys = keys
        self._gidx_to_file = []
        for src in keys:
            self._list.addItem(QListWidgetItem(self._list_label_for(src)))
        if self._list.count():
            self._list.setCurrentRow(0)

    def update_file_screen_type(self, src, stype: str) -> None:
        """Set the detected screen-type tag for a single preloaded file."""
        from warp.warp_importer import SCREEN_TYPE_TO_BUILD_TYPE
        try:
            key = str(Path(src).resolve())
        except OSError:
            key = str(src)
        bt = SCREEN_TYPE_TO_BUILD_TYPE.get(stype, '')
        if bt:
            self._bt_by_file[key] = bt
        if key in self._file_keys:
            row = self._file_keys.index(key)
            item = self._list.item(row)
            if item is not None:
                item.setText(self._list_label_for(key))
        if self._current_file == key:
            detected = self._bt_by_file.get(key, '')
            self._suppress_combo = True
            if detected and detected in _OVERRIDE_BUILD_TYPES:
                self._type_combo.setCurrentText(detected)
            else:
                self._type_combo.setCurrentIndex(0)
            self._suppress_combo = False
            self._type_combo.setEnabled(True)
            # Refresh the canvas badge so it matches the file-list tag.
            self._canvas.set_image(
                Path(key),
                self._items_by_file.get(key, []),
                [self._gidx_to_file_index(it)
                 for it in self._items_by_file.get(key, [])],
                self._overrides.get(key) or detected,
            )

    def set_result(self, result: ImportResult) -> None:
        kept_overrides = dict(self._overrides)
        self.clear()
        self._overrides = kept_overrides
        self._result = result

        # Bucket items by resolved source path so previews survive the
        # staging tempdir being deleted after the recognition thread ends.
        by_file: dict[str, list[RecognisedItem]] = {}
        gidx_to_file: list[str] = []
        for it in result.items:
            if not it.source_file:
                gidx_to_file.append('')
                continue
            try:
                key = str(Path(it.source_file).resolve())
            except OSError:
                key = it.source_file
            by_file.setdefault(key, []).append(it)
            gidx_to_file.append(key)
        self._items_by_file = by_file
        self._gidx_to_file  = gidx_to_file
        self._bt_by_file    = dict(getattr(result, 'per_file', {}) or {})
        self._stype_by_file = dict(
            getattr(result, 'per_file_screen_type', {}) or {})
        # Union: every file the importer touched + every file with items.
        self._file_keys = sorted(set(by_file) | set(self._bt_by_file))

        for src in self._file_keys:
            self._list.addItem(QListWidgetItem(self._list_label_for(src)))

        self._populate_tree(result)

        if self._list.count():
            self._list.setCurrentRow(0)
        self._refresh_rerun_visibility()

    def set_warp_core_handler(self, has: bool) -> None:
        self._has_warp_core_handler = bool(has)

    def set_fast_correction_handler(self, has: bool) -> None:
        self._has_fast_correction_handler = bool(has)

    def current_file(self) -> str:
        return self._current_file

    def current_build_type(self) -> str:
        if not self._current_file:
            return ''
        return self._overrides.get(self._current_file, '') or \
               self._bt_by_file.get(self._current_file, '')

    # ── Internals: left pane ────────────────────────────────────────

    def _list_label_for(self, src: str) -> str:
        items = self._items_by_file.get(src, [])
        with_bbox = sum(1 for it in items
                        if it.bbox and len(it.bbox) >= 4)
        bt = self._bt_by_file.get(src, '')
        ov = self._overrides.get(src, '')
        if ov and ov != bt:
            bt_tag = f'  [{bt} → {ov}]'
        elif bt:
            bt_tag = f'  [{bt}]'
        else:
            bt_tag = ''
        return f'{Path(src).name}   ({with_bbox}/{len(items)}){bt_tag}'

    def _refresh_bold_selected(self):
        """Bold the font of the currently selected leaf row so the
        active pick is legible against the ACCENT highlight."""
        sel = set(self._tree.selectedItems())
        for i in range(self._tree.topLevelItemCount()):
            parent = self._tree.topLevelItem(i)
            for j in range(parent.childCount()):
                child = parent.child(j)
                bold = child in sel
                f = child.font(0)
                if f.bold() == bold:
                    continue
                f.setBold(bold)
                for c in range(self._tree.columnCount()):
                    child.setFont(c, f)

    def _on_file_selected(self):
        row = self._list.currentRow()
        if row < 0 or row >= len(self._file_keys):
            self._canvas.clear()
            self._type_combo.setEnabled(False)
            self._override_lbl.setText('')
            self._current_file = ''
            self._apply_file_tint()
            return
        src = self._file_keys[row]
        self._current_file = src
        detected = self._bt_by_file.get(src, '')
        current  = self._overrides.get(src, '') or detected

        self._suppress_combo = True
        if current and current in _OVERRIDE_BUILD_TYPES:
            self._type_combo.setCurrentText(current)
        else:
            self._type_combo.setCurrentIndex(0)
        self._suppress_combo = False
        self._type_combo.setEnabled(True)

        if src in self._overrides and self._overrides[src] != detected:
            self._override_lbl.setText(
                f'(override — detected: {detected or "?"})')
        else:
            self._override_lbl.setText('')

        items = self._items_by_file.get(src, [])
        # Build canvas → global-index map by linear scan of the global list.
        gidx = []
        for it in items:
            try:
                gidx.append(self._gidx_to_file_index(it))
            except Exception:
                gidx.append(-1)
        self._canvas.set_image(Path(src), items, gidx,
                               self._overrides.get(src) or detected)
        self._apply_file_tint()

    def _gidx_to_file_index(self, item: RecognisedItem) -> int:
        """Return the global ImportResult.items index for `item`.
        Falls back to identity via `is`."""
        if self._result is None:
            return -1
        for i, it in enumerate(self._result.items):
            if it is item:
                return i
        return -1

    def _on_combo_changed(self, value: str):
        if self._suppress_combo:
            return
        row = self._list.currentRow()
        if row < 0 or row >= len(self._file_keys):
            return
        src = self._file_keys[row]
        detected = self._bt_by_file.get(src, '')
        if value == detected or not value:
            self._overrides.pop(src, None)
            self._override_lbl.setText('')
        else:
            self._overrides[src] = value
            self._override_lbl.setText(
                f'(override — detected: {detected or "?"})')
        item = self._list.item(row)
        if item is not None:
            item.setText(self._list_label_for(src))
        self._canvas.set_image(
            Path(src),
            self._items_by_file.get(src, []),
            [self._gidx_to_file_index(it)
             for it in self._items_by_file.get(src, [])],
            self._overrides.get(src) or detected,
        )
        self._refresh_rerun_visibility()

    def _refresh_rerun_visibility(self):
        any_diff = any(
            self._overrides.get(k, '') and
            self._overrides[k] != self._bt_by_file.get(k, '')
            for k in self._overrides
        )
        self._rerun_btn.setVisible(any_diff)

    def _on_rerun_clicked(self):
        payload = {
            k: v for k, v in self._overrides.items()
            if v and v != self._bt_by_file.get(k, '')
        }
        if not payload:
            return
        self.rerun_requested.emit(payload)

    # ── Internals: right tree ───────────────────────────────────────

    def _populate_tree(self, result: ImportResult):
        self._tree.clear()
        # Ordering + seat-aware BOFF grouping live in
        # warp.recognition.boff_keys.order_items_for_display — shared
        # with WARP CORE so the two UIs cannot drift on slot ordering.
        canonical = [sd['name'] for sd in SLOT_ORDER.get(result.build_type, [])]
        ordered_groups = order_items_for_display(
            result.items, canonical,
            fallback_canonical_slots=DISPLAY_CANONICAL_ORDER,
        )

        _boff_order = [lbl for lbl, _ in ordered_groups if lbl.startswith('Boff')]
        if _boff_order:
            from warp.debug import log as _wlog
            _wlog.info(f'results_view: BOFF group order: {_boff_order}')

        parent_brush = QBrush(QColor(_THEME_LBG))
        for slot, entries in ordered_groups:
            parent = QTreeWidgetItem(self._tree)
            parent.setText(0, slot)
            # Parent rows are pure group headers — only the label and
            # child count. Single-child groups (Ship Name / Tier / Type)
            # used to mirror the child's name/confidence here, which
            # made them visually different from multi-child groups
            # (Devices, Fore Weapons, …) where the value already lives
            # in the child row. Consistent now: always read children.
            parent.setText(1, str(len(entries)))
            for col in range(self._tree.columnCount()):
                parent.setBackground(col, parent_brush)
            f = parent.font(0)
            f.setBold(True)
            for col in range(self._tree.columnCount()):
                parent.setFont(col, f)
            for it in entries:
                child = QTreeWidgetItem(parent)
                child.setText(0, '')
                child.setText(1, str(it.slot_index + 1))
                _name = it.name or '—'
                origin_badge = ''
                if getattr(it, 'match_origin', '') == 'user':
                    _name = f'✓ {_name}'
                    child.setForeground(2, QBrush(QColor('#7effc8')))
                    origin_badge = '<br><i>Match from your own WARP CORE correction (live-seed)</i>'
                child.setText(2, _name)
                child.setText(3, f'{it.confidence:.0%}')
                # Rich tooltip with reference icon
                if it.name:
                    from warp.recognition.boff_keys import pretty_slot
                    _slot_disp = pretty_slot(it.slot or '?')
                    _conf = it.confidence or 0.0
                    _col = ('#7effc8' if _conf >= 0.85 else
                            '#e8c060' if _conf >= 0.70 else '#ff9966')
                    _info = (f'<b>{_slot_disp}</b><br>{it.name}'
                             f'<br>Confidence: <span style="color:{_col}">'
                             f'{_conf:.0%}</span>{origin_badge}')
                    _tip = _tooltip_html(it.thumbnail, it.name, _info)
                    child.setToolTip(2, _tip)
                # Stash global index for canvas↔tree sync (col 0 UserRole)
                # and the resolved screenshot path (col 2 UserRole — used
                # by the right-click menu / file-tint logic). The visible
                # Source column has been removed; the path lives only as
                # data on the row now.
                gidx = self._gidx_to_file_index(it)
                child.setData(0, Qt.ItemDataRole.UserRole, gidx)
                if it.source_file:
                    try:
                        real = str(Path(it.source_file).resolve())
                    except Exception:
                        real = str(it.source_file)
                    child.setData(2, Qt.ItemDataRole.UserRole, real)
            parent.setExpanded(True)

    def _apply_file_tint(self):
        """Pale-green wash across every column on rows whose source file
        matches the currently displayed file. Cleared when the user
        toggles the file selection off (see _on_list_clicked)."""
        tint  = QBrush(_FILE_TINT)
        clear = QBrush()
        cols  = self._tree.columnCount()
        for i in range(self._tree.topLevelItemCount()):
            parent = self._tree.topLevelItem(i)
            for j in range(parent.childCount()):
                child = parent.child(j)
                src = child.data(2, Qt.ItemDataRole.UserRole)
                tinted = (self._current_file
                          and isinstance(src, str)
                          and src == self._current_file)
                for c in range(cols):
                    child.setBackground(c, tint if tinted else clear)

    def _tree_item_for_gidx(self, gidx: int) -> QTreeWidgetItem | None:
        for i in range(self._tree.topLevelItemCount()):
            parent = self._tree.topLevelItem(i)
            for j in range(parent.childCount()):
                child = parent.child(j)
                g = child.data(0, Qt.ItemDataRole.UserRole)
                if isinstance(g, int) and g == gidx:
                    return child
        return None

    def _on_tree_selection_changed(self):
        if self._syncing_highlight:
            return
        items = self._tree.selectedItems()
        if not items:
            self._canvas.set_highlight(-1)
            return
        item = items[0]
        gidx = item.data(0, Qt.ItemDataRole.UserRole)
        # Parent (group) row: highlight every child bbox that belongs to
        # the currently displayed file.
        if item.parent() is None:
            self._highlight_group(item)
            return
        if not isinstance(gidx, int) or gidx < 0:
            self._canvas.set_highlight(-1)
            return
        # Switch to the item's source file if it isn't already displayed.
        src = self._gidx_to_file[gidx] if 0 <= gidx < len(self._gidx_to_file) else ''
        if src and src != self._current_file and src in self._file_keys:
            self._syncing_highlight = True
            try:
                row = self._file_keys.index(src)
                self._list.setCurrentRow(row)
            finally:
                self._syncing_highlight = False
        self._canvas.set_highlight(gidx)

    def _highlight_group(self, parent: QTreeWidgetItem) -> None:
        """Highlight every child bbox of `parent` that belongs to the
        currently displayed screenshot. If none do, fall back to the
        first child's source file (so clicking a group from another
        image still produces a meaningful preview)."""
        gidxs_current: list[int] = []
        gidxs_other: dict[str, list[int]] = {}
        for j in range(parent.childCount()):
            child = parent.child(j)
            g = child.data(0, Qt.ItemDataRole.UserRole)
            src = child.data(2, Qt.ItemDataRole.UserRole)
            if not isinstance(g, int) or g < 0:
                continue
            if isinstance(src, str) and src == self._current_file:
                gidxs_current.append(g)
            elif isinstance(src, str):
                gidxs_other.setdefault(src, []).append(g)
        if gidxs_current:
            self._canvas.set_highlight_set(gidxs_current)
            return
        # No children on this image — switch to the file owning the
        # first child and highlight its group there.
        if gidxs_other:
            src, gidxs = next(iter(gidxs_other.items()))
            if src in self._file_keys:
                self._syncing_highlight = True
                try:
                    self._list.setCurrentRow(self._file_keys.index(src))
                finally:
                    self._syncing_highlight = False
                self._canvas.set_highlight_set(gidxs)
                return
        self._canvas.set_highlight(-1)

    def _on_canvas_bbox_clicked(self, gidx: int):
        self._canvas.set_highlight(gidx)
        if gidx < 0:
            self._tree.clearSelection()
            return
        tree_item = self._tree_item_for_gidx(gidx)
        if tree_item is None:
            return
        self._syncing_highlight = True
        try:
            self._tree.setCurrentItem(tree_item)
            self._tree.scrollToItem(tree_item)
        finally:
            self._syncing_highlight = False

    # ── Context menu (shared) ───────────────────────────────────────

    def _on_list_context_menu(self, pos: QPoint):
        item = self._list.itemAt(pos)
        if item is None:
            return
        row = self._list.row(item)
        if not (0 <= row < len(self._file_keys)):
            return
        src = self._file_keys[row]
        self._show_context_menu(
            self._list.viewport().mapToGlobal(pos), src)

    def _on_tree_context_menu(self, pos: QPoint):
        item = self._tree.itemAt(pos)
        if item is None:
            return
        src = self._resolve_item_source(item)
        ri = self._resolve_item_recognised(item)
        self._show_context_menu(
            self._tree.viewport().mapToGlobal(pos), src, ri)

    def _on_canvas_context_menu(self, global_pos: QPoint, gidx: int):
        src = self._current_file
        ri: RecognisedItem | None = None
        if gidx >= 0 and 0 <= gidx < len(self._gidx_to_file):
            src = self._gidx_to_file[gidx] or src
        if self._result and gidx >= 0 and gidx < len(self._result.items):
            ri = self._result.items[gidx]
        self._show_context_menu(global_pos, src, ri)

    def _resolve_item_source(self, item: QTreeWidgetItem) -> str:
        own = item.data(2, Qt.ItemDataRole.UserRole)
        if isinstance(own, str) and own:
            return own
        for i in range(item.childCount()):
            child_src = item.child(i).data(2, Qt.ItemDataRole.UserRole)
            if isinstance(child_src, str) and child_src:
                return child_src
        return ''

    def _resolve_item_recognised(self, item: QTreeWidgetItem) -> RecognisedItem | None:
        """Return the RecognisedItem for a leaf tree row, or None."""
        gidx = item.data(0, Qt.ItemDataRole.UserRole)
        if not isinstance(gidx, int) or gidx < 0:
            return None
        if self._result and gidx < len(self._result.items):
            return self._result.items[gidx]
        return None

    def _show_context_menu(self, global_pos: QPoint, src: str,
                           ri: RecognisedItem | None = None):
        name = Path(src).name if src else ''
        st = self.style()
        icon_copy = st.standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton)
        icon_link = st.standardIcon(QStyle.StandardPixmap.SP_FileLinkIcon)
        icon_open = st.standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon)
        icon_fast = st.standardIcon(QStyle.StandardPixmap.SP_BrowserReload)

        menu = QMenu(self)
        if name:
            header = menu.addAction(name)
            header.setEnabled(False)
            f = header.font()
            f.setBold(True)
            header.setFont(f)
            menu.addSeparator()

        act_copy_name = menu.addAction(icon_copy, 'Copy filename')
        act_copy_path = menu.addAction(icon_link, 'Copy full path')
        if not src:
            act_copy_name.setEnabled(False)
            act_copy_path.setEnabled(False)

        act_open_core = None
        act_open_fast = None
        if src and self._has_warp_core_handler:
            menu.addSeparator()
            act_open_core = menu.addAction(icon_open, 'Open in WARP CORE')
        if src and self._has_fast_correction_handler and self._result \
                and self._result.items:
            if act_open_core is None:
                menu.addSeparator()
            act_open_fast = menu.addAction(
                icon_fast, 'Open in WARP Fast Correction Mode')

        # --- external-link actions for a specific item ---
        act_vger = None
        act_wiki = None
        if ri and ri.name:
            from warp.data.cargo import wiki_url, vger_url as _vger_url
            menu.addSeparator()
            v_url = _vger_url(ri.slot)
            if v_url:
                act_vger = menu.addAction('Open on vger.stobuilds.com')
                act_vger.setData(v_url)
            act_wiki = menu.addAction('Open on STO Wiki')
            act_wiki.setData(wiki_url(ri.name))

        chosen = menu.exec(global_pos)
        if chosen is None:
            return
        if chosen is act_copy_name and src:
            QApplication.clipboard().setText(name)
        elif chosen is act_copy_path and src:
            QApplication.clipboard().setText(src)
        elif chosen is act_open_core and src:
            items_for_src = self._items_by_file.get(src, [])
            self.open_in_warp_core.emit(src, list(items_for_src))
        elif chosen is act_open_fast:
            # Fast Correction sends ALL files from the current result,
            # not just the right-clicked one. The second dict carries
            # WARP's per-file screen_type so the trainer doesn't have to
            # re-classify (and doesn't fall back to TDM which holds the
            # user's *previous* labels — exactly what we want to fix).
            self.open_in_warp_fast_corr.emit(
                {k: list(v) for k, v in self._items_by_file.items()},
                dict(self._stype_by_file))
        elif chosen is act_vger or chosen is act_wiki:
            from PySide6.QtGui import QDesktopServices
            from PySide6.QtCore import QUrl
            QDesktopServices.openUrl(QUrl(chosen.data()))
