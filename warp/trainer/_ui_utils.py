# warp/trainer/_ui_utils.py
# WARP CORE UI helpers — small Qt utility widgets, dot-icons, and the
# per-image match-summary table. Extracted from trainer_window.py during
# the Phase-0 refactor so workers + window mixins can share them without
# pulling in the giant WarpCoreWindow module.

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QPalette
from PySide6.QtWidgets import (
    QAbstractItemView, QHeaderView, QStyledItemDelegate, QTreeWidget,
    QTreeWidgetItem,
)

from warp import userdata


class _ColorPreservingDelegate(QStyledItemDelegate):
    """Keep item's ForegroundRole color visible even when the row is selected."""
    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        brush = index.data(Qt.ItemDataRole.ForegroundRole)
        if isinstance(brush, QBrush) and brush.color().isValid():
            option.palette.setColor(QPalette.ColorRole.HighlightedText, brush.color())


class _ReviewListAdapter(QTreeWidget):
    """5-column grouped QTreeWidget exposing the QListWidget-style API
    used by the WARP CORE recognition-review panel.

    Visual layout mirrors WARP's Results view: top-level parent rows are
    slot headers (bold, with `(count)` in the Idx column); each
    recognition item is a child row. For single-item slots the parent
    mirrors the child's Item / Conf / Status so a collapsed tree still
    reads as a one-liner.

    Pre-refactor the panel was a QListWidget; the upgrade to a 5-column
    Slot / Idx / Item / Conf [%] / Status grid would touch ~30 call sites
    if we used QTreeWidget directly. This adapter keeps `addItem`,
    `item(N)`, `count()`, `currentRow()`, `setCurrentRow(N)`, `takeItem`,
    `insertItem`, `row(item)`, plus a `currentRowChanged(int)` signal so
    the existing callers continue to work unchanged. The flat `N` index
    refers to insertion order (== order of `_recognition_items`), not the
    tree's visual order — so callers that pair the panel with the
    recognition-items list stay aligned even when the tree groups them
    under different parents.
    """

    currentRowChanged = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(5)
        self.setHeaderLabels(['Slot', 'Idx', 'Item', 'Conf', 'Status'])
        # stretchLastSection defaults to True and silently overrides the
        # explicit Status width + steals the divider grab-zone on its
        # left edge. Turn it off so each column has a real, draggable
        # divider and the column widths below are actually honoured.
        h = self.header()
        h.setStretchLastSection(False)
        # All columns Interactive — Qt's Stretch mode silently fixes the
        # section's width to the calculated stretch-fill and refuses to
        # let the user drag the divider on either side of it, which is
        # what made the Item↔Conf boundary unresponsive. Without any
        # Stretch column the tree may leave trailing whitespace if the
        # panel is wider than the sum of column widths — acceptable in
        # exchange for fully resizable boundaries.
        for c in range(5):
            h.setSectionResizeMode(c, QHeaderView.ResizeMode.Interactive)
        self.setAlternatingRowColors(False)
        self.setRootIsDecorated(True)
        self.setUniformRowHeights(True)
        self.setColumnWidth(0, 150)
        self.setColumnWidth(1, 40)
        self.setColumnWidth(2, 240)
        self.setColumnWidth(3, 56)   # fits '100%'
        self.setColumnWidth(4, 86)   # fits 'Confirmed'
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)

        self._flat: list[QTreeWidgetItem] = []           # insertion order
        self._slot_parents: dict[str, QTreeWidgetItem] = {}

        # Translate Qt's (current, previous) into the row-int signal
        # callers from the old QListWidget era expect.
        self.currentItemChanged.connect(self._on_current_item_changed)
        # Bold the selected leaf row so the active pick stands out
        # against the ACCENT-tinted selection background.
        self.itemSelectionChanged.connect(self._refresh_bold_selected)

    # ── Internals ───────────────────────────────────────────────────

    def _on_current_item_changed(self, current, _previous):
        if current is None or current not in self._flat:
            self.currentRowChanged.emit(-1)
            return
        self.currentRowChanged.emit(self._flat.index(current))

    def _refresh_bold_selected(self):
        sel = set(self.selectedItems())
        cols = self.columnCount()
        for it in self._flat:
            bold = it in sel
            f = it.font(0)
            if f.bold() == bold:
                continue
            f.setBold(bold)
            for c in range(cols):
                it.setFont(c, f)

    def _get_or_create_parent(self, slot_raw: str,
                              slot_pretty: str) -> QTreeWidgetItem:
        if slot_raw in self._slot_parents:
            return self._slot_parents[slot_raw]
        p = QTreeWidgetItem(self)
        p.setText(0, slot_pretty)
        p.setData(0, Qt.ItemDataRole.UserRole, slot_raw)
        # Parents are not user-selectable — they're headers, not items.
        p.setFlags(p.flags() & ~Qt.ItemFlag.ItemIsSelectable)
        f = p.font(0)
        f.setBold(True)
        for c in range(self.columnCount()):
            p.setFont(c, f)
        self._slot_parents[slot_raw] = p
        return p

    def _drop_parent_if_empty(self, parent: QTreeWidgetItem) -> None:
        if parent.childCount() != 0:
            return
        slot_raw = parent.data(0, Qt.ItemDataRole.UserRole)
        idx = self.indexOfTopLevelItem(parent)
        if idx >= 0:
            self.takeTopLevelItem(idx)
        self._slot_parents.pop(slot_raw, None)

    def refresh_parent_of(self, item: QTreeWidgetItem) -> None:
        """Public helper — call after mutating a child so the parent's
        Idx column stays in sync. Idempotent.

        Parent rows are pure group headers: they show only the group
        label and the child count. Item / Conf / Status columns stay
        empty so single-child groups (Ship Name / Type / Tier) read the
        same as multi-child ones (Devices, Fore Weapons, …) — the user
        always finds the value in the child row.
        """
        parent = item.parent()
        if parent is None:
            return
        n = parent.childCount()
        parent.setText(1, str(n) if n > 1 else '')
        parent.setText(2, '')
        parent.setText(3, '')
        parent.setText(4, '')
        blank = QBrush()
        for c in range(self.columnCount()):
            parent.setForeground(c, blank)

    # ── QListWidget-style shims ─────────────────────────────────────

    def addItem(self, item):
        slot_raw    = item.data(0, Qt.ItemDataRole.UserRole) or ''
        slot_pretty = item.text(0)
        parent      = self._get_or_create_parent(slot_raw, slot_pretty)
        # Children leave the Slot column blank — the parent owns it.
        item.setText(0, '')
        parent.addChild(item)
        parent.setExpanded(True)
        self._flat.append(item)
        self.refresh_parent_of(item)

    def insertItem(self, row, item):
        slot_raw    = item.data(0, Qt.ItemDataRole.UserRole) or ''
        slot_pretty = item.text(0)
        parent      = self._get_or_create_parent(slot_raw, slot_pretty)
        item.setText(0, '')
        parent.addChild(item)
        parent.setExpanded(True)
        if row < 0 or row > len(self._flat):
            row = len(self._flat)
        self._flat.insert(row, item)
        self.refresh_parent_of(item)

    def takeItem(self, row):
        if not (0 <= row < len(self._flat)):
            return None
        item   = self._flat.pop(row)
        parent = item.parent()
        if parent is not None:
            parent.removeChild(item)
            if parent.childCount() == 0:
                self._drop_parent_if_empty(parent)
            else:
                self.refresh_parent_of(parent.child(0))
        return item

    def item(self, row):
        if 0 <= row < len(self._flat):
            return self._flat[row]
        return None

    def count(self):
        return len(self._flat)

    def currentRow(self):
        cur = self.currentItem()
        if cur in self._flat:
            return self._flat.index(cur)
        return -1

    def setCurrentRow(self, row):
        if 0 <= row < len(self._flat):
            self.setCurrentItem(self._flat[row])
        else:
            self.setCurrentItem(None)

    def row(self, item):
        if item in self._flat:
            return self._flat.index(item)
        return -1

    def clear(self):
        super().clear()
        self._flat.clear()
        self._slot_parents.clear()


# ── Match summary table + history ──────────────────────────────────────
_RECOG_HISTORY_PATH = userdata.training_data_dir() / 'recog_history.json'
_DELTA_EPS = 0.03  # minimum absolute conf change to render an arrow


def _arrow(prev: float | None, curr: float) -> str:
    if prev is None:
        return ' new'
    d = curr - prev
    if abs(d) < _DELTA_EPS:
        return '  ─ '
    return f'{"↑" if d > 0 else "↓"}{abs(d):.2f}'


def _fmt_score(v: float) -> str:
    return f'{v:.2f}' if v > 0 else ' -  '


def _load_recog_history() -> dict:
    try:
        if _RECOG_HISTORY_PATH.exists():
            import json
            return json.loads(_RECOG_HISTORY_PATH.read_text(encoding='utf-8'))
    except Exception:
        pass
    return {}


def _save_recog_history(hist: dict) -> None:
    try:
        import json
        _RECOG_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        _RECOG_HISTORY_PATH.write_text(
            json.dumps(hist, indent=2), encoding='utf-8')
    except Exception:
        pass


def _log_match_summary(image_name: str, match_log: list[dict]) -> None:
    """
    Render the per-image match summary table and update recog_history.json.

    Columns: slot | name | win | embed | sess | tmpl | knldg | Δ (vs previous
    run for the same image+slot). Totals per source at the bottom.
    """
    from warp.debug import log as _slog
    if not match_log:
        return

    hist  = _load_recog_history()
    prev  = hist.get(image_name, {})
    curr: dict[str, dict] = {}
    totals: dict[str, int] = {}

    header = (f'{"slot":<28} {"name":<32} {"win":<6} '
              f'{"embed":>6} {"sess":>6} {"tmpl":>6} {"knldg":>6}   Δ')
    rows: list[str] = [header, '-' * len(header)]

    # When the same slot appears multiple times (e.g. Boff Tactical row),
    # disambiguate by appending index. recog_history is keyed by these labels.
    slot_seen: dict[str, int] = {}
    for entry in match_log:
        slot = entry.get('slot', '')
        idx  = slot_seen.get(slot, 0)
        slot_seen[slot] = idx + 1
        key  = slot if idx == 0 else f'{slot}#{idx}'

        name   = entry.get('name', '') or ''
        src    = entry.get('src',  '') or ''
        stages = entry.get('stages', {}) or {}
        e = float(stages.get('embed',     0.0))
        f = float(stages.get('soft',      0.0))
        s = float(stages.get('session',   0.0))
        t = float(stages.get('template',  0.0))
        k = float(stages.get('knowledge', 0.0))

        prev_conf = prev.get(key, {}).get('conf')
        arrow = _arrow(prev_conf, float(entry.get('conf', 0.0)))
        # Track the active ML score (embed or soft, whichever the matcher used).
        ml_score = e if e > 0 else f
        rows.append(
            f'{key[:28]:<28} {name[:32]:<32} {src[:6]:<6} '
            f'{_fmt_score(ml_score):>6} {_fmt_score(s):>6} '
            f'{_fmt_score(t):>6} {_fmt_score(k):>6}   {arrow}'
        )
        totals[src] = totals.get(src, 0) + 1
        curr[key] = {'name': name, 'conf': float(entry.get('conf', 0.0)),
                     'src':  src}

    totals_str = '  '.join(f'{src or "?"}={cnt}'
                            for src, cnt in sorted(totals.items()))
    _slog.info(f'WARP CORE: match summary  {image_name}  ({len(match_log)} items)')
    for r in rows:
        _slog.info(f'  {r}')
    _slog.info(f'  TOTAL: {totals_str}')

    hist[image_name] = curr
    _save_recog_history(hist)


# ── Dot icons (green = user confirmed, yellow = ML auto) ───────────────
def _make_dot_icon(color: str) -> 'QIcon':
    """Small 14×14 filled circle icon for green/yellow confirmation state."""
    from PySide6.QtCore import Qt as _Qt
    from PySide6.QtGui import QPixmap, QPainter, QColor, QIcon
    pix = QPixmap(14, 14)
    pix.fill(_Qt.GlobalColor.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setBrush(QColor(color))
    p.setPen(_Qt.PenStyle.NoPen)
    p.drawEllipse(1, 1, 12, 12)
    p.end()
    return QIcon(pix)


_ICON_USER_CONFIRMED: 'QIcon | None' = None   # green — created lazily
_ICON_ML_AUTO:        'QIcon | None' = None   # yellow — created lazily


def _get_user_icon() -> 'QIcon':
    global _ICON_USER_CONFIRMED
    if _ICON_USER_CONFIRMED is None:
        _ICON_USER_CONFIRMED = _make_dot_icon('#44dd66')
    return _ICON_USER_CONFIRMED


def _get_ml_icon() -> 'QIcon':
    global _ICON_ML_AUTO
    if _ICON_ML_AUTO is None:
        _ICON_ML_AUTO = _make_dot_icon('#ffcc00')
    return _ICON_ML_AUTO
