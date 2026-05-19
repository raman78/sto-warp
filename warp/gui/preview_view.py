"""Read-only detection preview: left = source file list, right = screenshot
with bbox overlay coloured per slot family.

Used as the second tab in ``WarpWindow`` after a recognition run finishes.
Pulls coordinates straight from ``ImportResult.items[*].bbox``; does not
re-run any detection. Items without a bbox (e.g. boff abilities that fell
through to virtual placeholders) are skipped silently.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout, QListWidget, QListWidgetItem, QScrollArea, QSplitter,
    QWidget,
)

from warp.warp_importer import ImportResult, RecognisedItem


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
    # fallback — deterministic hue derived from the slot string.
    h = (sum(ord(c) for c in slot) * 17) % 360
    return QColor.fromHsv(h, 200, 240)


class _ImageCanvas(QWidget):
    """Paints one screenshot with its bbox overlay. Auto-fits to viewport."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
        self._scaled: QPixmap | None = None
        self._items: list[RecognisedItem] = []
        self._build_type: str = ''
        self._scale: float = 1.0
        self.setMinimumSize(200, 200)

    def set_image(self, path: Path, items: list[RecognisedItem],
                  build_type: str = '') -> None:
        pm = QPixmap(str(path)) if path.is_file() else QPixmap()
        self._pixmap = pm if not pm.isNull() else None
        self._items = list(items)
        self._build_type = build_type or ''
        self._scaled = None
        self._compute_fit()
        self.update()

    def clear(self) -> None:
        self._pixmap = None
        self._scaled = None
        self._items = []
        self._build_type = ''
        self.update()

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

    def paintEvent(self, _e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(30, 30, 30))
        if self._pixmap is None:
            p.setPen(QColor(160, 160, 160))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       '(no preview)')
            return

        if self._scaled is None:
            self._scaled = self._pixmap.scaled(
                int(self._pixmap.width()  * self._scale),
                int(self._pixmap.height() * self._scale),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

        x0 = (self.width()  - self._scaled.width())  // 2
        y0 = (self.height() - self._scaled.height()) // 2
        p.drawPixmap(x0, y0, self._scaled)

        # Build-type badge in the upper-left of the rendered image so the
        # user can see what screen type the autodetector picked, including
        # cases where no slots were detected.
        if self._build_type:
            badge = self._build_type
            bf = QFont('Monospace')
            bf.setPointSize(10)
            bf.setBold(True)
            p.setFont(bf)
            fm = p.fontMetrics()
            pad_x, pad_y = 8, 4
            tw = fm.horizontalAdvance(badge)
            th = fm.height()
            bx = x0 + 6
            by = y0 + 6
            p.fillRect(bx, by, tw + pad_x * 2, th + pad_y * 2,
                       QColor(0, 0, 0, 180))
            p.setPen(QColor(220, 220, 220))
            p.drawText(bx + pad_x, by + pad_y + fm.ascent(), badge)

        f = QFont('Monospace')
        f.setPointSize(8)
        p.setFont(f)
        for it in self._items:
            if not it.bbox or len(it.bbox) < 4:
                continue
            x, y, w, h = it.bbox[:4]
            rx = x0 + int(x * self._scale)
            ry = y0 + int(y * self._scale)
            rw = max(1, int(w * self._scale))
            rh = max(1, int(h * self._scale))
            color = _color_for_slot(it.slot)
            p.setPen(QPen(color, 2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(rx, ry, rw, rh)
            label = it.slot or ''
            if it.confidence is not None:
                label = f'{label}  {it.confidence:.2f}'
            p.setPen(QPen(QColor(0, 0, 0, 200), 3))
            p.drawText(rx + 2, max(ry - 2, 10), label)
            p.setPen(color)
            p.drawText(rx + 2, max(ry - 2, 10), label)


class PreviewView(QWidget):
    """Detection overlay tab — purely consumes ``ImportResult``."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._items_by_file: dict[str, list[RecognisedItem]] = {}
        self._bt_by_file:    dict[str, str] = {}
        self._keys: list[str] = []
        self._build_ui()

    # ── UI ──────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)

        split = QSplitter(Qt.Orientation.Horizontal, self)

        self._list = QListWidget(self)
        self._list.setMinimumWidth(220)
        self._list.itemSelectionChanged.connect(self._on_file_selected)
        split.addWidget(self._list)

        self._scroll = QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._canvas = _ImageCanvas(self._scroll)
        self._scroll.setWidget(self._canvas)
        split.addWidget(self._scroll)
        split.setSizes([240, 860])

        root.addWidget(split)

    # ── Public API ──────────────────────────────────────────────────

    def clear(self) -> None:
        self._items_by_file = {}
        self._bt_by_file = {}
        self._keys = []
        self._list.clear()
        self._canvas.clear()

    def set_result(self, result: ImportResult) -> None:
        self.clear()
        by_file: dict[str, list[RecognisedItem]] = {}
        for it in result.items:
            if not it.source_file:
                continue
            # Resolve up-front so previews survive the staging tempdir
            # being deleted after the recognition thread finishes.
            try:
                key = str(Path(it.source_file).resolve())
            except OSError:
                key = it.source_file
            by_file.setdefault(key, []).append(it)
        self._items_by_file = by_file
        self._bt_by_file = dict(getattr(result, 'per_file', {}) or {})
        # Union: every file the importer touched + every file that emitted
        # an item. Files with zero items still appear so the user can
        # confirm the image actually loaded and see the picked screen type.
        self._keys = sorted(set(by_file) | set(self._bt_by_file))
        for src in self._keys:
            items = by_file.get(src, [])
            with_bbox = sum(1 for it in items
                            if it.bbox and len(it.bbox) >= 4)
            bt = self._bt_by_file.get(src, '')
            bt_tag = f'  [{bt}]' if bt else ''
            QListWidgetItem(
                f'{Path(src).name}   ({with_bbox}/{len(items)}){bt_tag}',
                self._list,
            )
        if self._list.count():
            self._list.setCurrentRow(0)

    # ── Internals ───────────────────────────────────────────────────

    def _on_file_selected(self):
        row = self._list.currentRow()
        if row < 0 or row >= len(self._keys):
            self._canvas.clear()
            return
        src = self._keys[row]
        self._canvas.set_image(
            Path(src),
            self._items_by_file.get(src, []),
            self._bt_by_file.get(src, ''),
        )
