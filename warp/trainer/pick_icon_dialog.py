"""Pick the right item for a crop from its closest look-alikes.

When recognition names a slot wrongly, the fix used to be opening vger in a
browser and comparing icons by eye. This dialog puts the crop next to the
items that look most like it — restricted to what the slot can hold — and
lets a name search list the whole group when none of them is right.

It holds no recognition logic: the caller hands in the ranking
(`SETSIconMatcher.rank_candidates`) and a way to fetch each item's picture,
so the dialog can be tested without models.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from PySide6.QtCore import QByteArray, QSettings, QSize, Qt
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtWidgets import (QDialog, QHBoxLayout, QLabel, QLineEdit,
                               QListWidget, QListWidgetItem, QPushButton,
                               QVBoxLayout)

PAGE = 30          # tiles shown before "Show more"
GEOMETRY_KEY = 'pick_icon_dialog/geometry'
CROP_SCALE = 4     # the crop and every tile are shown at this scale


def _crop_pixmap(crop_bgr: np.ndarray) -> QPixmap:
    import cv2
    rgb = np.ascontiguousarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    h, w = rgb.shape[:2]
    img = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(img).scaled(
        w * CROP_SCALE, h * CROP_SCALE, Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.FastTransformation)


def matches(name: str, query: str) -> bool:
    """Every word of the query appears in the name, case-insensitive."""
    low = name.lower()
    return all(word in low for word in query.lower().split())


class PickIconDialog(QDialog):
    """`ranked` is [(name, score)] best first; `picture(name)` returns a
    QImage or None. After `exec()` returns Accepted, `chosen` holds the name."""

    def __init__(self, crop_bgr: np.ndarray, slot: str,
                 ranked: list[tuple[str, float]],
                 picture: Callable[[str], object],
                 current: str = '', parent=None):
        super().__init__(parent)
        self.setWindowTitle(f'Pick the item — {slot}')
        # A dialog gets no maximise button by default; this one is a
        # workspace, so it has one and remembers how it was left.
        self.setWindowFlags(self.windowFlags()
                            | Qt.WindowType.WindowMaximizeButtonHint
                            | Qt.WindowType.WindowMinimizeButtonHint)
        self.chosen = ''
        self._ranked = ranked
        self._picture = picture
        self._pictures: dict[str, QIcon] = {}
        self._limit = PAGE
        self._slot = slot
        # Tiles are drawn at the crop's own on-screen size, so the two are
        # compared like for like. Qt never enlarges an icon past its pixmap,
        # so each picture is scaled here rather than left to the view.
        h, w = crop_bgr.shape[:2]
        self._tile = QSize(w * CROP_SCALE, h * CROP_SCALE)

        crop = QLabel()
        crop.setPixmap(_crop_pixmap(crop_bgr))
        crop.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        cur = QLabel(f'Now: {current}' if current else 'Now: not recognised')
        cur.setWordWrap(True)
        cur.setMaximumWidth(crop.pixmap().width() + 20)
        left = QVBoxLayout()
        left.addWidget(QLabel('This crop'))
        left.addWidget(crop)
        left.addWidget(cur)
        left.addStretch(1)

        self._search = QLineEdit()
        self._search.setPlaceholderText(
            'Search by name — lists every item this slot can hold')
        self._search.textChanged.connect(self._refill)
        self._grid = QListWidget()
        self._grid.setViewMode(QListWidget.ViewMode.IconMode)
        self._grid.setIconSize(self._tile)
        # Room for three lines under the picture: the percentage leads, so a
        # long name that has to be cut never takes the score with it.
        self._grid.setGridSize(QSize(self._tile.width() + 40, self._tile.height() + 64))
        self._grid.setResizeMode(QListWidget.ResizeMode.Adjust)
        self._grid.setWordWrap(True)
        self._grid.setMovement(QListWidget.Movement.Static)
        # A changed background alone was hard to see on dark icons.
        # The caption keeps its normal colour when selected: with only the
        # border set, the selection text colour made it vanish.
        from warp.style import ACCENT, FG
        self._grid.setStyleSheet(
            'QListWidget::item { border: 3px solid transparent; border-radius: 4px; }'
            f'QListWidget::item:selected {{ border: 3px solid {ACCENT}; '
            f'background: transparent; color: {FG}; }}')
        self._grid.itemDoubleClicked.connect(lambda _it: self._use_selected())
        self._grid.currentItemChanged.connect(
            lambda it, _prev: self._use.setEnabled(it is not None))
        self._count = QLabel()
        self._more = QPushButton(f'Show {PAGE} more')
        self._more.clicked.connect(self._show_more)
        # When none of the closest is right and the name is not known either:
        # the whole group this slot can hold, like the category page on vger.
        self._all = QPushButton(f'Show all {len(ranked)} for {slot}')
        self._all.clicked.connect(self._show_all)
        self._use = QPushButton('Use selected')
        self._use.setDefault(True)
        self._use.setEnabled(False)
        self._use.clicked.connect(self._use_selected)
        cancel = QPushButton('Cancel')
        cancel.clicked.connect(self.reject)

        bottom = QHBoxLayout()
        bottom.addWidget(self._count)
        bottom.addWidget(self._more)
        bottom.addWidget(self._all)
        bottom.addStretch(1)
        bottom.addWidget(self._use)
        bottom.addWidget(cancel)
        right = QVBoxLayout()
        right.addWidget(self._search)
        right.addWidget(self._grid, 1)
        right.addLayout(bottom)

        root = QHBoxLayout(self)
        root.addLayout(left)
        root.addLayout(right, 1)
        geom = QSettings().value(GEOMETRY_KEY)
        if isinstance(geom, QByteArray) and not geom.isEmpty():
            self.restoreGeometry(geom)          # size, position, maximised
        else:
            self.setWindowState(Qt.WindowState.WindowMaximized)
        self._refill()

    # ── contents ────────────────────────────────────────────────────────────

    def shown_names(self) -> list[str]:
        """What the grid lists now: the closest `PAGE`s with no search, every
        match of the search otherwise — both in similarity order."""
        query = self._search.text().strip()
        if query:
            return [n for n, _s in self._ranked if matches(n, query)]
        return [n for n, _s in self._ranked[:self._limit]]

    def _icon(self, name: str) -> QIcon:
        if name not in self._pictures:
            img = self._picture(name)
            if isinstance(img, QImage) and not img.isNull():
                pm = QPixmap.fromImage(img).scaled(
                    self._tile, Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation)
                icon = QIcon(pm)
                # Qt tints a selected icon; the frame marks the choice, and
                # the picture has to stay comparable with the crop.
                icon.addPixmap(pm, QIcon.Mode.Selected)
                self._pictures[name] = icon
            else:
                self._pictures[name] = QIcon()
        return self._pictures[name]

    def _refill(self, *_args) -> None:
        score = dict(self._ranked)
        self._grid.clear()
        for name in self.shown_names():
            it = QListWidgetItem(self._icon(name), f'{score[name]:.0%}  {name}')
            it.setData(Qt.ItemDataRole.UserRole, name)
            it.setToolTip(f'{name} — similarity {score[name]:.0%}')
            self._grid.addItem(it)
        searching = bool(self._search.text().strip())
        total = len(self._ranked)
        self._count.setText(
            f'{self._grid.count()} match(es) of {total}' if searching
            else f'Closest {self._grid.count()} of {total}')
        self._more.setVisible(not searching and self._limit < total)
        self._all.setVisible(not searching and self._limit < total)
        self._use.setEnabled(False)

    def _show_more(self) -> None:
        self._limit += PAGE
        self._refill()

    def _show_all(self) -> None:
        self._limit = len(self._ranked)
        self._refill()

    def done(self, result: int) -> None:
        """Every way out — a pick, Cancel, the close button — keeps the
        size and position for next time."""
        QSettings().setValue(GEOMETRY_KEY, self.saveGeometry())
        super().done(result)

    def _use_selected(self) -> None:
        it = self._grid.currentItem()
        if it is None:
            return
        self.chosen = it.data(Qt.ItemDataRole.UserRole)
        self.accept()
