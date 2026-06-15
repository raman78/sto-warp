# warp/trainer/trainer_window.py
# WARP CORE — Interactive ML trainer + recognition review.
# PySide6, integrated with SETS.

from __future__ import annotations

import logging
from pathlib import Path
import json
import datetime

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QLabel, QPushButton, QListWidget, QListWidgetItem,
    QFileDialog, QComboBox, QLineEdit, QGroupBox,
    QProgressBar, QToolBar, QStatusBar, QMessageBox,
    QInputDialog, QSizePolicy, QFrame, QScrollArea,
    QAbstractItemView, QCompleter, QMenu, QPlainTextEdit,
    QCheckBox, QDoubleSpinBox, QTabWidget, QTreeWidgetItem,
)
from PySide6.QtCore import Qt, QSettings, QSortFilterProxyModel, QSize, QTimer, Signal
from PySide6.QtGui import QFont, QAction, QBrush, QColor, QIcon, QStandardItemModel, QStandardItem, QKeySequence, QShortcut


from warp import userdata
from warp.trainer.annotation_widget import AnnotationWidget
from warp.trainer.fast_session      import display_name as _disp_name
from warp.trainer.training_data      import (
    TrainingDataManager, AnnotationState, NON_ICON_SLOTS, SINGLE_INSTANCE_SLOTS,
    TEXT_LEARNING_SLOTS, VIRTUAL_ITEM_NAMES,
)
from warp.style import (
    apply_dark_style, primary_btn_style, secondary_btn_style,
    warning_btn_style, danger_btn_style, toggle_yellow_btn_style,
    accent_qss,
    ACCENT, FG, MFG, BG, MBG, LBG, BC, C_WARNING, C_SUCCESS, C_FAILURE,
)
# Phase-0 refactor extracted shared constants, UI helpers, and QThread
# workers into focused modules. trainer_window.py now imports the names
# it still uses internally; nothing external imports them from here.
from warp.trainer.constants import (
    _KEY_LAST_DIR, _KEY_AUTO_ACCEPT, _KEY_AUTO_CONF,
    CONF_HIGH, CONF_MEDIUM,
    SLOT_GROUPS, SCREEN_TYPE_LABELS, SCREEN_TYPE_ICONS, SCREEN_TO_SLOT_GROUP,
    FIXED_VALUE_SLOTS, ALL_SLOTS, SPECIALIZATION_NAMES, _SHIP_INFO_SLOTS,
)
from warp.trainer._ui_utils import (
    _ColorPreservingDelegate, _ReviewListAdapter,
    _get_user_icon, _get_ml_icon, _log_match_summary,
)
from warp.trainer.workers import (
    ScreenTypeDetectorWorker, OCRWorker, MatchWorker, RecognitionWorker,
)
from warp.recognition.text_extractor import SHIP_TIER_VALUES
from warp.recognition.boff_keys      import pretty_slot as _pretty_slot
from warp.warp_importer              import SLOT_ORDER as _SLOT_ORDER

log = logging.getLogger(__name__)


class _DebugDict(dict):
    """Thin dict wrapper that logs every mutation with a traceback.

    Temporary instrumentation to find the code path that evicts
    _recognition_cache keys during Fast Correction Mode.
    """

    def __init__(self, tag: str, *a, **kw):
        super().__init__(*a, **kw)
        self._tag = tag

    def _log(self, op: str, key=None):
        import traceback
        from warp.debug import log as _wlog
        stack = ''.join(traceback.format_stack(limit=8)[:-1])
        _wlog.info(f'[DEBUG-DICT] {self._tag}.{op} key={key!r} '
                   f'keys_after={list(self.keys())}\n{stack}')

    def __delitem__(self, key):
        super().__delitem__(key)
        self._log('__delitem__', key)

    def pop(self, key, *default):
        had = key in self
        result = super().pop(key, *default)
        if had:
            self._log('pop', key)
        return result

    def clear(self):
        keys_before = list(self.keys())
        super().clear()
        if keys_before:
            self._log(f'clear (was {keys_before})')

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        # Only log if in FC mode (avoid noise during normal training)
        # self._log('__setitem__', key)


class WarpCoreWindow(QMainWindow):
    # Emitted when the user presses "↗ Send to WARP" on a screenshot
    # marked Done. Payload is an ImportResult built from the user-confirmed
    # annotations for the current screenshot. Launcher wires this to
    # WarpWindow.set_external_result + tab switch so the user can JSON-export
    # the corrected build without re-running WARP detection.
    send_to_warp = Signal(object)
    # Emitted by `exit_fast_correction_mode()` so the launcher can restore
    # its tab title and (optionally) switch back to the WARP tab.
    fast_correction_exited = Signal()

    def __init__(self, sets_app=None, parent=None, embed: bool = False):
        super().__init__(parent)
        # Standalone sto-warp: synthesize a cargo-backed SETS-app shim
        # whenever the caller doesn't hand in a real SETS app. The shim
        # exposes `.cache` (cargo view) plus a settable `_warp_core_window`
        # attribute used by `SyncManager._data_manager`.
        if sets_app is None:
            from warp.data.cargo import app_view
            sets_app = app_view()
            sets_app._warp_core_window = self
        else:
            # Always make sure the live trainer window is reachable from
            # the shared app shim — SyncManager looks it up by attribute.
            try:
                sets_app._warp_core_window = self
            except Exception:
                pass
        self._sets = sets_app
        self._embed = embed
        self._settings = QSettings()
        self._sets_root = self._find_sets_root()
        self._data_mgr = TrainingDataManager(userdata.training_data_dir())
        self._screenshots: list[Path] = []
        self._current_idx = -1
        # Window mode: 'training' (default) or 'fast_correction'. Fast
        # Correction Mode is launched from WARP's Results pane to fix up
        # a fixed batch of files; folder-load + Open toolbar entries are
        # suppressed and a banner with an Exit button is shown.
        self._mode: str = 'training'
        # Populated by `set_fast_correction_mode` — holds the active
        # FastSession (staging dir + staged→orig path map) while the
        # window is in Fast Correction Mode; None otherwise.
        self._fast_session = None
        self._screen_types: dict[str, str] = {}
        self._screen_types_manual: set[str] = set()   # green — user confirmed
        self._screen_types_ml_auto: set[str] = set()  # yellow — ML ≥95% auto-accepted
        self._screenshots_done: set[str] = self._load_done()  # fully annotated, locked
        self._detect_trigger: str = 'unknown'
        self._recognition_cache: dict[str, list] = _DebugDict('_recognition_cache')
        self._recognition_items: list[dict] = []
        self._manual_bbox_mode = False
        self._add_bbox_mode = False
        self._loading_row = False
        self._sync_client = None
        self._sync_timer = None
        if not self._embed:
            self._init_sync_client()
        self._detect_worker = None
        self._suppress_next_focus_popup = False  # set True after programmatic setFocus
        self._recog_worker = None
        self._detect_dlg = None
        self._detect_loop_max: int = 1
        self._detect_loop_iter: int = 0
        self._detect_loop_prev_unresolved: int | None = None
        self._recog_dlg = None
        self._selection_just_changed = False
        self.setWindowTitle('WARP CORE — ML Trainer')
        self.setMinimumSize(1280, 740)
        apply_dark_style(self)
        self._build_ui()
        self._setup_shortcuts()
        self._build_toolbar()
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage('Ready — open a folder of STO screenshots to start annotating.')

        # Shared status-bar progress widget — replaces the modal popups
        # the trainer used to spawn for screen-type detection and icon
        # recognition. Cancel goes through `_cancel_active_run` which
        # routes the request to whichever worker is currently running.
        from warp.gui.progress_bar import StatusProgressBar
        self._status_progress = StatusProgressBar(self)
        self._status_progress.cancel_requested.connect(self._cancel_active_run)
        self.statusBar().addPermanentWidget(self._status_progress)

        # Periodic HF sync — fires every 5 minutes, uploads only if data changed.
        # Skipped in embed mode: the launcher's SyncCoordinator owns the timer
        # so we don't fire two refreshes per cycle.
        if not self._embed:
            self._sync_timer = QTimer(self)
            self._sync_timer.setInterval(5 * 60 * 1000)   # 5 minutes in ms
            self._sync_timer.timeout.connect(self._on_sync_timer)
            self._sync_timer.start()


    def showEvent(self, event):
        """Ensure canvas has focus once window is shown; restore last folder."""
        super().showEvent(event)
        self.activateWindow()
        self.raise_()
        if hasattr(self, '_ann_widget'):
            self._ann_widget.setFocus()
        # Re-check `_screenshots` and `_mode` *inside* the timer callback:
        # the launcher's setCurrentIndex fires this showEvent BEFORE
        # `set_fast_correction_mode` runs, so a check here would still
        # think we're in training mode. By the time the singleShot fires
        # (0 ms later, but after the current event loop iteration),
        # set_fast_correction_mode has already populated `_screenshots`
        # with the staged batch and flipped `_mode` — both gates skip
        # the clobbering `_load_folder` call.
        if not self._screenshots:
            last = self._settings.value(_KEY_LAST_DIR, '')
            if last and Path(last).is_dir():
                def _maybe_restore():
                    if self._screenshots or self._mode == 'fast_correction':
                        return
                    self._load_folder(Path(last))
                QTimer.singleShot(0, _maybe_restore)

    def _build_ui(self):
        c = QWidget()
        self.setCentralWidget(c)
        root = QVBoxLayout(c)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Fast Correction Mode banner — hidden by default; revealed by
        # `set_fast_correction_mode()`. ObjectName lets `accent_qss` scope
        # its styling here without bleeding onto unrelated frames.
        self._fast_banner = self._make_fast_correction_banner()
        self._fast_banner.setVisible(False)
        root.addWidget(self._fast_banner)

        sp = QSplitter(Qt.Orientation.Horizontal)
        sp.addWidget(self._make_left_panel())
        sp.addWidget(self._make_center_panel())
        sp.addWidget(self._make_right_panel())
        sp.setStretchFactor(0, 0)
        sp.setStretchFactor(1, 1)
        sp.setStretchFactor(2, 0)
        # Right pane hosts the slot/name controls plus the 5-column
        # recognition-review tree. The earlier 400 px squashed the Item
        # column; widen to match WARP's Results panel proportions so the
        # two tabs feel consistent.
        sp.setSizes([220, 640, 560])

        # Top-level tabs: the entire annotation workspace (list + canvas +
        # side panel) lives under "Screenshot"; Detection Logs is its
        # sibling so toggling between annotation and diagnostics swaps
        # the whole view, matching WARP's Results/Preview/Detection Logs
        # layout.
        from warp.gui.log_view import LogViewWidget
        self._top_tabs = QTabWidget()
        self._top_tabs.addTab(sp, 'Screenshot')
        # WARP CORE tails its own channel ('detection_core') so its logs
        # stay isolated from WARP's pane. Workers in this module wrap
        # their writes via `use_detection_channel('detection_core')`.
        self._log_view = LogViewWidget(channel='detection_core')
        self._log_view.set_default_save_name_cb(self._suggest_log_save_stem)
        self._top_tabs.addTab(self._log_view, 'Detection Logs')
        root.addWidget(self._top_tabs)

    def _make_left_panel(self) -> QWidget:
        left = QWidget()
        left.setMinimumWidth(400)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(8, 8, 8, 8)
        ll.setSpacing(6)
        lbl = QLabel('Screenshots')
        lbl.setFont(QFont('', 10, QFont.Weight.Bold))
        ll.addWidget(lbl)
        self._file_filter = QLineEdit()
        self._file_filter.setPlaceholderText('Filter by filename…')
        self._file_filter.setClearButtonEnabled(True)
        self._file_filter.textChanged.connect(self._apply_file_filter)
        ll.addWidget(self._file_filter)
        self._file_list = QListWidget()
        self._file_list.setItemDelegate(_ColorPreservingDelegate(self._file_list))
        self._file_list.currentRowChanged.connect(self._load_screenshot)
        self._file_list.itemChanged.connect(self._on_file_item_changed)
        self._file_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._file_list.customContextMenuRequested.connect(self._show_file_list_context_menu)
        ll.addWidget(self._file_list, 1)
        self._prog_lbl = QLabel('0 / 0 annotated')
        self._prog_lbl.setStyleSheet(f'color:{MFG};font-size:10px;')
        self._prog_bar = QProgressBar()
        self._prog_bar.setRange(0, 100)
        self._prog_bar.setValue(0)
        self._prog_bar.setFixedHeight(6)
        self._prog_bar.setTextVisible(False)
        ll.addWidget(self._prog_lbl)
        ll.addWidget(self._prog_bar)
        self._btn_done = QPushButton('✓ Mark Done')
        self._btn_done.setCheckable(True)
        self._btn_done.setEnabled(False)
        self._btn_done.clicked.connect(self._on_done_toggle)
        ll.addWidget(self._btn_done)
        # Training mode sends the currently loaded screenshot (single image);
        # Fast Correction Mode sends every file in the batch. The label flips
        # between "Send this to WARP" and "Send to WARP" on FC entry/exit.
        self._btn_send_to_warp = QPushButton('↗ Send this to WARP')
        self._btn_send_to_warp.setEnabled(False)
        self._btn_send_to_warp.setToolTip(
            'Mark this screenshot as Done first.')
        self._btn_send_to_warp.clicked.connect(self._on_send_to_warp)
        ll.addWidget(self._btn_send_to_warp)
        return left

    def _make_center_panel(self) -> QWidget:
        center = QWidget()
        center.setMinimumWidth(400)
        cl = QVBoxLayout(center)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(0)
        self._ann_widget = AnnotationWidget(self._data_mgr)
        self._ann_widget.installEventFilter(self)
        self._ann_widget.annotation_added.connect(self._on_bbox_drawn)
        self._ann_widget.item_selected.connect(self._on_item_selected)
        self._ann_widget.item_deselected.connect(self._on_canvas_deselected)
        self._ann_widget.bbox_changed.connect(self._on_bbox_changed)
        self._scroll_area = QScrollArea()
        self._scroll_area.setWidget(self._ann_widget)
        self._scroll_area.setWidgetResizable(False)
        self._scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._scroll_area.setStyleSheet(f'QScrollArea {{ background: {BG}; border: none; }}')
        cl.addWidget(self._scroll_area, 1)
        # Fixed-height frame so progress bar never shifts the canvas or bottom panel
        _pf = QWidget()
        _pf.setFixedHeight(6)
        _pf_lay = QHBoxLayout(_pf)
        _pf_lay.setContentsMargins(0, 0, 0, 0)
        self._match_progress = QProgressBar()
        self._match_progress.setRange(0, 0)
        self._match_progress.setFixedHeight(6)
        self._match_progress.setTextVisible(False)
        self._match_progress.setVisible(False)
        _pf_lay.addWidget(self._match_progress)
        cl.addWidget(_pf)
        cl.addWidget(self._make_bottom_panel())
        return center

    def _make_bottom_panel(self) -> QGroupBox:
        g = QGroupBox('Annotate Selected Icon')
        g.setFixedHeight(120)
        lay = QHBoxLayout(g)
        lay.setSpacing(10)
        sc = QVBoxLayout()
        sc.addWidget(QLabel('Slot:'))
        self._slot_combo = QComboBox()
        self._slot_combo.setFixedWidth(200)
        self._slot_combo.setEditable(False)
        for s in ALL_SLOTS:
            self._slot_combo.addItem(s)
        self._slot_combo.currentTextChanged.connect(self._on_slot_changed)
        sc.addWidget(self._slot_combo)
        lay.addLayout(sc)
        nc = QVBoxLayout()
        self._name_label = QLabel('Item name:')
        nc.addWidget(self._name_label)
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("Item name (or leave blank for 'Unknown')")
        self._name_edit.returnPressed.connect(self._on_accept)
        self._name_edit.textEdited.connect(self._on_name_edited)
        self._name_edit.focusInEvent  = self._on_name_focus_in
        self._name_edit.mousePressEvent = self._on_name_mouse_press
        self._completer_model = QStandardItemModel()
        self._completer = QCompleter(self._completer_model, self._name_edit)
        self._completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        self._completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self._completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self._completer.setMaxVisibleItems(12)
        self._completer.activated.connect(self._on_completer_activated)
        self._name_edit.setCompleter(self._completer)
        # Wayland: store popup widget and install event filter so we can set
        # transientParent on first Show (handle does not exist until then).
        self._completer_popup = self._completer.popup()
        self._completer_popup.installEventFilter(self)
        nc.addWidget(self._name_edit)
        self._tier_combo = QComboBox()
        for t in SHIP_TIER_VALUES:
            self._tier_combo.addItem(t)
        self._tier_combo.hide()
        self._tier_combo.textActivated.connect(
            lambda _: self._on_accept() if self._slot_combo.currentText() == 'Ship Tier' else None)
        nc.addWidget(self._tier_combo)
        self._ship_type_combo = QComboBox()
        self._ship_type_combo.setEditable(True)
        self._ship_type_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self._ship_type_combo.lineEdit().setPlaceholderText('Type to search ship...')
        stc = QCompleter(self._ship_type_combo.model(), self._ship_type_combo)
        stc.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        stc.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        stc.setFilterMode(Qt.MatchFlag.MatchContains)
        stc.setMaxVisibleItems(14)
        self._ship_type_combo.setCompleter(stc)
        self._ship_type_combo.hide()
        self._ship_type_combo.textActivated.connect(
            lambda _: self._on_accept() if self._slot_combo.currentText() == 'Ship Type' else None)
        stc.activated.connect(
            lambda _: self._on_accept() if self._slot_combo.currentText() == 'Ship Type' else None)
        nc.addWidget(self._ship_type_combo)
        lay.addLayout(nc, 1)
        bc = QVBoxLayout()
        bc.addStretch()
        br = QHBoxLayout()
        self._btn_accept = QPushButton('Accept')
        self._btn_accept.setStyleSheet(primary_btn_style())
        self._btn_accept.clicked.connect(self._on_accept)
        self._btn_accept.setToolTip('Accept (Enter)')
        br.addWidget(self._btn_accept)
        self._chk_auto_accept = QCheckBox('Auto ≥')
        self._chk_auto_accept.setToolTip(
            'Auto-accept items where ML confidence meets threshold')
        self._chk_auto_accept.setChecked(False)
        self._spin_auto_conf = QDoubleSpinBox()
        self._spin_auto_conf.setRange(0.5, 1.0)
        self._spin_auto_conf.setSingleStep(0.05)
        self._spin_auto_conf.setValue(0.75)
        self._spin_auto_conf.setDecimals(2)
        self._spin_auto_conf.setFixedWidth(72)
        self._spin_auto_conf.setToolTip('Min confidence for auto-accept')
        br.addWidget(self._chk_auto_accept)
        br.addWidget(self._spin_auto_conf)
        bc.addLayout(br)
        lay.addLayout(bc)
        return g

    def _make_right_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(400)
        pl = QVBoxLayout(panel)
        pl.setContentsMargins(6, 8, 6, 8)
        pl.setSpacing(6)
        hdr = QLabel('Recognition Review')
        hdr.setFont(QFont('', 10, QFont.Weight.Bold))
        hdr.setStyleSheet(f'color:{ACCENT};')
        pl.addWidget(hdr)
        hint = QLabel('Confirmed · Auto-confirmed (needs review) · Unmatched\nClick item to select on canvas.')
        hint.setWordWrap(True)
        hint.setStyleSheet(f'color:{MFG};font-size:10px;')
        pl.addWidget(hint)
        self._screen_type_badge = QLabel('Screen type: —')
        self._screen_type_badge.setStyleSheet(f'color:{C_WARNING};background:{MBG};border:1px solid {LBG};border-radius:3px;padding:2px 6px;font-size:11px;')
        pl.addWidget(self._screen_type_badge)
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f'color:{BC};')
        pl.addWidget(sep)
        # 5-column flat tree (Slot / Idx / Item / Conf [%] / Status) wearing
        # a QListWidget-compatible API so the dozens of `_review_list.item(N)`,
        # `setCurrentRow`, `takeItem` call sites scattered through this
        # window keep working unchanged.
        self._review_list = _ReviewListAdapter()
        self._review_list.setItemDelegate(_ColorPreservingDelegate(self._review_list))
        # Selected row picks up the amber accent (matches WARP's Results
        # view + Export to SETS JSON button) so the user can spot the
        # active row at a glance regardless of per-state foreground.
        self._review_list.setStyleSheet(
            f'QTreeWidget::item:selected {{ '
            f'background-color: {ACCENT}; color: #1a1a1a; }}'
        )
        self._review_list.currentRowChanged.connect(self._on_review_row_changed)
        self._review_list.parentRowSelected.connect(self._on_review_parent_selected)
        self._review_list.installEventFilter(self)
        self._review_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._review_list.customContextMenuRequested.connect(self._show_review_context_menu)
        # ann_widget event filter set after creation in _make_center_panel.
        # QTreeWidget.itemClicked passes (item, col); the old handler only
        # cares about the item — wrap to drop the column.
        self._review_list.itemClicked.connect(
            lambda it, _col: self._on_review_item_clicked(it))
        pl.addWidget(self._review_list, 1)
        self._review_summary = QLabel('')
        self._review_summary.setStyleSheet(f'color:{MFG};font-size:10px;')
        self._review_summary.setWordWrap(True)
        pl.addWidget(self._review_summary)
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet(f'color:{BC};')
        pl.addWidget(sep2)
        self._btn_edit_bbox = QPushButton('Edit BBox')
        self._btn_edit_bbox.setStyleSheet(secondary_btn_style(checked_border=True))
        self._btn_edit_bbox.setCheckable(True)
        self._btn_edit_bbox.clicked.connect(self._on_edit_bbox_toggle)
        self._btn_edit_bbox.setVisible(False)  # Resize/move disabled — reserved for future
        pl.addWidget(self._btn_edit_bbox)
        mgmt = QHBoxLayout()
        self._btn_add_bbox = QPushButton('+ Add BBox')
        self._btn_add_bbox.setStyleSheet(warning_btn_style(checked_border=True))
        self._btn_add_bbox.setCheckable(True)
        self._btn_add_bbox.clicked.connect(self._on_add_bbox_toggle)
        self._btn_remove_item = QPushButton('- Remove BBox')
        self._btn_remove_item.setStyleSheet(danger_btn_style())
        self._btn_remove_item.clicked.connect(self._on_remove_item)
        self._btn_clear_all_bboxes = QPushButton('Clear All BBoxes')
        self._btn_clear_all_bboxes.setStyleSheet(danger_btn_style())
        self._btn_clear_all_bboxes.setToolTip(
            'Remove every bbox on the current screenshot. A confirmation dialog '
            'offers the option to spare bboxes already marked confirmed.'
        )
        self._btn_clear_all_bboxes.clicked.connect(self._on_clear_all_bboxes)
        mgmt.addWidget(self._btn_add_bbox)
        mgmt.addWidget(self._btn_remove_item)
        mgmt.addWidget(self._btn_clear_all_bboxes)
        pl.addLayout(mgmt)
        self._manual_mode_lbl = QLabel('')
        self._manual_mode_lbl.setStyleSheet(f'color:{C_WARNING};font-size:10px;background:{MBG};border:1px solid {LBG};border-radius:3px;padding:3px;')
        self._manual_mode_lbl.setWordWrap(True)
        self._manual_mode_lbl.setVisible(False)
        pl.addWidget(self._manual_mode_lbl)
        self._set_review_buttons_enabled(False)
        return panel

    def _build_toolbar(self):
        tb = QToolBar('Main')
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.addToolBar(tb)
        def act(l, t, s):
            a = QAction(l, self)
            a.setToolTip(t)
            a.triggered.connect(s)
            tb.addAction(a)
            tb.addSeparator()
            return a
        self._action_open_screenshot = act(
            'Open Screenshot', 'Open a single screenshot (loads its parent folder and selects the file)',
            self._on_open_screenshot)
        self._action_open_folder = act(
            'Open Folder', 'Open screenshots folder', self._on_open)
        self._action_detect_screen_types = act(
            'Detect Screen Types', 'Re-classify screen types', self._on_detect_screen_types)
        self._action_auto_detect = act(
            'Auto-Detect Slots', 'Auto-detect icons', self._on_auto_detect)

    def _set_toolbar_actions_enabled(self, enabled: bool) -> None:
        """Toggle the four detect-relevant toolbar actions together.

        Used by the screen-type detection flow to lock the toolbar while a
        run is in flight — re-enabling Auto-Detect Slots mid-classification
        would let the user kick off recognition with stale screen types.
        After re-enabling, `_load_screenshot` reapplies the lock-state
        gating on Auto-Detect Slots (Done screenshots stay disabled).
        """
        for a in (self._action_open_screenshot,
                  self._action_open_folder,
                  self._action_detect_screen_types,
                  self._action_auto_detect):
            a.setEnabled(enabled)

    # ── Fast Correction Mode ────────────────────────────────────────────
    def _make_fast_correction_banner(self) -> QFrame:
        f = QFrame()
        f.setObjectName('accent_banner')
        lay = QHBoxLayout(f)
        lay.setContentsMargins(10, 4, 6, 4)
        lay.setSpacing(8)
        lbl = QLabel('WARP Fast Correction Mode — fix items, Mark Done, then Send All to WARP.')
        lbl.setObjectName('accent_banner_text')
        lay.addWidget(lbl, 1)
        btn = QPushButton('Exit Fast Correction')
        btn.setObjectName('accent_exit_btn')
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.clicked.connect(self.exit_fast_correction_mode)
        lay.addWidget(btn)
        return f

    def set_fast_correction_mode(self, files: list, items_by_file: dict,
                                 stype_by_file: dict | None = None) -> None:
        """Switch into Fast Correction Mode.

        `files` is a list of Path-like screenshot paths from WARP. The mode
        replaces the regular folder-load workflow: file list is fixed, Open
        toolbar entries are hidden, and the window wears a warm-amber accent
        so the user can tell at a glance they are in the ephemeral
        correction view rather than the full training-data review.

        `items_by_file` maps the *original* file path (str) → WARP's
        RecognisedItem list. We convert each list to the trainer's dict
        shape and seed `_recognition_cache` under the *staged* basename
        so `_load_screenshot` renders WARP's detection on entry — the
        whole point of Fast Mode is to fix exactly those items, not the
        user's older disk-confirmed annotations.
        """
        orig_paths = [Path(p) for p in files]
        if not orig_paths:
            self.statusBar().showMessage(
                'Fast Correction: no files supplied — nothing to do.', 5000)
            return
        # Stage every input under a content-hashed dir in the warp cache
        # so Fast Mode never touches the original file's training data
        # (separate filenames → separate TDM keys, separate crops). The
        # session also persists across re-entry: same file set → same
        # hash → resume in-progress annotations.
        from warp.trainer import fast_session as _fs
        sess = _fs.prepare(orig_paths)
        if not sess.staged_paths:
            self.statusBar().showMessage(
                'Fast Correction: failed to stage any input file.', 5000)
            return
        # Snapshot the training-mode view (folder, screen types, done set,
        # recognition cache, selection, filter) so `exit_fast_correction_mode`
        # can drop the user back exactly where they were before WARP handed
        # the batch over. Only taken on first entry — re-entering FC keeps
        # the original training snapshot intact, otherwise the second FC
        # session would snapshot the first FC's staged batch and exit could
        # never escape Fast Mode.
        if self._mode != 'fast_correction':
            self._pre_fc_snapshot = {
                'screenshots':          list(self._screenshots),
                'screen_types':         dict(self._screen_types),
                'screen_types_manual':  set(self._screen_types_manual),
                'screen_types_ml_auto': set(self._screen_types_ml_auto),
                'recognition_cache':    dict(self._recognition_cache),
                'recognition_items':    list(self._recognition_items),
                'screenshots_done':     set(self._screenshots_done),
                'current_idx':          self._current_idx,
                'file_filter':          self._file_filter.text(),
            }
        self._fast_session = sess
        self._mode = 'fast_correction'
        from warp.debug import log as _wlog
        _wlog.info(
            f'Fast Correction: ENTER hash={sess.hash} '
            f'staged={len(sess.staged_paths)} '
            f'items_by_file_keys={list(items_by_file.keys())[:3]}{"..." if len(items_by_file) > 3 else ""} '
            f'total_keys={len(items_by_file)}')
        # Reset state so the new batch is the only thing on screen.
        self._screenshots = list(sess.staged_paths)
        self._screen_types.clear()
        self._screen_types_manual.clear()
        self._screen_types_ml_auto.clear()
        self._recognition_cache.clear()
        self._recognition_items = []
        # Seed each staged file's cache with WARP's items so the review
        # panel renders WARP's raw detection (bboxes possibly off-by-a-px
        # vs. the user's older annotations — this is the *point*: Fast
        # Mode is an independent confirmation pass that adds a fresh
        # training signal). Match WARP's items to staged paths by
        # basename, not full path: WARP keys `items_by_file` using
        # `Path(source_file).resolve()` which can disagree with our
        # un-resolved `Path(p)` (symlinks, cwd, trailing slashes), so
        # comparing strings is fragile. Basenames always line up because
        # `prepare()` uses `orig.name` to form the staged filename.
        staged_by_origname = {o.name: s for s, o in sess.paths_map.items()}
        for orig_key, warp_items in items_by_file.items():
            if not warp_items:
                continue
            staged = staged_by_origname.get(Path(orig_key).name)
            if staged is None:
                log.debug(
                    f'Fast Correction: no staged path for orig {orig_key!r}; '
                    f'skipping {len(warp_items)} item(s)')
                continue
            try:
                dicts = self._recognition_dicts_from_warp_items(staged, warp_items)
                self._recognition_cache[staged.name] = dicts
                log.info(
                    f'Fast Correction: seeded {len(dicts)} items for '
                    f'{Path(orig_key).name} → {staged.name}')
            except Exception as e:
                log.warning(
                    f'Fast Correction: seed conversion failed for '
                    f'{Path(orig_key).name}: {e}')
        # Each FC entry is a fresh correction pass — clear any Done flag a
        # previous run of the same hashed batch left in `_screenshots_done`.
        # Otherwise the file list shows the staged names as green ("done")
        # the moment the user re-enters, even though the new pass is open
        # for editing. Persist the cleared state so the disk JSON matches.
        staged_names = {p.name for p in self._screenshots}
        if staged_names & self._screenshots_done:
            self._screenshots_done.difference_update(staged_names)
            self._save_done()
        self._current_idx = -1
        self._file_list.clear()
        # Screen type comes from WARP's classifier for *this* recognition
        # session — not from the TDM (which would carry the user's previous
        # labels on the original file and contradict the FC principle of
        # treating WARP's batch as an independent detection pass). Match
        # `stype_by_file` to staged paths by basename, same as items above.
        stype_by_origname = {}
        if stype_by_file:
            for k, v in stype_by_file.items():
                if v:
                    stype_by_origname[Path(k).name] = v
        for p in self._screenshots:
            orig_name = sess.paths_map[p].name if p in sess.paths_map else p.name
            stype = stype_by_origname.get(orig_name, '')
            self._screen_types[p.name] = stype if stype else 'UNKNOWN'
            if stype:
                self._screen_types_ml_auto.add(p.name)
            self._file_list.addItem(self._make_file_list_item(p, self._screen_types[p.name]))
        self._apply_file_filter(self._file_filter.text())
        # Hide controls that don't apply: caller picks the file set.
        for a in (self._action_open_screenshot, self._action_open_folder):
            a.setVisible(False)
        # Warm-amber chrome + banner.
        self.setStyleSheet(accent_qss('fast_correction'))
        self._fast_banner.setVisible(True)
        self._btn_send_to_warp.setText('↗ Send to WARP')
        self.setWindowTitle('WARP CORE — Fast Correction Mode')
        if self._file_list.count():
            self._file_list.setCurrentRow(0)
        self.statusBar().showMessage(
            f'Fast Correction Mode — {len(self._screenshots)} screenshot(s) '
            f'loaded from WARP (session {sess.hash}).')

    def exit_fast_correction_mode(self) -> None:
        """Leave Fast Correction Mode and restore the regular trainer UI.

        If a training-mode snapshot was captured on FC entry, the previous
        folder, selection, screen types, done flags and recognition cache
        are restored so the trainer comes back exactly as the user left it.
        Without a snapshot (FC entered before any training-mode folder was
        loaded) the trainer falls back to its empty initial state.
        """
        if self._mode != 'fast_correction':
            return
        self._mode = 'training'
        self._fast_session = None
        self.setStyleSheet('')
        apply_dark_style(self)
        self._fast_banner.setVisible(False)
        self._btn_send_to_warp.setText('↗ Send this to WARP')
        for a in (self._action_open_screenshot, self._action_open_folder):
            a.setVisible(True)
        self.setWindowTitle('WARP CORE — ML Trainer')

        snap = getattr(self, '_pre_fc_snapshot', None)
        self._pre_fc_snapshot = None
        self._file_list.clear()

        if snap and snap.get('screenshots'):
            self._screenshots          = snap['screenshots']
            self._screen_types         = snap['screen_types']
            self._screen_types_manual  = snap['screen_types_manual']
            self._screen_types_ml_auto = snap['screen_types_ml_auto']
            self._recognition_cache    = _DebugDict('_recognition_cache', snap['recognition_cache'])
            self._recognition_items    = snap['recognition_items']
            self._screenshots_done     = snap['screenshots_done']
            for p in self._screenshots:
                self._file_list.addItem(
                    self._make_file_list_item(
                        p, self._screen_types.get(p.name, 'UNKNOWN')))
            self._file_filter.setText(snap.get('file_filter', ''))
            self._apply_file_filter(self._file_filter.text())
            prev_idx = snap.get('current_idx', -1)
            if 0 <= prev_idx < len(self._screenshots):
                self._file_list.setCurrentRow(prev_idx)
            else:
                self._current_idx = -1
            self.statusBar().showMessage(
                'Back to training mode — previous folder restored.')
        else:
            # No previous training-mode state — drop the ephemeral batch
            # and let the user open a real folder.
            self._screenshots = []
            self._screen_types.clear()
            self._screen_types_manual.clear()
            self._screen_types_ml_auto.clear()
            self._recognition_cache.clear()
            self._recognition_items = []
            self._current_idx = -1
            self.statusBar().showMessage(
                'Back to training mode — open a folder to start annotating.')

        self.fast_correction_exited.emit()

    def _on_open(self):
        from warp.folder_picker import pick_folder
        last = self._settings.value(_KEY_LAST_DIR, '')
        folder = pick_folder(self, title='Open Screenshots Folder', start_dir=last)
        if folder is None:
            return
        self._settings.setValue(_KEY_LAST_DIR, str(folder))
        self._load_folder(folder)

    def _on_open_screenshot(self):
        from warp.folder_picker import pick_files, DEFAULT_IMAGE_FILTERS
        last = self._settings.value(_KEY_LAST_DIR, '') or str(Path.home())
        files = pick_files(
            self,
            title='Open screenshot',
            start_dir=last,
            image_filters=DEFAULT_IMAGE_FILTERS,
            multi=False,
        )
        if not files:
            return
        self._settings.setValue(_KEY_LAST_DIR, str(files[0].parent))
        self.open_screenshot(files[0])

    def open_screenshot(self, path, preload_items=None):
        """Load `path`'s parent folder (if not already loaded) and select
        the matching row in the file list. Public entry point used by the
        toolbar's "Open Screenshot" action and by the launcher when WARP
        hands off via the Results-row "Open in WARP CORE" menu.

        `preload_items`: optional list[RecognisedItem] from WARP. When
        provided, the trainer skips its own Auto-Detect and seeds the
        review panel with these items (as `pending`, so the user has to
        confirm/correct each one). Existing disk annotations for the same
        bbox+slot still take precedence via `_populate_review_panel`.
        """
        # Resolve symlinks so launcher hand-offs from WARP single-file
        # selections (staged through a temp dir) land on the original
        # screenshot's real folder, not the temp staging dir.
        try:
            p = Path(path).resolve()
        except Exception:
            p = Path(path)
        if not p.is_file():
            QMessageBox.warning(self, 'Open Screenshot',
                                f'File not found:\n{p}')
            return
        folder = p.parent
        current_folder = self._screenshots[0].parent if self._screenshots else None
        if current_folder != folder:
            self._settings.setValue(_KEY_LAST_DIR, str(folder))
            self._load_folder(folder)
        if self._file_filter.text():
            self._file_filter.clear()
        # Seed the recognition cache before selecting the row so the
        # subsequent `_load_screenshot` picks WARP's items up as if a
        # detection run had just finished.
        if preload_items:
            try:
                dicts = self._recognition_dicts_from_warp_items(p, preload_items)
                self._recognition_cache[p.name] = dicts
                log.info(
                    f'open_screenshot: preloaded {len(dicts)} items from WARP '
                    f'for {p.name}')
            except Exception as e:
                log.warning(f'open_screenshot: preload conversion failed: {e}')
        # Defer selection one event-loop tick: `_load_folder` triggers
        # async screen-type detection which can race with `setCurrentRow`
        # and leave the row unhighlighted. A single-shot timer lets the
        # list finish laying out before we move the cursor.
        target = p.name
        force_reload = bool(preload_items)
        def _select_now():
            for i, sp in enumerate(self._screenshots):
                if sp.name == target:
                    prev_idx = self._current_idx
                    self._file_list.setCurrentRow(i)
                    it = self._file_list.item(i)
                    if it is not None:
                        self._file_list.scrollToItem(
                            it,
                            QAbstractItemView.ScrollHint.PositionAtCenter,
                        )
                    self._file_list.setFocus()
                    # `setCurrentRow` only fires `currentRowChanged` when
                    # the index actually moves. When the launcher hands off
                    # the same file the user is already viewing, force a
                    # reload so the freshly-injected cache is rendered.
                    if force_reload and prev_idx == i:
                        self._load_screenshot(i)
                    break
        QTimer.singleShot(0, _select_now)

    def _recognition_dicts_from_warp_items(self, path: Path, warp_items):
        """Convert WARP's RecognisedItem list to the trainer's review-panel
        dict shape. Crops are sliced from the image once so `_on_accept`
        can still call `add_session_example` for any item the user
        manually accepts inside the trainer."""
        import cv2
        img = cv2.imread(str(path))
        out: list[dict] = []
        for it in warp_items:
            bbox = tuple(it.bbox) if it.bbox else ()
            crop = None
            if img is not None and bbox and len(bbox) == 4:
                x, y, w, h = bbox
                if w > 0 and h > 0 and y >= 0 and x >= 0:
                    crop_view = img[y:y + h, x:x + w]
                    if crop_view.size > 0:
                        crop = crop_view.copy()
            out.append({
                'name':        it.name or '',
                'slot':        it.slot,
                'conf':        float(it.confidence or 0.0),
                'bbox':        bbox,
                'state':       'pending',
                'thumb':       None,
                'crop_bgr':    crop,
                'orig_name':   it.name or '',
                'ship_name':   '',
                'cross_check_failed': False,
                'auto_confirmed':     False,
                # Carry layout metadata so the group view in Fast Mode
                # orders slots L→R / by seat, matching what WARP results
                # showed. Without these the trainer falls back to insertion
                # order, which collapses group structure.
                'seat_key':    getattr(it, 'seat_key', '') or '',
                'slot_index':  int(getattr(it, 'slot_index', -1) if getattr(it, 'slot_index', None) is not None else -1),
            })
        return out

    def _apply_file_filter(self, text: str):
        """Hide file-list rows that don't contain `text` (case-insensitive
        substring). Empty `text` reveals all rows.
        """
        needle = (text or '').strip().lower()
        for row in range(self._file_list.count()):
            item = self._file_list.item(row)
            if not item:
                continue
            if not needle:
                item.setHidden(False)
                continue
            # Each file-list item is rendered as "<icon> <label>\n  <name>"
            # (see `_make_file_list_item`); the filename lives on the
            # second line. Match against the whole item text so users can
            # also filter by screen-type label.
            item.setHidden(needle not in item.text().lower())

    def _load_folder(self, folder: Path):
        exts = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
        self._screenshots = sorted([f for f in folder.iterdir() if f.suffix.lower() in exts])
        if not self._screenshots:
            self.statusBar().showMessage('No images found.')
            return
        self._screen_types.clear()
        self._screen_types_manual.clear()
        self._screen_types_ml_auto.clear()
        self._screenshots_done = self._load_done()
        self._recognition_cache.clear()
        self._recognition_items = []
        self._current_idx = -1
        self._file_list.clear()
        # Restore persisted screen type labels and confirmation state.
        # Lookup by path (content hash) — not via the filename projection
        # in get_all_screen_types / get_user_confirmed_set, which silently
        # drops sha16 entries whose `_image_meta` wasn't seeded by
        # annotations.json (e.g. screenshots confirmed before any item
        # was annotated).
        for p in self._screenshots:
            saved = self._data_mgr.get_screen_type(p)
            self._screen_types[p.name] = saved if saved else 'UNKNOWN'
            if self._data_mgr.is_user_confirmed(p):
                self._screen_types_manual.add(p.name)
            elif saved:
                self._screen_types_ml_auto.add(p.name)
            self._file_list.addItem(self._make_file_list_item(p, self._screen_types[p.name]))
        # Re-apply any active filter so freshly added items honor it.
        self._apply_file_filter(self._file_filter.text())
        self._start_screen_type_detection("open_folder")

    def _start_screen_type_detection(self, trigger: str = 'unknown', max_iter: int = 1):
        self._detect_trigger = trigger
        self._detect_loop_max = max_iter
        self._detect_loop_iter = 0
        self._detect_loop_prev_unresolved = None
        # Iteration-1 seed is `_screen_types_manual` only. If every screenshot
        # is already user-confirmed, there is nothing left for the classifier
        # to do — skip the dialog entirely instead of opening an empty one.
        pending = [p for p in self._screenshots
                   if p.name not in self._screen_types_manual]
        if not pending:
            # Popup only when triggered by the toolbar button — folder-open
            # auto-runs the same function and should stay quiet.
            if trigger == 'detect_screen_types_button':
                QMessageBox.information(
                    self,
                    'Detect Screen Types',
                    'All screenshots are already confirmed — nothing to detect.',
                )
            return
        # Status-bar progress now replaces the popup. _detect_dlg stays
        # set to a truthy sentinel so the rest of the detect flow's
        # `if self._detect_dlg:` guards still scope status-bar updates
        # to runs that the user actually started here.
        self._detect_dlg = '__statusbar__'
        self._detect_max_iter = max_iter
        self._set_toolbar_actions_enabled(False)
        self._status_progress.start(determinate=True, maximum=max(1, len(pending)))
        self.statusBar().showMessage(
            f'Classifying screenshots — loop mode ({max_iter} iterations)…'
            if max_iter > 1 else 'Classifying screenshots with ML model…'
        )
        self._run_detect_iteration()

    def _run_detect_iteration(self):
        self._detect_loop_iter += 1
        models_dir = userdata.models_dir()
        # Iteration 1: seed from green only. Subsequent iterations: green + yellow (ml_auto).
        seed_names = (self._screen_types_manual if self._detect_loop_iter == 1
                      else self._screen_types_manual | self._screen_types_ml_auto)
        confirmed_types = {
            p: self._screen_types[p.name]
            for p in self._screenshots
            if p.name in seed_names
            and self._screen_types.get(p.name, 'UNKNOWN') != 'UNKNOWN'
        }
        # Skip already-confirmed screenshots: their type is fixed and the ML
        # result would be discarded downstream anyway. Saves a classify() call
        # per confirmed file on every folder open / re-detect.
        paths_to_scan = [p for p in self._screenshots if p.name not in seed_names]
        total = len(paths_to_scan)
        if self._detect_dlg:
            self._status_progress.start(determinate=True, maximum=max(1, total))
        self._detect_worker = ScreenTypeDetectorWorker(
            paths_to_scan, models_dir=models_dir,
            confirmed_types=confirmed_types, parent=self,
        )
        self._detect_worker.progress.connect(self._on_detect_progress)
        self._detect_worker.finished.connect(self._on_detect_finished)
        self._detect_worker.start()
        self.statusBar().showMessage(
            f'Detecting screen types — iteration {self._detect_loop_iter} / {self._detect_loop_max}…'
        )

    def _on_detect_progress(self, idx: int, total: int, filename: str, stype: str, conf: float):
        def _push_status(disp_stype: str):
            icon  = SCREEN_TYPE_ICONS.get(disp_stype, '?')
            label = SCREEN_TYPE_LABELS.get(disp_stype, 'Unknown')
            self._status_progress.set_progress(idx)
            self.statusBar().showMessage(f'[{idx}/{total}] {_disp_name(filename)}  →  {icon} {label}')

        # During scan: only update UI for items without a user-confirmed type
        if filename in self._screen_types_manual:
            if self._detect_dlg:
                _push_status(self._screen_types.get(filename, 'UNKNOWN'))
            return
        self._screen_types[filename] = stype
        for row, p in enumerate(self._screenshots):
            if p.name == filename:
                item = self._file_list.item(row)
                if item:
                    sc_icon = SCREEN_TYPE_ICONS.get(stype, '?')
                    label   = SCREEN_TYPE_LABELS.get(stype, 'Unknown')
                    self._file_list.blockSignals(True)
                    item.setText(f'{sc_icon} {label}\n  {_disp_name(filename)}')
                    item.setCheckState(Qt.CheckState.Unchecked)
                    item.setIcon(QIcon())  # no dot yet — final state set in _on_detect_finished
                    self._file_list.blockSignals(False)
                    item.setForeground(self._file_item_color(p))
                if row == 0 and self._current_idx < 0:
                    self._file_list.setCurrentRow(0)
                break
        if self._detect_dlg:
            _push_status(stype)

    def _on_detect_finished(self, results: dict):
        """
        Apply ML results with confirmation state logic:

        green (user-confirmed) → ALWAYS preserved, never overwritten.
        yellow / UNKNOWN + ML conf ≥ 0.95 → yellow (update type).
        yellow / UNKNOWN + ML conf < 0.95 → keep existing state.
        """
        for fname, (ml_stype, ml_conf) in results.items():
            path = next((p for p in self._screenshots if p.name == fname), None)
            if path is None:
                continue
            is_user = fname in self._screen_types_manual

            if is_user:
                pass   # green dot preserved — k-NN already pre-seeded with correct type
            else:
                if ml_stype != 'UNKNOWN' and ml_conf >= 0.95:
                    self._screen_types[fname] = ml_stype
                    self._screen_types_ml_auto.add(fname)
                    self._data_mgr.set_screen_type(path, ml_stype, user_confirmed=False)
                # conf < 0.95: keep whatever _on_detect_progress already set (or UNKNOWN)

        # Update file list UI
        for row, p in enumerate(self._screenshots):
            stype     = self._screen_types.get(p.name, 'UNKNOWN')
            is_user   = p.name in self._screen_types_manual
            is_ml     = p.name in self._screen_types_ml_auto
            item = self._file_list.item(row)
            if item:
                sc_icon = SCREEN_TYPE_ICONS.get(stype, '?')
                label   = SCREEN_TYPE_LABELS.get(stype, 'Unknown')
                self._file_list.blockSignals(True)
                item.setText(f'{sc_icon} {label}\n  {_disp_name(p.name)}')
                # Checkbox semantics: checked == user-confirmed ONLY. ML
                # auto-accept produces a yellow dot icon but leaves the
                # checkbox empty so "checked" is an honest persistent signal.
                item.setCheckState(Qt.CheckState.Checked if is_user else Qt.CheckState.Unchecked)
                item.setIcon(_get_user_icon() if is_user else (_get_ml_icon() if is_ml else QIcon()))
                self._file_list.blockSignals(False)
                item.setForeground(self._file_item_color(p))

        # Auto-Mark Done for DISCARD screenshots detected by ML
        for row, p in enumerate(self._screenshots):
            stype = self._screen_types.get(p.name, 'UNKNOWN')
            if stype == 'DISCARD' and p.name not in self._screenshots_done:
                self._screenshots_done.add(p.name)
                log.info(f'DISCARD: auto-marked Done for {p.name} (ML detection)')
        self._save_done()

        # --- Loop logic ---
        if self._detect_loop_max > 1:
            unknown_count = 0
            wrong_count = 0
            for fname, (ml_stype, _ml_conf) in results.items():
                if fname not in self._screen_types_manual:
                    continue
                user_stype = self._screen_types.get(fname, 'UNKNOWN')
                if ml_stype == 'UNKNOWN':
                    unknown_count += 1
                elif ml_stype != user_stype:
                    wrong_count += 1
            unresolved = unknown_count + wrong_count
            prev = self._detect_loop_prev_unresolved
            self._detect_loop_prev_unresolved = unresolved
            if self._detect_dlg:
                trend = ('' if prev is None
                         else f'  {prev} → {unresolved} ' + ('↓' if unresolved < prev
                                                              else '→' if unresolved == prev
                                                              else '↑'))
                self.statusBar().showMessage(
                    f'Iteration {self._detect_loop_iter} / {self._detect_loop_max}{trend}'
                    f'   (UNKNOWN: {unknown_count}, wrong: {wrong_count})'
                )
            progress_made = (prev is None) or (unresolved < prev)
            if progress_made and self._detect_loop_iter < self._detect_loop_max and unresolved > 0:
                self._detect_worker = None
                self._run_detect_iteration()
                return  # loop continues, status bar keeps updating
            reason = 'no progress' if not progress_made else ('all resolved' if unresolved == 0 else 'max iterations reached')
            log.info(f'Screen type loop stopped after {self._detect_loop_iter} iteration(s): {reason}')
            self._detect_worker = None
            self._status_progress.finish()
            self._detect_dlg = None
            self._set_toolbar_actions_enabled(True)
            self.statusBar().showMessage(f'Screen-type detection done — {reason}.')
            if self._screenshots:
                if self._current_idx < 0:
                    self._file_list.setCurrentRow(0)
                else:
                    self._load_screenshot(self._current_idx)
            self._update_progress()
            return

        if self._detect_dlg:
            self._status_progress.finish()
            self._detect_dlg = None
            self._set_toolbar_actions_enabled(True)
            self.statusBar().showMessage('Screen-type detection done.')
        self._detect_worker = None
        if self._screenshots:
            if self._current_idx < 0:
                self._file_list.setCurrentRow(0)
            else:
                self._load_screenshot(self._current_idx)
        self._update_progress()

    def _on_detect_cancelled(self):
        if self._detect_worker and self._detect_worker.isRunning():
            self._detect_worker.requestInterruption()
            self._detect_worker.wait(3000)
        self._status_progress.finish()
        self._detect_dlg = None
        self._set_toolbar_actions_enabled(True)
        self.statusBar().showMessage('Screen-type detection cancelled.')
        if self._screenshots:
            self._file_list.setCurrentRow(0)
        self._update_progress()

    def _cancel_active_run(self):
        """Status-bar Cancel button — routes to whichever worker is live.

        The trainer never runs detect + recog concurrently, so checking
        `isRunning()` on each in turn is sufficient. Disables Cancel
        immediately so the user can't click twice while interruption
        propagates through the worker's next progress checkpoint."""
        self._status_progress.set_cancel_enabled(False)
        if self._detect_worker is not None and self._detect_worker.isRunning():
            self._on_detect_cancelled()
            return
        if self._recog_worker is not None and self._recog_worker.isRunning():
            self._on_recognition_cancelled()
            return
        # No live worker — just hide the bar (defensive).
        self._status_progress.finish()

    def _make_file_list_item(self, p: Path, stype: str) -> QListWidgetItem:
        sc_icon = SCREEN_TYPE_ICONS.get(stype, '?')
        label   = SCREEN_TYPE_LABELS.get(stype, 'Unknown')
        item = QListWidgetItem(f'{sc_icon} {label}\n  {_disp_name(p.name)}')
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        is_user = p.name in self._screen_types_manual
        is_ml   = p.name in self._screen_types_ml_auto
        # Checkbox semantics: checked == user-confirmed ONLY. ML auto-accept
        # paints the yellow dot icon but never the checkbox.
        item.setCheckState(Qt.CheckState.Checked if is_user else Qt.CheckState.Unchecked)
        item.setIcon(_get_user_icon() if is_user else (_get_ml_icon() if is_ml else QIcon()))
        item.setForeground(self._file_item_color(p))
        return item

    def _update_file_item_check(self, row: int):
        """Sync checkbox state and dot icon with confirmation state."""
        item = self._file_list.item(row)
        if item is None or row >= len(self._screenshots):
            return
        fname   = self._screenshots[row].name
        is_user = fname in self._screen_types_manual
        is_ml   = fname in self._screen_types_ml_auto
        self._file_list.blockSignals(True)
        # Checkbox tracks user-confirmation only — see `_make_file_list_item`.
        item.setCheckState(Qt.CheckState.Checked if is_user else Qt.CheckState.Unchecked)
        item.setIcon(_get_user_icon() if is_user else (_get_ml_icon() if is_ml else QIcon()))
        self._file_list.blockSignals(False)

    def _on_file_item_changed(self, item: QListWidgetItem):
        """Checkbox toggled by user on file list item."""
        row = self._file_list.row(item)
        if row < 0 or row >= len(self._screenshots):
            return
        path = self._screenshots[row]
        is_checked = item.checkState() == Qt.CheckState.Checked
        stype = self._screen_types.get(path.name, 'UNKNOWN')
        if is_checked:
            # User confirms current type → green
            self._screen_types_manual.add(path.name)
            self._screen_types_ml_auto.discard(path.name)
            self._data_mgr.set_screen_type(path, stype, user_confirmed=True)
            try:
                import cv2
                from warp.recognition.screen_classifier import ScreenTypeClassifier
                img = cv2.imread(str(path))
                if img is not None:
                    ScreenTypeClassifier.add_session_example(img, stype)
            except Exception:
                pass
        else:
            # User un-confirms — clear both confirmed states, remove type label
            self._screen_types_manual.discard(path.name)
            self._screen_types_ml_auto.discard(path.name)
            self._data_mgr.remove_screen_type(path, stype)
        self._update_file_item_check(row)

    def _suggest_log_save_stem(self) -> str:
        """Default 'Save As' name for the Detection Logs tab.

        Uses the currently-loaded screenshot's stem plus its classifier
        verdict (SPACE / GROUND, if recognised) and the date — so the
        saved log file is traceable back to the annotation session it
        documents."""
        date = datetime.datetime.now().strftime('%Y%m%d')
        idx = getattr(self, '_current_idx', -1)
        try:
            if 0 <= idx < len(self._screenshots):
                path = self._screenshots[idx]
                stem = path.stem
                stype = (self._screen_types.get(path.name) or '').upper()
                fam = ''
                if stype.startswith('SPACE'):
                    fam = '_space'
                elif stype.startswith('GROUND'):
                    fam = '_ground'
                return f'{stem}{fam}_{date}'
        except Exception:
            pass
        return f'detection_{date}'

    def _load_screenshot(self, row: int):
        if row < 0 or row >= len(self._screenshots): return
        # Save layout for the screenshot we're leaving (if not marked Done)
        prev_idx = self._current_idx
        if 0 <= prev_idx < len(self._screenshots) and prev_idx != row:
            prev_path = self._screenshots[prev_idx]
            if prev_path.name not in self._screenshots_done:
                self._learn_layout_for(prev_path)
        self._current_idx = row; path = self._screenshots[row]; stype = self._screen_types.get(path.name, 'UNKNOWN')
        # Update Done toggle button state
        is_done = path.name in self._screenshots_done
        self._btn_done.blockSignals(True)
        self._btn_done.setChecked(is_done)
        self._btn_done.setText('↩ Back to Edit' if is_done else '✓ Mark Done')
        self._btn_done.blockSignals(False)
        self._refresh_mark_done_btn()
        self._ann_widget.set_locked(is_done)
        self._ann_widget.load_image(path); self._exit_manual_bbox_mode(); self._update_screen_type_ui(stype)
        from PySide6.QtCore import QTimer
        QTimer.singleShot(100, self._ann_widget.setFocus)
        # Clear Item Name and reset completer when switching to a new screenshot
        self._completer.setCompletionPrefix('')
        self._name_edit.blockSignals(True)
        self._name_edit.clear()
        self._name_edit.blockSignals(False)
        cached = self._recognition_cache.get(path.name)
        if self._mode == 'fast_correction':
            from warp.debug import log as _wlog
            _wlog.info(
                f'Fast Correction: _load_screenshot row={row} path.name={path.name!r} '
                f'cache_hit={cached is not None} '
                f'cache_size={len(cached) if cached else 0} '
                f'all_cache_keys={list(self._recognition_cache.keys())[:3]}')
        if cached is not None: self._populate_review_panel(cached, stype)
        else:
            self._populate_review_panel([], stype)
            if not self._recognition_items:
                if stype == 'DISCARD': self._review_summary.setText('Discarded — not a build screenshot')
                elif stype in ('SKILLS', 'SPACE_SKILLS', 'GROUND_SKILLS'): self._review_summary.setText('Skills screen — recognition not yet supported')
                elif stype == 'UNKNOWN': self._review_summary.setText('Detecting screen type...')
                else: self._review_summary.setText('Click Auto-Detect to recognise items on this screenshot.')
        self._update_add_bbox_btn()

    def _show_file_list_context_menu(self, pos):
        item = self._file_list.itemAt(pos)
        if not item: return
        menu = QMenu(self)
        row = self._file_list.row(item)
        if row < 0 or row >= len(self._screenshots): return
        path = self._screenshots[row]
        current_stype = self._screen_types.get(path.name, 'UNKNOWN')
        for key in SCREEN_TYPE_LABELS:
            icon = SCREEN_TYPE_ICONS.get(key, '')
            label = SCREEN_TYPE_LABELS[key]
            action = menu.addAction(f'{icon} {label}')
            action.setData(key)
            action.setCheckable(True)
            if key == current_stype:
                action.setChecked(True)
        action = menu.exec(self._file_list.mapToGlobal(pos))
        if action:
            self._on_type_override_changed(action.data())

    def _update_screen_type_ui(self, stype: str):
        icon = SCREEN_TYPE_ICONS.get(stype, '?')
        label = SCREEN_TYPE_LABELS.get(stype, 'Unknown')
        self._screen_type_badge.setText(f'Screen: {icon} {label}')
        self._refresh_slot_combo(stype)
        _NO_BBOX_TYPES = ('SPECIALIZATIONS', 'SKILLS', 'SPACE_SKILLS', 'GROUND_SKILLS')
        is_spec    = (stype in _NO_BBOX_TYPES)
        is_discard = (stype == 'DISCARD')
        is_locked  = self._is_current_locked()
        editable   = not is_spec and not is_discard and not is_locked
        self._slot_combo.setEnabled(editable)
        self._name_edit.setEnabled(editable)
        self._btn_accept.setEnabled(editable)
        self._tier_combo.setEnabled(editable)
        self._ship_type_combo.setEnabled(editable)
        if is_discard:
            self._slot_combo.blockSignals(True)
            self._slot_combo.clear()
            self._slot_combo.blockSignals(False)
            self._name_edit.blockSignals(True)
            self._name_edit.clear()
            self._name_edit.blockSignals(False)
            self._name_edit.setPlaceholderText(
                'Discarded — not a build screenshot')
        elif is_spec:
            self._slot_combo.blockSignals(True)
            self._slot_combo.clear()
            self._slot_combo.blockSignals(False)
            self._name_edit.blockSignals(True)
            self._name_edit.clear()
            self._name_edit.blockSignals(False)
            self._name_edit.setPlaceholderText(
                'No bboxes — recognition not yet supported for this screen type')
        elif is_locked:
            self._name_edit.setPlaceholderText(
                'Screenshot is marked Done — press ↩ Back to Edit to modify')
        else:
            self._name_edit.setPlaceholderText("Item name (or leave blank for 'Unknown')")

    def _refresh_slot_combo(self, stype: str, keep_slot: str = ''):
        """Rebuild slot combo for screen type, hiding confirmed NON_ICON_SLOTS.

        keep_slot: slot of the currently displayed item — always kept visible
        so the user can read/edit an already-confirmed Ship Name/Type/Tier bbox.
        """
        group_key = SCREEN_TO_SLOT_GROUP.get(stype, 'SPACE_EQ')
        slots = SLOT_GROUPS.get(group_key, ALL_SLOTS)
        # Hide confirmed NON_ICON_SLOTS, but always show the currently active one
        confirmed_non_icon: set[str] = set()
        if self._current_idx >= 0:
            path = self._screenshots[self._current_idx]
            confirmed_non_icon = {
                ann.slot for ann in self._data_mgr.get_annotations(path)
                if ann.state == AnnotationState.CONFIRMED and ann.slot in NON_ICON_SLOTS
            }
        confirmed_non_icon.discard(keep_slot)
        current_slot = self._slot_combo.currentText()
        self._slot_combo.blockSignals(True)
        self._slot_combo.clear()
        for s in slots:
            if s not in confirmed_non_icon:
                self._slot_combo.addItem(s)
        idx = self._slot_combo.findText(current_slot)
        if idx >= 0:
            self._slot_combo.setCurrentIndex(idx)
        self._slot_combo.blockSignals(False)

    def _on_type_override_changed(self, stype: str):
        if self._current_idx < 0: return
        path = self._screenshots[self._current_idx]
        self._screen_types[path.name] = stype
        self._screen_types_manual.add(path.name)
        self._screen_types_ml_auto.discard(path.name)
        self._recognition_cache.pop(path.name, None)
        self._data_mgr.set_screen_type(path, stype, user_confirmed=True)
        try:
            import cv2
            from warp.recognition.screen_classifier import ScreenTypeClassifier
            img = cv2.imread(str(path))
            if img is not None:
                ScreenTypeClassifier.add_session_example(img, stype)
        except Exception:
            pass
        item = self._file_list.item(self._current_idx)
        if item:
            sc_icon = SCREEN_TYPE_ICONS.get(stype, '?')
            label   = SCREEN_TYPE_LABELS.get(stype, 'Unknown')
            self._file_list.blockSignals(True)
            item.setText(f'{sc_icon} {label}\n  {_disp_name(path.name)}')
            item.setCheckState(Qt.CheckState.Checked)
            item.setIcon(_get_user_icon())
            self._file_list.blockSignals(False)
        self._update_screen_type_ui(stype)
        self._update_progress()
        log.info(f'Manual screen type override: {path.name} → {stype}')
        # DISCARD screenshots have no items to review — auto-Mark Done
        if stype == 'DISCARD' and path.name not in self._screenshots_done:
            self._recognition_items = []
            self._review_list.clear()
            self._review_summary.setText('Discarded — not a build screenshot')
            self._screenshots_done.add(path.name)
            self._save_done()
            self._btn_done.blockSignals(True)
            self._btn_done.setChecked(True)
            self._btn_done.setText('↩ Back to Edit')
            self._btn_done.blockSignals(False)
            self._ann_widget.set_locked(True)
            self._update_file_list_color(self._current_idx)
            log.info(f'DISCARD: auto-marked Done for {path.name}')

    # _save_screen_type_example removed — logic consolidated into
    # _on_type_override_changed (dropdown) and _on_file_item_changed (checkbox).

    def _seed_matcher_from_confirmed(self, path: Path):
        """
        Prime SETSIconMatcher with confirmed training-data crops before Auto-Detect.
        Delegates to seed_from_training_data (all confirmed crops, guarded against
        re-seeding). New in-session confirmations are already added live via
        add_session_example in _on_accept / _on_accept_all.
        """
        try:
            from warp.recognition.icon_matcher import SETSIconMatcher
            SETSIconMatcher.seed_from_training_data(userdata.training_data_dir())
            SETSIconMatcher.seed_from_community_crops()
        except Exception as e:
            log.warning(f'seed_matcher_from_confirmed failed: {e}')

    def _on_auto_detect(self):
        if self._current_idx < 0: return
        if self._is_current_locked():
            self.statusBar().showMessage(
                'Auto-Detect blocked: screenshot is marked Done — '
                'press ↩ Back to Edit to modify.', 6000)
            return
        path = self._screenshots[self._current_idx]
        stype = self._screen_types.get(path.name, 'UNKNOWN')

        # Clear detection logs from the UI before starting a new run.
        # WARP CORE's own logs live on the 'detection_core' channel — using
        # 'detection' here used to wipe the WARP window's log buffer instead.
        import warp.debug
        warp.debug.clear_logs('detection_core')

        # Seed the icon matcher with all confirmed crops from this image
        # so Auto-Detect benefits from what the user has already confirmed
        self._seed_matcher_from_confirmed(path)

        # Preserve every existing row (confirmed AND pending) — auto-detect is
        # "find what's not on screen yet", not "rebuild from scratch". The
        # recognition_done merge drops new items that overlap existing bboxes
        # (IoU) or duplicate a SINGLE_INSTANCE_SLOTS row.
        #
        # Source = `_recognition_items` (the merged view that already
        # includes disk-confirmed annotations injected by `_populate_review_panel`),
        # not `_recognition_cache`. The cache is empty on first entry to a
        # screenshot, so reading it would hide previously confirmed bboxes
        # from the IoU dedup and let fresh detections re-emit near-identical
        # duplicates on top of them.
        existing = list(self._recognition_items)
        self._recognition_cache.pop(path.name, None)
        self._start_recognition(path, stype, preserve_existing=existing)

    def _on_detect_screen_types(self):
        if not self._screenshots: return
        self._start_screen_type_detection('detect_screen_types_button', max_iter=10)

    def _folder_environment(self) -> str:
        """Infer SPACE / GROUND environment from screen types in the open folder.

        Rules (user-defined):
          - space signal = any image classified SPACE_EQ / SPACE_MIXED /
            SPACE_BOFFS / SPACE_TRAITS
          - ground signal = any image classified GROUND_EQ / GROUND_MIXED /
            GROUND_BOFFS / GROUND_TRAITS
          - only space → 'SPACE'; only ground → 'GROUND';
            both → 'SPACE' (let user force); neither → 'SPACE'
        Generic TRAITS / BOFFS / SPECIALIZATIONS labels do not contribute —
        they're ambiguous and would create a circular signal.
        """
        space_sig  = {'SPACE_EQ', 'SPACE_MIXED', 'SPACE_BOFFS', 'SPACE_TRAITS'}
        ground_sig = {'GROUND_EQ', 'GROUND_MIXED', 'GROUND_BOFFS', 'GROUND_TRAITS'}
        has_space = any(s in space_sig  for s in self._screen_types.values())
        has_ground = any(s in ground_sig for s in self._screen_types.values())
        if has_ground and not has_space:
            return 'GROUND'
        return 'SPACE'

    def _promote_generic_stype(self, path: Path, stype: str) -> str:
        """Pass-through. Auto-promotion of generic TRAITS/BOFFS based on
        folder context was removed: generic = "mixed evidence" (image may
        contain both space and ground sections) and narrows detection
        scope only when the user manually picks a SPACE_* / GROUND_*
        variant. _folder_environment() remains as a helper for other call
        sites (e.g. picking a default write target downstream)."""
        return stype

    def _start_recognition(self, path: Path, stype: str, preserve_existing: list | None = None):
        if self._recog_worker and self._recog_worker.isRunning():
            self._recog_worker.requestInterruption()
            self._recog_worker.wait(2000)
        # Keep the existing list visible while detection runs — _populate_review_panel
        # rebuilds it once results arrive (merged list = preserve_existing + new).
        self._review_summary.setText('Running recognition...')
        self._set_review_buttons_enabled(False)
        # Determinate bar driven by importer's per-stage progress callback
        # (same signal WARP listens to). `_recog_dlg` is a sentinel so
        # done/error/cancel callbacks scope cleanup to the run we just started.
        self._recog_dlg = '__statusbar__'
        self._status_progress.start(determinate=True, maximum=100)
        icon = SCREEN_TYPE_ICONS.get(stype, '?')
        label = SCREEN_TYPE_LABELS.get(stype, stype)
        self.statusBar().showMessage(
            f'Matching icons against SETS library —  {icon} {label}   {_disp_name(path.name)}'
        )
        # Filter out legacy rows missing detector-derived display fields
        # (seat_key for BOFFs, slot_index for everything) so the layout
        # pass re-detects them with today's geometry. The disk-confirmed
        # name is re-applied to the fresh dict by `_populate_review_panel`,
        # which re-reads disk annotations and re-attaches fresh
        # seat_key / slot_index after the merge — no user confirmation
        # is lost; only the per-group sort key gets upgraded to today's
        # detection order, restoring L→R BOFF / T→B trait / numeric EQ
        # ordering within each parent row.
        def _is_legacy(ri: dict) -> bool:
            is_boff = (ri.get('slot', '') or '').startswith('Boff')
            if is_boff and not (ri.get('seat_key') or ''):
                return True
            if ri.get('slot_index') is None:
                return True
            return False
        _preserve = [ri for ri in (preserve_existing or [])
                     if not _is_legacy(ri)]
        skip_bboxes = [ri.get('bbox') for ri in _preserve
                       if ri.get('bbox')]
        self._recog_worker = RecognitionWorker(path, stype, self._sets, parent=self,
                                                skip_bboxes=skip_bboxes)
        self._recog_worker.progress.connect(self._on_recognition_progress)
        self._recog_worker.finished.connect(
            lambda items: self._on_recognition_done(path.name, stype, items,
                                                    preserve_existing=_preserve))
        self._recog_worker.error.connect(self._on_recognition_error)
        self._recog_worker.start()

    def _on_recognition_progress(self, pct: int, label: str):
        if not self._recog_dlg:
            return
        self._status_progress.set_progress(pct)
        self.statusBar().showMessage(label)

    @staticmethod
    def _bbox_iou(a: tuple, b: tuple) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ix1 = max(ax, bx); iy1 = max(ay, by)
        ix2 = min(ax + aw, bx + bw); iy2 = min(ay + ah, by + bh)
        iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0

    def _merge_recognition(self, existing: list, new: list) -> list:
        """Add only new detections that neither overlap existing bboxes nor
        duplicate a SINGLE_INSTANCE_SLOTS row. NON_ICON_SLOTS pairs (Ship Tier
        / Ship Type) are exempt from the IoU rule — they legitimately sit in
        the same image strip and may touch."""
        IOU_THRESHOLD = 0.3
        single_taken = {ri.get('slot') for ri in existing
                        if ri.get('slot') in SINGLE_INSTANCE_SLOTS}
        existing_boxes = [(ri.get('bbox'), ri.get('slot', '')) for ri in existing
                          if ri.get('bbox')]
        kept_new: list = []
        for ri in new:
            slot = ri.get('slot', '')
            bbox = ri.get('bbox')
            if slot in SINGLE_INSTANCE_SLOTS and slot in single_taken:
                log.debug(f'merge: drop {slot!r} — single-instance already taken')
                continue
            if bbox is not None:
                overlaps = False
                for ebbox, eslot in existing_boxes:
                    if ebbox is None:
                        continue
                    if slot in NON_ICON_SLOTS and eslot in NON_ICON_SLOTS:
                        continue
                    if self._bbox_iou(bbox, ebbox) >= IOU_THRESHOLD:
                        overlaps = True
                        log.debug(f'merge: drop {slot!r} bbox={bbox} — '
                                  f'IoU overlap with existing {eslot!r} bbox={ebbox}')
                        break
                if overlaps:
                    continue
            kept_new.append(ri)
            if slot in SINGLE_INSTANCE_SLOTS:
                single_taken.add(slot)
            if bbox is not None:
                existing_boxes.append((bbox, slot))
        log.info(f'merge: preserved {len(existing)}, kept {len(kept_new)}/{len(new)} new')
        return existing + kept_new

    def _on_recognition_done(self, filename: str, stype: str, items: list,
                             preserve_existing: list | None = None):
        if self._recog_dlg:
            self._status_progress.finish()
            self._recog_dlg = None
            self.statusBar().showMessage(f'Recognition done — {len(items)} item(s).')
        merged = self._merge_recognition(preserve_existing or [], items)
        self._recognition_cache[filename] = merged
        if self._current_idx >= 0 and self._screenshots[self._current_idx].name == filename:
            self._populate_review_panel(merged, stype)
            # Overlay the EQ geometry grid captured during detection (cleared on next image load)
            geom = getattr(self._recog_worker, 'eq_geom', None) if self._recog_worker else None
            self._ann_widget.set_eq_geom(geom)
            # Run auto-accept after panel is populated
            self._run_auto_accept()

    def _ocr_empty_non_icon_items(self):
        """Re-run OCR for any confirmed NON_ICON_SLOT items that have an empty name.

        This covers the case where Ship Type/Tier was confirmed before OCR finished
        (or OCR failed), leaving a blank name saved to disk.  On every panel refresh
        we detect such entries and kick off OCRWorker to fill them in.
        """
        if self._current_idx < 0:
            return
        try:
            import cv2
            path = self._screenshots[self._current_idx]
            img = None  # lazy-load
            if self._ship_type_combo.count() == 0:
                self._populate_ship_type_combo()
            v_tiers = [self._tier_combo.itemText(i) for i in range(self._tier_combo.count())]
            v_types = [self._ship_type_combo.itemText(i) for i in range(self._ship_type_combo.count())]
            for row, ri in enumerate(self._recognition_items):
                if ri.get('slot') not in NON_ICON_SLOTS:
                    continue
                if ri.get('slot') == 'Ship Name':
                    continue  # position-only, never store name
                if ri.get('name'):
                    continue  # already has a name
                bbox = ri.get('bbox')
                if not bbox:
                    continue
                if img is None:
                    img = cv2.imread(str(path))
                    if img is None:
                        break
                x, y, w, h = bbox
                crop = img[y:y+h, x:x+w].copy()
                if crop.size == 0:
                    continue
                ri['crop_bgr'] = crop
                worker = OCRWorker(row, crop, ri['slot'], v_tiers, v_types, parent=self)
                worker.finished.connect(self._on_ocr_finished)
                worker.start()
                if not hasattr(self, '_ocr_workers'):
                    self._ocr_workers = []
                self._ocr_workers.append(worker)
        except Exception as _e:
            from warp.debug import log as _sl
            _sl.debug(f'_ocr_empty_non_icon_items: {_e}')

    def _on_recognition_error(self, msg: str):
        if self._recog_dlg:
            self._status_progress.finish()
            self._recog_dlg = None
        if msg == 'Cancelled':
            self._review_summary.setText('Recognition cancelled.')
            self.statusBar().showMessage('Recognition cancelled.')
            return
        self._review_summary.setText(f'Recognition error: {msg}')
        self.statusBar().showMessage(f'Recognition error: {msg}')

    def _on_recognition_cancelled(self):
        if self._recog_worker and self._recog_worker.isRunning():
            self._recog_worker.requestInterruption()
            self._recog_worker.wait(2000)
        if self._recog_dlg:
            self._status_progress.finish()
            self._recog_dlg = None
        self._review_summary.setText('Recognition cancelled.')
        self.statusBar().showMessage('Recognition cancelled.')

    def _order_items_for_review(self, items: list, stype: str) -> list:
        """Reorder `_recognition_items` for the review tree and stamp each
        item with `_group_label` — the parent-row label the
        `_ReviewListAdapter` should put the item under.

        Delegates ordering and seat-aware BOFF grouping to
        `warp.recognition.boff_keys.order_items_for_display`, the single
        source of truth shared with WARP Results. So BOFF rows that share
        a physical seat (e.g. Lieutenant Tactical) land under a unified
        'Boff Tactical' parent here too instead of one parent per raw
        seat key.
        """
        from warp.recognition.boff_keys import order_items_for_display
        from warp.warp_importer import DISPLAY_CANONICAL_ORDER
        build_type = self._STYPE_TO_BUILD.get(stype, 'SPACE')
        # Display-only override: a 'TRAITS' screen mixes ground + space
        # traits, but `_STYPE_TO_BUILD` maps it to 'SPACE_TRAITS' for
        # the calibration / icon-matcher paths. For the review panel we
        # want the full mixed canonical so ground rows land with the
        # other trait rows instead of falling through to the alphabetical
        # fallback at the end.
        canonical_bt = 'TRAITS' if stype == 'TRAITS' else build_type
        canonical = [sd['name'] for sd in _SLOT_ORDER.get(canonical_bt, [])]
        flat: list = []
        _boff_diag: list[str] = []
        _group_order: list[str] = []
        for label, group in order_items_for_display(
            items, canonical,
            fallback_canonical_slots=DISPLAY_CANONICAL_ORDER,
        ):
            _group_order.append(f'{label}({len(group)})')
            for ri in group:
                ri['_group_label'] = label
                flat.append(ri)
            if label.startswith('Boff'):
                first = group[0] if group else {}
                bb = first.get('bbox') if isinstance(first, dict) else None
                sk = first.get('seat_key', '') if isinstance(first, dict) else ''
                _boff_diag.append(f'{label!r}@bbox={bb} sk={sk!r}')
        from warp.debug import log as _sl
        _sl.info(f'order_for_review bt={build_type} canonical_bt={canonical_bt} '
                 f'canonical={canonical[:8]}{"..." if len(canonical) > 8 else ""}')
        _sl.info('order_for_review groups: ' + ' → '.join(_group_order))
        if _boff_diag:
            _sl.info('order_for_review BOFF order: '
                     + ' | '.join(_boff_diag))
        return flat

    def _populate_review_panel(self, items: list, stype: str):
        # Fast Correction Mode is a pure WARP correction loop — the user
        # is here to fix exactly what WARP detected, so we must NOT merge
        # in disk-confirmed annotations, NOT raise community-conflict
        # state, and NOT silently substitute equivalence-class names.
        # Every item renders as pending with WARP's raw values.
        if self._mode == 'fast_correction':
            from warp.debug import log as _wlog
            _wlog.info(
                f'Fast Correction: populate_review_panel items={len(items)} '
                f'stype={stype} current_idx={self._current_idx}')
            self._recognition_items = self._order_items_for_review(
                list(items), stype)
            self._review_list.clear()
            self._review_summary.setText('')
            self._set_review_buttons_enabled(False)
            for ri in self._recognition_items:
                # Respect each item's actual state so a re-run of
                # Auto-Detect (which goes through `_merge_recognition`
                # and preserves prior items) keeps already-confirmed
                # rows visually green on the list. The canvas already
                # reads `state` directly from the item; without this
                # the two views disagreed after a second detection.
                self._add_review_row(
                    ri.get('name', ''), ri.get('slot', ''),
                    ri.get('conf', 0.0),
                    confirmed=(ri.get('state') == 'confirmed'),
                    cross_check_failed=ri.get('cross_check_failed', False),
                    auto_confirmed=ri.get('auto_confirmed', False),
                    conflict_disk_name='',
                    group_label=ri.get('_group_label'))
            # Slot / Idx column neutrality is now enforced inside
            # `_populate_review_item` itself (cols 0-1 always white), and
            # `refresh_parent_of` mirrors that onto each group header — so
            # no extra post-pass is needed here.
            self._ann_widget.set_review_items(self._recognition_items)
            self._ann_widget.set_selected_row(-1)
            n = len(self._recognition_items)
            matched = sum(1 for i in self._recognition_items if i.get('name'))
            icon = SCREEN_TYPE_ICONS.get(stype, '?')
            label = SCREEN_TYPE_LABELS.get(stype, stype)
            ship = (self._recognition_items[0].get('ship_name') or '--') \
                if self._recognition_items else '--'
            self._review_summary.setText(
                f'{matched}/{n} from WARP — confirm or correct each.  '
                f'Ship: {ship}  {icon} {label}')
            self._set_review_buttons_enabled(n > 0)
            self._refresh_slot_combo(stype)
            if n > 0:
                self._review_list.setCurrentRow(0)
            self._refresh_mark_done_btn()
            return
        confirmed_by_id: dict[str, dict] = {}
        if self._current_idx >= 0:
            path = self._screenshots[self._current_idx]
            for ann in self._data_mgr.get_annotations(path):
                if ann.state == AnnotationState.CONFIRMED:
                    log.debug(f'populate: confirmed from disk slot={ann.slot!r} '
                              f'bbox={ann.bbox} name={ann.name!r} ann_id={ann.ann_id}')
                    _entry: dict = {
                        'name': ann.name, 'slot': ann.slot, 'bbox': ann.bbox,
                        'state': 'confirmed',
                        'auto_confirmed': ann.auto_confirmed,
                        'community_rejected': ann.community_rejected,
                        'conf': ann.ml_conf,          # real ML confidence, 0.0 if unknown
                        'orig_name': ann.ml_name or ann.name,  # what ML originally saw
                        'thumb': None, 'crop_bgr': None, 'ship_name': '', 'ann_id': ann.ann_id,
                    }
                    # Layout fields are optional on disk (legacy entries
                    # may lack them). Only carry forward when present so
                    # the existing `if ri.get('seat_key')` checks down
                    # the merge path keep their "missing → re-derive"
                    # semantics for old data.
                    if ann.seat_key:
                        _entry['seat_key'] = ann.seat_key
                    if ann.slot_index >= 0:
                        _entry['slot_index'] = ann.slot_index
                    confirmed_by_id[ann.ann_id] = _entry
        from warp.trainer.training_data import Annotation as _Ann
        # Build a parallel index of disk-confirmed entries keyed by bbox
        # so we can recover from slot-format drift: an annotation saved
        # as 'Boff Tactical' has a different ann_id than the same bbox
        # detected today as 'Boff Seat L[T]_<y>', so the ann_id-based
        # match fails. Bbox is the stable physical anchor.
        confirmed_bbox_list: list[tuple[tuple, str]] = [
            (tuple(_ci.get('bbox') or ()), _aid)
            for _aid, _ci in confirmed_by_id.items()
            if _ci.get('bbox')
        ]
        IOU_RECOVER = 0.5
        def _find_legacy_aid(fresh_bbox) -> str | None:
            """Find a disk-confirmed entry whose bbox best overlaps
            `fresh_bbox`. Handles detector drift between save-time and
            today (a few px shift). Returns the legacy ann_id if a match
            above `IOU_RECOVER` exists, else None."""
            best_aid, best_iou = None, IOU_RECOVER
            for bb, aid in confirmed_bbox_list:
                if not bb or aid in seen_ids:
                    continue
                iou = self._bbox_iou(tuple(fresh_bbox), bb)
                if iou > best_iou:
                    best_iou, best_aid = iou, aid
            return best_aid
        merged: list[dict] = []
        seen_ids: set[str] = set()
        # Track disk-confirmed entries that gained seat_key / slot_index
        # via fresh Auto-Detect — flushed once after the loop so the
        # next restart can render the per-seat layout straight from
        # annotations.json without forcing the user to re-detect.
        _layout_backfilled = False
        for ri in items:
            bbox = ri.get('bbox')
            # Capture fresh detector geometry before any merge branch
            # reassigns `ri` from a disk-confirmed dict. Legacy
            # annotations don't carry seat_key / slot_index (both are
            # newer than the JSON schema), so `ri = dict(confirmed)`
            # would drop them and:
            #   - collapse per-seat grouping back to one profession row
            #   - lose left-to-right child order (sort falls to name)
            _fresh_seat_key = (ri.get('seat_key') or '') if isinstance(ri, dict) else ''
            _fresh_slot_index = ri.get('slot_index') if isinstance(ri, dict) else None
            if bbox:
                aid = _Ann(bbox=bbox, slot=ri.get('slot',''), name=ri.get('name','')).ann_id
                # Bbox-IoU fallback: when the fresh seat-keyed slot
                # doesn't ann_id-match, find the disk entry whose bbox
                # overlaps. Promote it onto the fresh ann_id, overwrite
                # slot + seat_key with today's geometry. User's
                # confirmed name/state survive; group_items_by_seat now
                # sees the seat-keyed slot and groups per physical seat.
                if aid not in confirmed_by_id:
                    legacy_aid = _find_legacy_aid(bbox)
                    if legacy_aid:
                        legacy = confirmed_by_id[legacy_aid]
                        promoted = dict(legacy)
                        promoted['slot'] = ri.get('slot') or legacy.get('slot', '')
                        if ri.get('seat_key'):
                            promoted['seat_key'] = ri['seat_key']
                        promoted['bbox'] = bbox  # adopt today's geometry
                        promoted['ann_id'] = legacy_aid
                        confirmed_by_id[aid] = promoted
                        seen_ids.add(legacy_aid)
                        log.debug(
                            f'populate: IoU-recovered legacy slot '
                            f'{legacy.get("slot")!r} → {promoted["slot"]!r} '
                            f'(seat_key={promoted.get("seat_key","")!r})'
                        )
                if aid in confirmed_by_id:
                    confirmed = confirmed_by_id[aid]
                    fresh_name = (ri.get('name') or '').strip()
                    saved_name = (confirmed.get('name') or '').strip()
                    # If fresh recognition disagrees with what's stored on
                    # disk for the same bbox+slot, prefer the fresh value
                    # and demote to pending so the user re-confirms. The
                    # previous behaviour silently kept the disk name —
                    # which let stale Ship Tier annotations (e.g. T1)
                    # shadow a freshly-detected T6-X2 because ann_id is
                    # hash(bbox+slot) and ignores name.
                    if fresh_name and saved_name and fresh_name != saved_name:
                        was_user_confirmed = not bool(confirmed.get('auto_confirmed', False))
                        already_rejected = (confirmed.get('community_rejected', '') or '').strip()
                        # Equivalence-class shortcut: when both names share
                        # the same icon art (admin-curated list mirrored from
                        # HF, see warp.tools.icon_equivalence), no human can
                        # disambiguate them from a crop — so we silently keep
                        # the user's disk choice without raising a conflict.
                        equivalent = False
                        if was_user_confirmed and self._sync_client is not None:
                            try:
                                equivalent = self._sync_client.are_equivalent(
                                    saved_name, fresh_name)
                            except Exception as e:
                                log.debug(f'populate: are_equivalent failed: {e}')
                        if equivalent:
                            log.info(
                                f'populate: community proposes {fresh_name!r} '
                                f'but it shares icon art with disk={saved_name!r} '
                                f'(equivalence class) — keeping disk silently '
                                f'for slot={ri.get("slot")!r} bbox={bbox}'
                            )
                            ri = dict(confirmed)
                        elif was_user_confirmed and already_rejected and already_rejected == fresh_name:
                            # User has previously resolved a conflict against
                            # exactly this community proposal — silently keep
                            # the user's pick. The community DB hasn't changed
                            # its mind, so there's nothing new to verify.
                            log.info(
                                f'populate: community proposes {fresh_name!r} again — '
                                f'previously rejected by user for slot={ri.get("slot")!r} '
                                f'bbox={bbox}; keeping disk={saved_name!r} silently'
                            )
                            ri = dict(confirmed)
                        elif was_user_confirmed:
                            # User confirmed this bbox manually, but the
                            # current detection (knowledge / embedder /
                            # session) disagrees. Don't silently overwrite
                            # the user's vote and don't silently keep it
                            # either — surface as 'community_conflict' so
                            # the user re-verifies. On Accept this rejoins
                            # the normal save path (which contributes a
                            # community vote), strengthening signal.
                            log.info(
                                f'populate: COMMUNITY CONFLICT — '
                                f'slot={ri.get("slot")!r} bbox={bbox} '
                                f'disk(user)={saved_name!r} '
                                f'community={fresh_name!r} → state=community_conflict'
                            )
                            ri = dict(ri)
                            ri['state'] = 'community_conflict'
                            ri['auto_confirmed'] = False
                            ri['disk_name'] = saved_name
                            ri['ann_id'] = confirmed.get('ann_id', aid)
                        else:
                            log.info(
                                f'populate: fresh recognition disagrees with '
                                f'auto-confirmed annotation — slot={ri.get("slot")!r} '
                                f'bbox={bbox} disk={saved_name!r} fresh={fresh_name!r} '
                                f'→ using fresh, state=pending'
                            )
                            ri = dict(ri)
                            ri['state'] = 'pending'
                            ri['auto_confirmed'] = False
                            ri['ann_id'] = confirmed.get('ann_id', aid)
                    else:
                        ri = dict(confirmed)
                # Re-attach fresh detector fields after any disk-merge
                # branch: `ri = dict(confirmed)` returns a legacy dict
                # with no seat_key / slot_index, but the display layer
                # needs both — seat_key for per-seat grouping,
                # slot_index for left-to-right child order within a
                # group.
                _need_reattach = (
                    (_fresh_seat_key and not ri.get('seat_key'))
                    or (_fresh_slot_index is not None
                        and ri.get('slot_index') is None)
                )
                if _need_reattach:
                    ri = dict(ri)
                    if _fresh_seat_key and not ri.get('seat_key'):
                        ri['seat_key'] = _fresh_seat_key
                    if _fresh_slot_index is not None \
                            and ri.get('slot_index') is None:
                        ri['slot_index'] = _fresh_slot_index
                    # Persist the layout fields onto the matching disk
                    # annotation so a future cold load skips the
                    # re-attach path entirely. Only fires when the merge
                    # actually consumed a confirmed entry (state would
                    # have flipped from pending), so we don't write
                    # seat_key onto unrelated fresh items.
                    disk_aid = ri.get('ann_id') or aid
                    if (ri.get('state') == 'confirmed'
                            and disk_aid in {*confirmed_by_id.keys()}
                            and self._current_idx >= 0):
                        if self._data_mgr.update_layout_fields(
                            self._screenshots[self._current_idx],
                            disk_aid,
                            seat_key=ri.get('seat_key', '') or '',
                            slot_index=ri.get('slot_index')
                                if isinstance(ri.get('slot_index'), int)
                                else -1,
                        ):
                            _layout_backfilled = True
                seen_ids.add(aid)
            merged.append(ri)
        if _layout_backfilled:
            try:
                self._data_mgr.save()
                log.info('populate: persisted backfilled seat_key/slot_index '
                         'to annotations.json')
            except Exception as _e:
                log.warning(f'populate: layout-field save failed: {_e}')
        for aid, ci in confirmed_by_id.items():
            if aid not in seen_ids:
                merged.append(ci)
        self._recognition_items = self._order_items_for_review(merged, stype)
        # Auto-accept high-conf items before drawing the list
        self._apply_auto_accept()
        self._review_list.clear()
        self._review_summary.setText('')
        self._set_review_buttons_enabled(False)
        for ri in self._recognition_items:
            _conflict = ri.get('disk_name', '') if ri.get('state') == 'community_conflict' else ''
            self._add_review_row(ri['name'], ri['slot'], ri.get('conf', 0.0), confirmed=(ri.get('state') == 'confirmed'), cross_check_failed=ri.get('cross_check_failed', False), auto_confirmed=ri.get('auto_confirmed', False), conflict_disk_name=_conflict, group_label=ri.get('_group_label'))
        self._ann_widget.set_review_items(self._recognition_items)
        self._ann_widget.set_selected_row(-1)
        n = len(self._recognition_items)
        matched = sum(1 for i in self._recognition_items if i.get('name'))
        icon = SCREEN_TYPE_ICONS.get(stype, '?')
        label = SCREEN_TYPE_LABELS.get(stype, stype)
        ship = (self._recognition_items[0].get('ship_name') or '--') if self._recognition_items else '--'
        self._review_summary.setText(f'{matched}/{n} identified  Ship: {ship}  {icon} {label}')
        self._set_review_buttons_enabled(n > 0)
        # Hide confirmed NON_ICON_SLOTS from slot combo for this image
        self._refresh_slot_combo(stype)
        if n > 0:
            self._review_list.setCurrentRow(0)
        # For confirmed NON_ICON_SLOT items with empty name (e.g. Ship Type confirmed
        # before OCR finished), re-run OCR now so the name is filled in.
        self._ocr_empty_non_icon_items()
        self._refresh_mark_done_btn()

    # ── Review list (5-column QTreeWidget) helpers ─────────────────────
    #
    # Columns: 0=Slot, 1=Idx (in-slot ordinal), 2=Item, 3=Conf, 4=Status.
    # State color is applied to every column so the row reads uniformly.
    # The slot is stored in col-0 UserRole so duplicate slots (e.g. multiple
    # Boff Tactical rows) get a 1-based index in col 1 — matching the
    # `slot#N` keying used by `_log_match_summary`.

    _AUTO_COLOR        = '#5cbfff'   # light blue — Auto-confirmed (matches canvas)
    _CONFIRMED_COLOR   = '#7effc8'   # green — user-confirmed
    _CONFLICT_COLOR    = '#ff9a3c'   # orange — community conflict
    _VIRTUAL_CONFIRMED = '#888888'   # grey — confirmed empty/inactive
    _VIRTUAL_PENDING   = '#aaaaaa'   # lighter grey — pending virtual
    _CROSS_CHECK_COLOR = '#ffcc00'   # gold — slot type mismatch
    _UNMATCHED_COLOR   = '#ff5555'   # red — no name / low conf
    _MED_COLOR         = '#ff8888'   # medium conf
    _HIGH_COLOR        = '#ffaaaa'   # high conf

    def _review_row_visuals(self, name: str, conf: float, *,
                            confirmed: bool, cross_check_failed: bool,
                            auto_confirmed: bool,
                            conflict_disk_name: str) -> tuple[str, str, str]:
        """Return (item_text, conf_text, status_text, color_hex) for the row.

        Centralised so `_add_review_row` and the inline refresh sites
        (rematch, OCR finish, accept, slot-change scanning) produce
        identical 5-column output without duplicating the state ladder.
        """
        is_virtual  = name in VIRTUAL_ITEM_NAMES
        is_conflict = bool(conflict_disk_name)

        if is_conflict:
            item_text   = f'[CONFLICT] disk: {conflict_disk_name or "—"} | community: {name or "—"}'
            status_text = 'Conflict'
            color       = self._CONFLICT_COLOR
        elif confirmed and is_virtual:
            item_text   = '[empty slot]' if name == '__empty__' else '[inactive slot]'
            status_text = 'Empty' if name == '__empty__' else 'Inactive'
            color       = self._VIRTUAL_CONFIRMED
        elif confirmed and auto_confirmed:
            item_text   = name or '—'
            status_text = 'Auto'
            color       = self._AUTO_COLOR
        elif confirmed:
            item_text   = name or '—'
            status_text = 'Confirmed'
            color       = self._CONFIRMED_COLOR
        elif is_virtual:
            item_text   = '[empty slot]' if name == '__empty__' else '[inactive slot]'
            status_text = 'Empty' if name == '__empty__' else 'Inactive'
            color       = self._VIRTUAL_PENDING
        elif cross_check_failed:
            item_text   = f'⚠ {name or "— unmatched —"}'
            status_text = 'Type ✕'
            color       = self._CROSS_CHECK_COLOR
        elif not name:
            item_text   = '— unmatched —'
            status_text = 'Unmatched'
            color       = self._UNMATCHED_COLOR
        elif conf >= CONF_HIGH:
            item_text   = name
            status_text = 'Match'
            color       = self._HIGH_COLOR
        elif conf >= CONF_MEDIUM:
            item_text   = name
            status_text = 'Match'
            color       = self._MED_COLOR
        else:
            item_text   = name
            status_text = 'Low'
            color       = self._UNMATCHED_COLOR

        if conf > 0.0:
            conf_text = f'{conf:.0%}'
        elif confirmed:
            conf_text = '—'           # legacy annotation without saved confidence
        else:
            conf_text = ''

        return item_text, conf_text, status_text, color

    def _slot_ordinal(self, slot: str, exclude_item=None) -> int:
        """1-based in-slot index for `slot`. Counts existing rows whose
        col-0 UserRole equals `slot`, skipping `exclude_item` if given.

        Works for both pre- and post-addItem state: the slot id lives on
        the item itself (col-0 UserRole), so the adapter's grouping is
        irrelevant — we just walk the flat insertion-order list."""
        n = 0
        for i in range(self._review_list.count()):
            it = self._review_list.item(i)
            if it is None or it is exclude_item:
                continue
            if it.data(0, Qt.ItemDataRole.UserRole) == slot:
                n += 1
        return n + 1

    def _populate_review_item(self, item, name: str, slot: str, conf: float, *,
                              confirmed: bool, cross_check_failed: bool,
                              auto_confirmed: bool, conflict_disk_name: str,
                              idx: int | None = None,
                              group_label: str | None = None) -> None:
        slot_disp = _pretty_slot(slot)
        # `group_label` (when given) is the seat-aware parent-row label
        # produced by `order_items_for_display` — e.g. 'Boff Tactical #2'
        # — and it doubles as the adapter's group key. Without it we
        # fall back to the slot-as-key behaviour, which is what manual
        # bbox additions and pre-grouping call sites still rely on.
        group_id   = group_label if group_label else slot
        group_disp = group_label if group_label else slot_disp
        item_text, conf_text, status_text, color = self._review_row_visuals(
            name, conf,
            confirmed=confirmed, cross_check_failed=cross_check_failed,
            auto_confirmed=auto_confirmed, conflict_disk_name=conflict_disk_name,
        )
        # Col 0 carries the group label only on standalone items (those
        # not yet added to the tree). Once `addItem` places the item under
        # a parent, the parent owns the Slot column and the child's col 0
        # stays blank — preserve that here so refresh sites don't
        # reintroduce the slot text on every child row.
        item.setData(0, Qt.ItemDataRole.UserRole, group_id)
        if item.parent() is None:
            item.setText(0, group_disp)
        if idx is None:
            try:
                idx = int(item.text(1)) if item.text(1) else 1
            except ValueError:
                idx = 1
        item.setText(1, str(idx))
        item.setText(2, item_text)
        item.setText(3, conf_text)
        item.setText(4, status_text)

        # Tooltip on the Item column carries the verbose context the old
        # single-cell label used to embed; col 0 (slot) gets a short copy
        # so users hovering near the slot name still see it.
        if conflict_disk_name:
            tooltip = (f'Slot: {slot_disp}\n'
                       f'Disk (your previous confirmation): {conflict_disk_name}\n'
                       f'Community / current detector: {name or "—"}\n\n'
                       f'These disagree. Re-verify the icon and Accept the '
                       f'correct name to cast another community vote.')
        elif confirmed:
            status = 'auto-confirmed by detector' if auto_confirmed else 'confirmed by user'
            conf_line = f'ML recognition: {conf:.1%}' if conf > 0.0 else \
                        'ML recognition: unknown (previous session)'
            tooltip = (f'Slot: {slot_disp}\nItem: {name or "—"}\n'
                       f'Status: {status}\n{conf_line}')
        elif name:
            tooltip = f'Slot: {slot_disp}\nItem: {name}\nConfidence: {conf:.1%}'
            if cross_check_failed:
                tooltip += '\n\n⚠ WARNING: Item type does not match slot type!'
        else:
            tooltip = f'Slot: {slot_disp}\nNo item recognised'
        item.setToolTip(0, slot_disp)
        item.setToolTip(2, tooltip)

        # Slot (col 0) and Idx (col 1) are structural grouping columns —
        # keep them in the chrome foreground (white) so they don't shift
        # between white and the high/medium/low-conf colours depending on
        # the row's confidence. Only the Item, Conf and Status columns
        # carry the state colour. `refresh_parent_of` then mirrors these
        # foregrounds onto the parent row, so the group header stays neutral
        # in cols 0-1 too. This makes the WARP CORE review tree match Fast
        # Correction Mode's existing convention.
        fg_state = QBrush(QColor(color))
        fg_white = QBrush(QColor(FG))
        item.setForeground(0, fg_white)
        item.setForeground(1, fg_white)
        for c in range(2, 5):
            item.setForeground(c, fg_state)
        # If this item is already living under a parent in the grouped
        # tree, mirror the new texts/foreground to the parent so the
        # collapsed-state summary line stays in sync.
        if item.parent() is not None:
            self._review_list.refresh_parent_of(item)

    def _add_review_row(self, name: str, slot: str, conf: float, confirmed: bool = False, cross_check_failed: bool = False, auto_confirmed: bool = False, conflict_disk_name: str = '', group_label: str | None = None):
        item = QTreeWidgetItem()
        # `_slot_ordinal` counts existing rows sharing the same col-0
        # UserRole. When seat-grouping is active the UserRole is the
        # group label (one parent row per physical seat), so we count
        # within the seat to keep Idx contiguous; without grouping we
        # fall back to counting by slot, matching legacy behaviour.
        idx  = self._slot_ordinal(group_label if group_label else slot)
        self._populate_review_item(
            item, name, slot, conf,
            confirmed=confirmed, cross_check_failed=cross_check_failed,
            auto_confirmed=auto_confirmed, conflict_disk_name=conflict_disk_name,
            idx=idx, group_label=group_label,
        )
        self._review_list.addItem(item)

    def _on_review_item_clicked(self, item: QListWidgetItem):
        if not self._selection_just_changed:
            self._review_list.setCurrentRow(-1)
            self._ann_widget.clear_highlight()
        self._selection_just_changed = False

    def _on_review_parent_selected(self, parent):
        """Group-header click — highlight every child bbox at once.

        Mirrors WARP Results' behaviour: selecting a slot/seat header in
        the tree paints all its rows on the canvas so the user can see
        the whole group's geometry at a glance. `parent=None` clears the
        group highlight; emitted by the adapter when selection moves
        back to a leaf or empties.
        """
        aw = getattr(self, '_ann_widget', None)
        if aw is None:
            return
        if parent is None:
            aw.set_highlighted_rows(())
            return
        rows = self._review_list.child_rows_of(parent)
        aw.set_highlighted_rows(rows)
        # `_on_review_item_clicked` toggles selection off when the
        # subsequent itemClicked sees no fresh selection change — the
        # leaf handler sets this flag for that purpose. Parent clicks
        # need the same protection, otherwise the highlight blinks on
        # and off in a single click.
        self._selection_just_changed = True

    # ── BOFF group context menu ──────────────────────────────────────

    _BOFF_BASE_PROFS = ('Tactical', 'Engineering', 'Science')
    _BOFF_SPEC_PROFS = ('Command', 'Intelligence', 'Miracle Worker', 'Pilot', 'Temporal')

    def _show_review_context_menu(self, pos):
        """Right-click on a BOFF group header → Change Group Type submenu."""
        item = self._review_list.itemAt(pos)
        if item is None:
            return
        # Only act on parent (group header) rows, not leaf items
        if item in self._review_list._flat:
            return
        group_label = item.data(0, Qt.ItemDataRole.UserRole) or ''
        if not group_label.startswith('Boff'):
            return
        if self._is_current_locked():
            return

        menu = QMenu(self)
        sub = menu.addMenu('Change Group Type')

        # Base-only entries
        for prof in self._BOFF_BASE_PROFS:
            label = f'Boff {prof}'
            act = sub.addAction(label)
            act.setData(label)
        sub.addSeparator()
        # Base+Spec entries
        for base in self._BOFF_BASE_PROFS:
            for spec in self._BOFF_SPEC_PROFS:
                label = f'Boff {base}+{spec}'
                act = sub.addAction(label)
                act.setData(label)

        chosen = menu.exec(self._review_list.viewport().mapToGlobal(pos))
        if chosen is None:
            return
        new_type = chosen.data()
        if new_type and new_type != group_label:
            self._change_boff_group_type(item, group_label, new_type)

    def _change_boff_group_type(self, parent_item, old_label: str, new_label: str):
        """Change all items in a BOFF group to a new group type and rematch."""
        from warp.recognition.icon_matcher import SETSIconMatcher
        from warp.debug import log as _sl

        rows = self._review_list.child_rows_of(parent_item)
        if not rows:
            return

        _sl.info(f'change_boff_group: {old_label!r} → {new_label!r} ({len(rows)} items)')

        # Determine the base slot for rematching: e.g. 'Boff Tactical+Command'
        # → items get slot 'Boff Tactical' (base) but we search candidates
        # from both base and spec professions.
        if '+' in new_label:
            base_part, _spec_part = new_label.split('+', 1)
            base_slot = base_part.strip()
        else:
            base_slot = new_label

        # Build combined candidate set for the new type
        candidates = set(self._build_search_candidates(base_slot))
        if '+' in new_label:
            spec_slot = f'Boff {new_label.split("+", 1)[1].strip()}'
            candidates |= set(self._build_search_candidates(spec_slot))

        # Update each item in the group
        for row in rows:
            if row >= len(self._recognition_items):
                continue
            ri = self._recognition_items[row]
            crop_bgr = ri.get('crop_bgr')

            # If crop is missing, try to load it from the screenshot
            if crop_bgr is None and self._current_idx >= 0:
                import cv2
                img = cv2.imread(str(self._screenshots[self._current_idx]))
                if img is not None:
                    bbox = ri.get('bbox')
                    if bbox:
                        x, y, w, h = bbox
                        crop_bgr = img[y:y+h, x:x+w].copy()
                        ri['crop_bgr'] = crop_bgr

            # Rematch against the new candidate set
            name, conf, thumb = '', 0.0, None
            if crop_bgr is not None and candidates:
                try:
                    _name, _conf, _thumb, _used_sess = SETSIconMatcher(
                        self._sets).match(crop_bgr, candidate_names=candidates)
                    if _conf >= 0.40:
                        name, conf, thumb = _name, _conf, _thumb
                except Exception as e:
                    _sl.warning(f'change_boff_group rematch failed row {row}: {e}')

            # Update the recognition item
            ri['name'] = name
            ri['conf'] = conf
            ri['thumb'] = thumb
            ri['slot'] = base_slot
            ri['state'] = 'pending'
            ri['auto_confirmed'] = False
            ri['cross_check_failed'] = False
            ri['_group_label'] = new_label

            _sl.debug(f'  row {row}: slot={base_slot!r} name={name!r} conf={conf:.2f}')

            # Update the tree item visuals
            litem = self._review_list.item(row)
            if litem:
                self._populate_review_item(
                    litem, name, base_slot, conf,
                    confirmed=False, cross_check_failed=False,
                    auto_confirmed=False, conflict_disk_name='',
                    group_label=new_label,
                )

        # Reparent all children to the new group
        for row in rows:
            litem = self._review_list.item(row)
            if litem:
                self._review_list.reparent_item(litem, new_label, new_label)

        self._resort_parents_canonical()
        self._ann_widget.set_review_items(self._recognition_items)
        self.statusBar().showMessage(
            f'Group changed: {old_label} → {new_label} ({len(rows)} items rematched)')

    def _on_review_row_changed(self, row: int):
        if row == -1:
            self._set_review_buttons_enabled(False)
            self._ann_widget.clear_highlight()
            return
        self._selection_just_changed = True
        self._loading_row = True
        try:
            if 0 <= row < len(self._recognition_items):
                ri = self._recognition_items[row]
                is_confirmed = ri.get('state') == 'confirmed'
                self._btn_remove_item.setEnabled(not self._is_current_locked())
                # self._btn_edit_bbox.setEnabled(True)  # disabled
                # if is_confirmed:
                #     self._btn_edit_bbox.setChecked(False)
                #     self._ann_widget.set_draw_mode(False)
                slot = ri['slot']
                # Map seat keys / 'Boff Universal' to a real profession label
                # before touching the combo (dropdown only has static slots).
                combo_slot = self._slot_for_combo(slot)
                # Ensure this slot is visible in combo (confirmed NON_ICON_SLOTS
                # are normally hidden, but must show when the item is selected)
                if self._current_idx >= 0:
                    _stype = self._screen_types.get(
                        self._screenshots[self._current_idx].name, 'UNKNOWN')
                    self._refresh_slot_combo(_stype, keep_slot=combo_slot)
                idx = self._slot_combo.findText(combo_slot)
                if idx >= 0:
                    self._slot_combo.setCurrentIndex(idx)
                # Populate completer for this slot without triggering clear on name_edit
                # NON_ICON_SLOTS use their own widgets — skip completer (avoids iterating all equipment)
                if slot not in NON_ICON_SLOTS:
                    self._populate_name_completer(slot)
                # Always configure name field explicitly — setCurrentIndex may not fire
                # currentIndexChanged if the numerical index didn't change (e.g. after
                # _refresh_slot_combo rebuilt the combo), leaving a stale label/state.
                self._configure_name_field(combo_slot)
                # Sync the right input widget for the row's slot. NON_ICON_SLOTS
                # (Ship Tier / Ship Type) use dedicated combos that are visible
                # instead of _name_edit — without this branch the combos kept
                # whatever was in them last (typically T1 / first ship type),
                # which made the Annotate panel disagree with the review row.
                row_name = ri.get('name', '') or ''
                if slot == 'Ship Tier':
                    idx_t = self._tier_combo.findText(row_name)
                    self._tier_combo.blockSignals(True)
                    if idx_t >= 0:
                        self._tier_combo.setCurrentIndex(idx_t)
                    else:
                        self._tier_combo.setCurrentIndex(-1)
                    self._tier_combo.blockSignals(False)
                elif slot == 'Ship Type':
                    if self._ship_type_combo.count() == 0:
                        self._populate_ship_type_combo()
                    self._ship_type_combo.blockSignals(True)
                    idx_s = self._ship_type_combo.findText(row_name)
                    if idx_s >= 0:
                        self._ship_type_combo.setCurrentIndex(idx_s)
                    else:
                        self._ship_type_combo.lineEdit().setText(row_name)
                    self._ship_type_combo.blockSignals(False)
                else:
                    # Icon slots + Ship Name: text input is _name_edit.
                    self._name_edit.blockSignals(True)
                    self._name_edit.setText(row_name)
                    self._name_edit.blockSignals(False)
                    if hasattr(self, '_completer'):
                        self._completer.setCompletionPrefix(row_name)
                if ri.get('bbox'):
                    self._ann_widget.set_highlighted_row(row)
                else:
                    self._ann_widget.clear_highlight()
                if is_confirmed:
                    self._review_list.setFocus()
        finally:
            self._loading_row = False

    def _init_sync_client(self):
        try:
            from warp.knowledge.sync_client import WARPSyncClient
            self._sync_client = WARPSyncClient()
        except Exception as e:
            log.warning(f'WARP CORE: sync client init failed: {e}')

    def _contribute(self, ri: dict, confirmed_name: str):
        try:
            if self._sync_client is None:
                return
            wrong = ri.get('orig_name', '')
            if wrong == confirmed_name:
                wrong = ''
            self._sync_client.contribute(crop_bgr=ri['crop_bgr'], item_name=confirmed_name, wrong_name=wrong, confirmed=True)
        except Exception as e:
            log.warning(f'WARP CORE: contribute failed: {e}')

    def _on_edit_bbox_toggle(self, checked: bool):
        pass  # Edit BBox disabled — reserved for future implementation

    def _setup_shortcuts(self):
        """Global keyboard shortcuts — work regardless of focus."""
        from PySide6.QtWidgets import QApplication
        QApplication.instance().installEventFilter(self)
        QShortcut(QKeySequence('Alt+A'), self,
                  activated=lambda: self._btn_add_bbox.click())
        QShortcut(QKeySequence('Alt+R'), self,
                  activated=self._on_remove_item)
        QShortcut(QKeySequence('Alt+D'), self,
                  activated=self._btn_done.click)
        QShortcut(QKeySequence('Return'), self,
                  activated=self._on_accept)
        QShortcut(QKeySequence('Delete'), self,
                  activated=self._on_remove_item)
        QShortcut(QKeySequence('Alt+Up'), self,
                  activated=self._nav_prev_screenshot)
        QShortcut(QKeySequence('Alt+Down'), self,
                  activated=self._nav_next_screenshot)
        # Restore auto-accept settings
        self._chk_auto_accept.setChecked(
            self._settings.value(_KEY_AUTO_ACCEPT, True, type=bool))
        self._spin_auto_conf.setValue(
            float(self._settings.value(_KEY_AUTO_CONF, 0.75)))
        # Save on change
        self._chk_auto_accept.toggled.connect(
            lambda v: self._settings.setValue(_KEY_AUTO_ACCEPT, v))
        self._spin_auto_conf.valueChanged.connect(
            lambda v: self._settings.setValue(_KEY_AUTO_CONF, v))

    def _nav_prev_screenshot(self):
        row = self._file_list.currentRow()
        if row > 0:
            self._file_list.setCurrentRow(row - 1)

    def _nav_next_screenshot(self):
        row = self._file_list.currentRow()
        if row < self._file_list.count() - 1:
            self._file_list.setCurrentRow(row + 1)

    def eventFilter(self, obj, event):
        """Handle Delete key on review list/canvas, and forward Ctrl+wheel from scroll area to canvas."""
        from PySide6.QtCore import QEvent
        rl = getattr(self, '_review_list', None)
        aw = getattr(self, '_ann_widget', None)
        sa = getattr(self, '_scroll_area', None)
        if obj in (rl, aw) and obj is not None and event.type() == QEvent.Type.KeyPress:
            key = event.key()
            if key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
                if self._is_current_locked():
                    return True
                self._on_remove_item()
                return True
            # Enter / Return on the bbox list = same as Enter in the name
            # field: accept the current row and advance to the next
            # unconfirmed one. Without this, focus-on-list Enter was a
            # no-op, leaving users stranded on the current row depending
            # on which widget had focus — inconsistent and surprising.
            if obj is rl and key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self._on_accept()
                return True
        # Forward wheel events from anywhere in scroll area to the canvas widget
        # (only when WARP CORE is the active window)
        if event.type() == QEvent.Type.Wheel and sa and aw and self.isActiveWindow():
            from PySide6.QtGui import QCursor
            gpos = QCursor.pos()
            sa_pos = sa.mapFromGlobal(gpos)
            aw_pos = aw.mapFromGlobal(gpos)
            if sa.rect().contains(sa_pos) and not aw.rect().contains(aw_pos):
                aw.wheelEvent(event)
                return True
        return super().eventFilter(obj, event)

    def _on_add_bbox_toggle(self, checked: bool):
        if checked:
            self._btn_edit_bbox.setChecked(False)
            self._manual_bbox_mode = False
            self._add_bbox_mode = True
            self._manual_mode_lbl.setText('Draw a rectangle to add a new item.')
            self._manual_mode_lbl.setVisible(True)
            self._ann_widget.set_draw_mode(True)
        else:
            self._add_bbox_mode = False
            self._manual_mode_lbl.setVisible(False)
            self._ann_widget.set_draw_mode(False)

    def _on_remove_item(self):
        if self._is_current_locked():
            return
        row = self._review_list.currentRow()
        if row < 0 or row >= len(self._recognition_items):
            return
        ri = self._recognition_items[row]
        if ri.get('state') == 'confirmed':
            name = ri.get('name') or ri.get('slot') or 'this item'
            reply = QMessageBox.question(
                self, 'Remove confirmed annotation',
                f'Remove confirmed bbox for "{name}"?\n\n'
                f'This will delete the saved annotation for this slot.',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        # Remove matching disk annotation — by exact bbox first, then
        # IoU fallback for near-overlapping bboxes (user-drawn vs
        # detector grid). Runs for BOTH confirmed and pending items:
        # in Fast Correction mode items are seeded as 'pending' even
        # when the same bbox has a confirmed annotation on disk; without
        # this the disk entry survives and Send to WARP still emits it.
        if self._current_idx >= 0 and ri.get('bbox'):
            path = self._screenshots[self._current_idx]
            ri_bbox = tuple(ri['bbox'])
            from warp.trainer.training_data import _bbox_iou
            best_ann, best_iou = None, 0.0
            for ann in self._data_mgr.get_annotations(path):
                if tuple(ann.bbox) == ri_bbox:
                    best_ann = ann
                    break
                iou = _bbox_iou(ri_bbox, tuple(ann.bbox))
                if iou > best_iou:
                    best_iou, best_ann = iou, ann
            if best_ann is not None and (tuple(best_ann.bbox) == ri_bbox
                                          or best_iou >= 0.5):
                self._data_mgr.remove_annotation(path, best_ann)
                self._data_mgr.save()
        self._review_list.takeItem(row)
        self._recognition_items.pop(row)
        self._exit_manual_bbox_mode()
        if self._current_idx >= 0:
            fname = self._screenshots[self._current_idx].name
            self._recognition_cache[fname] = list(self._recognition_items)
        n = len(self._recognition_items)
        if n == 0:
            self._set_review_buttons_enabled(False)
            self._ann_widget.clear_highlight()
        else:
            new_row = min(row, n - 1)
            self._review_list.setCurrentRow(new_row)
            self._on_review_row_changed(new_row)
        self._ann_widget.set_review_items(self._recognition_items)
        self._update_progress()
        # Restore removed NON_ICON_SLOT back to combo if it was confirmed
        if self._current_idx >= 0:
            _stype = self._screen_types.get(self._screenshots[self._current_idx].name, 'UNKNOWN')
            self._refresh_slot_combo(_stype)

    def _on_clear_all_bboxes(self):
        """Remove every bbox on the current screenshot.

        Three-way dialog: clear all, clear only pending (spare confirmed),
        or cancel. Confirmed bboxes that get cleared are also wiped from
        the on-disk annotations file via `_data_mgr.remove_annotation`,
        same as the single-item Remove flow."""
        if self._current_idx < 0 or not self._recognition_items:
            return
        if self._is_current_locked():
            return
        path = self._screenshots[self._current_idx]
        total = len(self._recognition_items)
        confirmed_count = sum(
            1 for ri in self._recognition_items if ri.get('state') == 'confirmed'
        )

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle('Clear all bboxes')
        if confirmed_count:
            msg = (
                f'Remove all {total} bbox(es) from "{_disp_name(path.name)}"?\n\n'
                f'{confirmed_count} are confirmed — they will also be '
                f'deleted from the saved annotations on disk.'
            )
        else:
            msg = (
                f'Remove all {total} bbox(es) from "{_disp_name(path.name)}"?\n\n'
                f'None of them are confirmed yet.'
            )
        box.setText(msg)
        btn_yes = box.addButton('Yes', QMessageBox.ButtonRole.DestructiveRole)
        btn_spare = None
        if confirmed_count and confirmed_count < total:
            btn_spare = box.addButton('Spare All Confirmed',
                                      QMessageBox.ButtonRole.ActionRole)
        btn_no = box.addButton('Cancel', QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(btn_no)
        box.exec()
        clicked = box.clickedButton()
        if clicked is btn_no or clicked is None:
            return

        spare_confirmed = (clicked is btn_spare)
        removed = 0
        kept: list[dict] = []
        for ri in self._recognition_items:
            if spare_confirmed and ri.get('state') == 'confirmed':
                kept.append(ri)
                continue
            if ri.get('state') == 'confirmed' and ri.get('bbox'):
                for ann in self._data_mgr.get_annotations(path):
                    if ann.bbox == ri['bbox']:
                        self._data_mgr.remove_annotation(path, ann)
                        break
            removed += 1
        if removed:
            self._data_mgr.save()
        self._recognition_items = kept

        log.info(
            f'clear_all_bboxes: image={path.name} removed={removed} '
            f'kept={len(kept)} spare_confirmed={spare_confirmed}'
        )

        self._review_list.clear()
        for ri in self._recognition_items:
            self._add_review_row(
                ri.get('name', ''), ri.get('slot', ''),
                ri.get('conf', 0.0),
                confirmed=(ri.get('state') == 'confirmed'),
                cross_check_failed=ri.get('cross_check_failed', False),
                auto_confirmed=ri.get('auto_confirmed', False),
                group_label=ri.get('_group_label'),
            )
        self._exit_manual_bbox_mode()
        self._recognition_cache[path.name] = list(self._recognition_items)
        self._ann_widget.refresh_annotations(path)
        self._ann_widget.set_review_items(self._recognition_items)
        self._ann_widget.clear_highlight()
        self._ann_widget.update()
        self._set_review_buttons_enabled(False)
        self._update_progress()
        _stype = self._screen_types.get(path.name, 'UNKNOWN')
        self._refresh_slot_combo(_stype)
        self.statusBar().showMessage(
            f'Cleared {removed} bbox(es) from {_disp_name(path.name)}'
            + (f' — kept {len(kept)} confirmed.' if spare_confirmed and kept else '.')
        )

    def _enter_manual_bbox_mode(self):
        pass  # Resize/move disabled — reserved for future implementation
        # self._manual_bbox_mode = True
        # self._btn_edit_bbox.setChecked(True)
        row = self._review_list.currentRow()
        if 0 <= row < len(self._recognition_items):
            ri = self._recognition_items[row]
            slot = ri['slot']
            if ri.get('state') == 'confirmed':
                ri['state'] = 'pending'
            self._ann_widget.set_review_items(self._recognition_items)
            self._ann_widget.set_selected_row(row)
        else:
            slot = '?'
        self._manual_mode_lbl.setText(f'Draw a rectangle to redefine region for:\n{_pretty_slot(slot)}')
        self._manual_mode_lbl.setVisible(True)
        self._ann_widget.set_draw_mode(True)

    def _exit_manual_bbox_mode(self):
        self._manual_bbox_mode = False
        self._btn_edit_bbox.setChecked(False)
        self._manual_mode_lbl.setVisible(False)
        self._ann_widget.set_draw_mode(False)
        self._ann_widget.set_selected_row(-1)

    def _resort_group_of(self, item):
        """Re-sort the tree-group containing `item` by spatial bbox order
        (row-bucketed y, then x) so a row that was just added or
        reparented lands in L→R / T→B position within its group. Then
        resync `_flat` and `_recognition_items` to match the new visual
        order so Enter-advance and every other row-indexed code path
        walk the list in the order the user actually sees.

        Raw (y, x) sort is too brittle: same-row bboxes that differ by
        1–2 px in y (detector jitter or manual-draw rounding) would
        sort by y instead of x and produce e.g. T-Lock@y=377 before
        Kobayashi@y=379. Bucketing y by `ROW_TOL` collapses jitter so
        items in the same physical row sort purely by x.
        """
        parent = item.parent() if item is not None else None
        if parent is None:
            return
        rl = self._review_list

        def _bbox_of(ch):
            row = rl.row(ch)
            if 0 <= row < len(self._recognition_items):
                bbox = self._recognition_items[row].get('bbox')
                if bbox and len(bbox) >= 2:
                    return bbox
            return None

        # Cluster y values into row buckets — within ROW_TOL px counts
        # as the same row. Mirrors group_items_by_seat's parent-level
        # bucketing so within-group ordering matches.
        ROW_TOL = 30
        ys = sorted({_bbox_of(parent.child(i))[1]
                     for i in range(parent.childCount())
                     if _bbox_of(parent.child(i)) is not None})
        row_of_y: dict[int, int] = {}
        cur_row = 0
        prev_y: int | None = None
        for y in ys:
            if prev_y is not None and y - prev_y > ROW_TOL:
                cur_row += 1
            row_of_y[y] = cur_row
            prev_y = y

        def _key(ch):
            bbox = _bbox_of(ch)
            if bbox is None:
                return (1_000_000, 1_000_000)
            return (row_of_y.get(bbox[1], 0), bbox[0])

        rl.resort_group(parent, _key)
        self._resync_recognition_with_visual()

    def _resort_parents_canonical(self):
        """Reorder top-level parents in the tree to match the canonical
        slot order (same logic `_populate_review_panel` applies on cold
        load). Without this a freshly created group (manual bbox draw
        for a slot that wasn't on the list yet, or a slot change that
        opens a new parent) ends up appended to the bottom — adapter's
        `_get_or_create_parent` always tacks new parents on at the end.
        """
        if self._current_idx < 0:
            return
        path = self._screenshots[self._current_idx]
        stype = self._screen_types.get(path.name, 'UNKNOWN')
        from warp.recognition.boff_keys import order_items_for_display
        from warp.warp_importer import DISPLAY_CANONICAL_ORDER
        build_type = self._STYPE_TO_BUILD.get(stype, 'SPACE')
        # Match `_order_items_for_review`: a TRAITS screen uses the
        # mixed ground+space canonical for display ordering.
        canonical_bt = 'TRAITS' if stype == 'TRAITS' else build_type
        canonical = [sd['name'] for sd in _SLOT_ORDER.get(canonical_bt, [])]
        ordered = order_items_for_display(
            self._recognition_items, canonical,
            fallback_canonical_slots=DISPLAY_CANONICAL_ORDER,
        )
        slot_order = [label for label, _ in ordered]
        self._review_list.reorder_parents(slot_order)
        self._resync_recognition_with_visual()

    def _resync_recognition_with_visual(self):
        """Permute `_flat` (in the adapter) and `_recognition_items` (here)
        to match the tree's current visual top-down order. Cheap (one
        walk) and keeps every row-indexed caller — Enter-advance,
        `_review_row_changed`, ann_widget highlight, recognition cache —
        aligned with what the user sees."""
        order = self._review_list.visual_row_order()
        if len(order) != len(self._recognition_items):
            return
        if order == list(range(len(order))):
            return
        self._recognition_items = [self._recognition_items[i] for i in order]
        self._review_list.apply_row_order(order)
        # ann_widget paints bboxes keyed off `_recognition_items` row
        # order (selected_row, highlighted_row), so it needs the fresh
        # list too — otherwise canvas highlights point at the wrong bbox
        # after the resort. Also refresh the highlight from the current
        # row, because `currentItemChanged` does NOT fire on a pure
        # _flat permutation (the QTreeWidgetItem object hasn't changed),
        # leaving ann_widget's cached row index stale.
        self._ann_widget.set_review_items(self._recognition_items)
        cur_row = self._review_list.currentRow()
        if 0 <= cur_row < len(self._recognition_items) \
                and self._recognition_items[cur_row].get('bbox'):
            self._ann_widget.set_highlighted_row(cur_row)
        if self._current_idx >= 0:
            fname = self._screenshots[self._current_idx].name
            self._recognition_cache[fname] = list(self._recognition_items)

    def _advance_to_next_unconfirmed(self, current_row: int):
        # Auto-confirmed rows (yellow) are program decisions awaiting human
        # review — treat them as still needing advance, otherwise Enter
        # strands the user on screens where every row was auto-accepted
        # (typical for BOFF, where most abilities clear the conf threshold).
        for i in range(current_row + 1, len(self._recognition_items)):
            ri = self._recognition_items[i]
            if (ri['state'] in ('pending', 'community_conflict')
                    or ri.get('auto_confirmed')):
                self._review_list.setCurrentRow(i)
                return

    def _set_review_buttons_enabled(self, enabled: bool):
        for btn in (self._btn_remove_item,):  # btn_edit_bbox disabled
            btn.setEnabled(enabled)

    def _is_current_locked(self) -> bool:
        return (self._current_idx >= 0
                and self._screenshots[self._current_idx].name in self._screenshots_done)

    def _update_add_bbox_btn(self):
        is_done = self._is_current_locked()
        is_spec = (self._current_idx >= 0
                   and self._screen_types.get(
                       self._screenshots[self._current_idx].name, 'UNKNOWN') == 'SPECIALIZATIONS')
        enabled = self._current_idx >= 0 and not is_done and not is_spec
        self._btn_add_bbox.setEnabled(enabled)
        if is_spec:
            self._btn_add_bbox.setToolTip(
                'Specialization screens are used only for screen-type training.\n'
                'Icon annotation is not supported for this screen type.')
        else:
            self._btn_add_bbox.setToolTip('')
        # Destructive bbox actions are also blocked while the screenshot is Done.
        if is_done:
            locked_tip = 'Screenshot is marked Done — press ↩ Back to Edit to modify.'
            self._btn_remove_item.setEnabled(False)
            self._btn_remove_item.setToolTip(locked_tip)
            self._btn_clear_all_bboxes.setEnabled(False)
            self._btn_clear_all_bboxes.setToolTip(locked_tip)
            self._action_auto_detect.setEnabled(False)
            self._action_auto_detect.setToolTip(locked_tip)
        else:
            self._btn_remove_item.setToolTip('')
            self._btn_clear_all_bboxes.setEnabled(True)
            self._btn_clear_all_bboxes.setToolTip(
                'Remove every bbox on the current screenshot. A confirmation dialog '
                'offers the option to spare bboxes already marked confirmed.'
            )
            self._action_auto_detect.setEnabled(True)
            self._action_auto_detect.setToolTip('Auto-detect icons')
        # Send to WARP is the inverse: only enabled once the screenshot is
        # locked, since "Done" means the user has reviewed every bbox and
        # the result is safe to hand back to WARP for JSON export.
        # In Fast Correction Mode the gate is stricter: every file in the
        # ephemeral batch must be Done before the user can send the
        # corrected batch back to WARP.
        if self._mode == 'fast_correction':
            all_done = bool(self._screenshots) and all(
                p.name in self._screenshots_done for p in self._screenshots)
            self._btn_send_to_warp.setEnabled(all_done)
            self._btn_send_to_warp.setToolTip(
                'Send all corrected screenshots back to WARP.'
                if all_done else
                'Mark every screenshot Done first — then send the batch back to WARP.')
        elif is_done:
            self._btn_send_to_warp.setEnabled(True)
            self._btn_send_to_warp.setToolTip(
                'Send the confirmed results to WARP and switch tabs — '
                'lets you export the JSON build without re-running detection.')
        else:
            self._btn_send_to_warp.setEnabled(False)
            self._btn_send_to_warp.setToolTip(
                'Mark this screenshot as Done first.')

    def _on_bbox_drawn(self, bbox: tuple):
        if self._current_idx >= 0 and self._screen_types.get(
                self._screenshots[self._current_idx].name, 'UNKNOWN') == 'SPECIALIZATIONS':
            self._add_bbox_mode = False
            self._btn_add_bbox.setChecked(False)
            self._ann_widget.set_draw_mode(False)
            return
        if self._manual_bbox_mode:
            row = self._review_list.currentRow()
            if 0 <= row < len(self._recognition_items):
                self._recognition_items[row]['bbox'] = bbox
            self._rematch_current_item(row, bbox)
        elif getattr(self, '_add_bbox_mode', False) \
                or getattr(self._ann_widget, '_alt_draw', False):
            self._add_bbox_mode = False
            self._btn_add_bbox.setChecked(False)
            self._manual_mode_lbl.setVisible(False)
            self._ann_widget.set_draw_mode(False)
            self._ann_widget.set_review_items(self._recognition_items)
            name, conf, thumb, crop_bgr = '', 0.0, None, None
            if self._current_idx >= 0:
                try:
                    import cv2
                    path = self._screenshots[self._current_idx]
                    img = cv2.imread(str(path))
                    if img is not None:
                        x, y, w, h = bbox
                        crop_bgr = img[y:y+h, x:x+w].copy()
                        from warp.debug import log as _slog
                        _slog.info(f'add_bbox: crop {x},{y},{w},{h} px from {path.name}')
                        # ── P1: suggest slot from bbox position ──────────────
                        suggested = self._suggest_slot_from_position(bbox)
                        if suggested:
                            self._slot_combo.blockSignals(True)
                            self._slot_combo.setCurrentText(suggested)
                            self._slot_combo.blockSignals(False)
                            _slog.info(f'add_bbox: P1 slot suggestion → {suggested!r}')
                        _current_slot = self._slot_combo.currentText()
                        # If slot_suggest gave no suggestion and the combo still shows a
                        # confirmed single-instance slot, skip matching entirely — the
                        # icon is in a different panel (e.g. traits to the right of the
                        # equipment column). Show unmatched; user picks slot manually.
                        if (not suggested
                                and _current_slot in SINGLE_INSTANCE_SLOTS
                                and self._current_idx >= 0):
                            _already = {
                                ann.slot for ann in self._data_mgr.get_annotations(path)
                                if ann.state == AnnotationState.CONFIRMED
                            }
                            if _current_slot in _already:
                                _slog.info(
                                    f'add_bbox: no suggestion, {_current_slot!r} already '
                                    f'confirmed — skipping match (different panel)')
                                self._finish_bbox_drawn('', 0.0, None, crop_bgr, bbox)
                                return
                        # If current slot is a NON_ICON_SLOT already confirmed for this image,
                        # advance to the next unconfirmed NON_ICON_SLOT to prevent
                        # SINGLE_INSTANCE step from silently deleting the earlier annotation.
                        if _current_slot in NON_ICON_SLOTS and self._current_idx >= 0:
                            _confirmed_slots = {
                                ann.slot for ann in self._data_mgr.get_annotations(path)
                                if ann.state == AnnotationState.CONFIRMED
                            }
                            if _current_slot in _confirmed_slots:
                                for _next in ('Ship Name', 'Ship Type', 'Ship Tier'):
                                    if _next not in _confirmed_slots:
                                        _current_slot = _next
                                        self._slot_combo.blockSignals(True)
                                        self._slot_combo.setCurrentText(_next)
                                        self._slot_combo.blockSignals(False)
                                        log.info(f'add_bbox: {_current_slot!r} already confirmed '
                                                 f'→ advanced slot to {_next!r}')
                                        break
                        if _current_slot not in NON_ICON_SLOTS:
                            _candidates = set(self._build_search_candidates(_current_slot))
                            # Optional slots (Sec-Def, Experimental, Hangars) may not exist on
                            # the current ship. Expand candidates with the next mandatory slot so
                            # the ML can determine which one this icon actually is.
                            # _infer_slot_from_name will assign the correct slot after matching.
                            _OPTIONAL_SLOTS = ('Sec-Def', 'Experimental', 'Hangars')
                            if _current_slot in _OPTIONAL_SLOTS:
                                _stype_key = self._screen_types.get(path.name, 'UNKNOWN')
                                _grp = SCREEN_TO_SLOT_GROUP.get(_stype_key, 'ALL')
                                _ord = SLOT_GROUPS.get(_grp, ALL_SLOTS)
                                if _current_slot in _ord:
                                    for _ns in _ord[_ord.index(_current_slot) + 1:]:
                                        if _ns not in NON_ICON_SLOTS and _ns not in _OPTIONAL_SLOTS:
                                            _candidates |= set(self._build_search_candidates(_ns))
                                            _slog.info(
                                                f'add_bbox: {_current_slot!r} is optional — '
                                                f'expanding candidates with {_ns!r}')
                                            break
                            self._start_match_worker(crop_bgr, bbox, _candidates or None)
                            return
                        # NON_ICON_SLOT: icon matching skipped, fall through to _finish_bbox_drawn
                except Exception as _e:
                    from warp.debug import log as _slog
                    _slog.warning(f'add_bbox: error: {_e}')
            self._finish_bbox_drawn('', 0.0, None, crop_bgr, bbox)
        else:
            self._name_edit.setFocus()
            self._name_edit.clear()

    def _start_match_worker(self, crop_bgr, bbox: tuple, candidate_names) -> None:
        """Start async icon matching; show spinner after 500ms if still running."""
        if not hasattr(self, '_match_workers'):
            self._match_workers = []
        worker = MatchWorker(crop_bgr, bbox, candidate_names, self._sets, parent=self)
        worker.finished.connect(self._on_match_worker_done)
        self._match_workers.append(worker)
        QTimer.singleShot(500, lambda: self._match_progress.setVisible(
            any(w.isRunning() for w in self._match_workers)))
        worker.start()

    def _on_match_worker_done(self, name: str, conf: float, thumb, crop_bgr, bbox: tuple) -> None:
        self._match_progress.setVisible(False)
        self._finish_bbox_drawn(name, conf, thumb, crop_bgr, bbox)

    def _finish_bbox_drawn(self, name: str, conf: float, thumb, crop_bgr, bbox: tuple) -> None:
        """Finalise a drawn bbox: infer slot, add to review list, trigger OCR if needed."""
        slot = self._slot_combo.currentText()
        # If matcher found a name, infer the correct slot from cache item type
        # Restrict to slots allowed by the current screen type
        if name:
            stype = 'UNKNOWN'
            if self._current_idx >= 0:
                stype = self._screen_types.get(
                    self._screenshots[self._current_idx].name, 'UNKNOWN')
            group_key = SCREEN_TO_SLOT_GROUP.get(stype, 'ALL')
            allowed = SLOT_GROUPS.get(group_key)  # None means no restriction
            inferred = self._infer_slot_from_name(name, allowed_slots=allowed)
            if inferred:
                # Weapon slots (Fore/Aft/Experimental) are positionally interchangeable
                # for generic 'Ship Weapon' type items — the cache type only says "weapon",
                # not fore vs aft.  P1 position suggestion is more accurate.
                _weapon_slots = frozenset({'Fore Weapons', 'Aft Weapons', 'Experimental'})
                # Universal Console items are valid in any console slot.
                # If P1 position already identified a specific console slot, trust it.
                _specific_console_slots = frozenset({
                    'Engineering Consoles', 'Science Consoles', 'Tactical Consoles'})
                if inferred in _weapon_slots and slot in _weapon_slots:
                    pass  # keep P1 position suggestion — fore vs aft determined by position
                elif inferred == 'Universal Consoles' and slot in _specific_console_slots:
                    pass  # keep P1 position suggestion
                elif inferred in SINGLE_INSTANCE_SLOTS and self._current_idx >= 0:
                    # Skip suggestion if this single-instance slot is already confirmed
                    _already = {
                        ann.slot for ann in self._data_mgr.get_annotations(
                            self._screenshots[self._current_idx])
                        if ann.state == AnnotationState.CONFIRMED
                    }
                    if inferred not in _already:
                        slot = inferred
                    # else: keep P1 position suggestion — slot is taken
                else:
                    slot = inferred
            elif name not in VIRTUAL_ITEM_NAMES:
                # Name found by matcher but doesn't belong to any allowed slot
                # for this screen type — discard to avoid wrong slot assignment.
                # Virtual names (__inactive__, __empty__) are always valid training labels,
                # so they keep the positional slot suggestion and are never discarded here.
                from warp.debug import log as _slog2
                _slog2.info(f'add_bbox: discarding {name!r} — not valid for stype={stype}')
                name, conf, thumb = '', 0.0, None
        # NON_ICON_SLOTS (Ship Name/Type/Tier) — text/position only, never icon.
        # If the matcher ran (because slot inference moved us into NON_ICON after
        # match) or P1 misrouted us, suppress any icon-name leak — OCR is the
        # only valid content source for these slots.
        if slot in NON_ICON_SLOTS:
            name, conf, thumb = '', 0.0, None
        if slot == 'Ship Name':
            _auto = True
        elif slot in NON_ICON_SLOTS:
            _auto = False
        else:
            # Auto-accept before adding to list if conf >= threshold
            _auto = (name and conf > 0
                     and getattr(self, '_chk_auto_accept', None)
                     and self._chk_auto_accept.isChecked()
                     and conf >= self._spin_auto_conf.value())

        _cross_check = False
        try:
            if name:
                from warp.warp_importer import WarpImporter
                _cross_check = not WarpImporter(sets_app=self._sets)._item_valid_for_slot(name, slot)
        except: pass

        _state = 'confirmed' if _auto else 'pending'
        # _auto means "program decided based on conf threshold" → yellow (auto-confirmed),
        # to be distinguished from green user-confirmed in the canvas.
        _auto_conf_flag = bool(_auto and slot != 'Ship Name')
        new_item = {'name': name, 'slot': slot, 'conf': conf, 'bbox': bbox, 'state': _state,
                    'thumb': thumb, 'crop_bgr': crop_bgr, 'orig_name': name, 'ship_name': '',
                    'cross_check_failed': _cross_check, 'auto_confirmed': _auto_conf_flag}
        self._recognition_items.append(new_item)
        if _auto and self._current_idx >= 0:
            _path = self._screenshots[self._current_idx]
            _saved = self._data_mgr.add_annotation(
                image_path=_path, bbox=bbox, slot=slot, name=name,
                state=AnnotationState.CONFIRMED, ml_conf=conf, ml_name=name,
                auto_confirmed=_auto_conf_flag,
                seat_key=new_item.get('seat_key', '') or '',
                slot_index=new_item.get('slot_index')
                    if isinstance(new_item.get('slot_index'), int) else -1)
            new_item['ann_id'] = _saved.ann_id
            if crop_bgr is not None:
                from warp.recognition.icon_matcher import SETSIconMatcher
                SETSIconMatcher.add_session_example(crop_bgr, name)
            self._data_mgr.save()
        self._add_review_row(name, slot, conf, confirmed=_auto, cross_check_failed=_cross_check,
                             auto_confirmed=_auto_conf_flag)
        # Spatial sort within the group so the freshly-added bbox slides
        # into L→R / T→B position instead of always landing at the end.
        # The sort also permutes `_recognition_items` so we must re-fetch
        # the row index from the QTreeWidgetItem before selecting.
        # NON_ICON slots are single-instance per screen — no spatial sort
        # is possible (or needed) within a one-child group, and skipping
        # the permutation keeps `new_row` stable for the pending OCRWorker
        # started further down.
        _new_item = self._review_list.item(len(self._recognition_items) - 1)
        if _new_item is not None and slot not in NON_ICON_SLOTS:
            self._resort_group_of(_new_item)
            # If the slot didn't have a group yet, its parent landed at
            # the bottom of the tree — re-position parents to canonical
            # SLOT_ORDER so e.g. a freshly-added Impulse Engine slots
            # between Deflector and Shield instead of after Traits.
            self._resort_parents_canonical()
            new_row = self._review_list.row(_new_item)
        else:
            new_row = len(self._recognition_items) - 1
        self._review_list.setCurrentRow(new_row)
        self._set_review_buttons_enabled(True)
        if self._current_idx >= 0:
            fname = self._screenshots[self._current_idx].name
            self._recognition_cache[fname] = list(self._recognition_items)
        self._ann_widget.clear_pending()

        if slot in NON_ICON_SLOTS and crop_bgr is not None and slot != 'Ship Name':
            if self._ship_type_combo.count() == 0:
                self._populate_ship_type_combo()
            v_tiers = [self._tier_combo.itemText(i) for i in range(self._tier_combo.count())]
            v_types = [self._ship_type_combo.itemText(i) for i in range(self._ship_type_combo.count())]
            worker = OCRWorker(new_row, crop_bgr, slot, v_tiers, v_types, parent=self)
            worker.finished.connect(self._on_ocr_finished)
            worker.start()
            if not hasattr(self, '_ocr_workers'): self._ocr_workers = []
            self._ocr_workers.append(worker)
        # Update slot combo to match inferred slot (suppressing textEdited on name field)
        if slot != self._slot_combo.currentText():
            self._slot_combo.blockSignals(True)
            self._slot_combo.setCurrentText(slot)
            self._slot_combo.blockSignals(False)
            self._populate_name_completer(slot)
        # Fill recognised name but do NOT open the dropdown automatically.
        # User can click the field to browse all slot-compatible items.
        self._name_edit.blockSignals(True)
        self._name_edit.setText(name)
        self._name_edit.blockSignals(False)
        if not _auto:
            self._ann_widget.setFocus()
        else:
            self._review_list.setFocus()

    def _on_ocr_finished(self, row: int, text: str, conf: float, crop_bgr, ocr_raw: str = ''):
        if row < 0 or row >= len(self._recognition_items): return
        ri = self._recognition_items[row]
        slot = ri['slot']

        # Ship Name is position-only — discard OCR text, never store content
        if slot == 'Ship Name':
            text = ''
        ri['name'] = text
        ri['conf'] = conf
        ri['crop_bgr'] = crop_bgr
        ri['ocr_raw'] = ocr_raw
        
        cross_check = False
        try:
            if text:
                from warp.warp_importer import WarpImporter
                cross_check = not WarpImporter(sets_app=self._sets)._item_valid_for_slot(text, slot)
        except Exception:
            pass
        ri['cross_check_failed'] = cross_check
        
        litem = self._review_list.item(row)
        if litem:
            self._populate_review_item(
                litem, text, slot, conf,
                confirmed=False, cross_check_failed=cross_check,
                auto_confirmed=False, conflict_disk_name='',
                group_label=ri.get('_group_label'),
            )

        if self._review_list.currentRow() == row:
            self._on_item_selected(ri)

        if (text and conf >= 0.4
            and getattr(self, '_chk_auto_accept', None)
            and self._chk_auto_accept.isChecked()
            and conf >= self._spin_auto_conf.value()):
            if self._review_list.currentRow() != row:
                self._review_list.setCurrentRow(row)
            self._on_accept()


    def _rematch_current_item(self, row: int, bbox: tuple):  # noqa — kept for future use
        if row < 0 or self._current_idx < 0:
            return
        if 0 <= row < len(self._recognition_items):
            pass  # Removed early return for NON_ICON_SLOTS to allow OCR on edit
        try:
            import cv2
            from warp.recognition.icon_matcher import SETSIconMatcher
            path = self._screenshots[self._current_idx]
            img = cv2.imread(str(path))
            if img is None:
                return
            x, y, w, h = bbox
            crop = img[y:y+h, x:x+w]
            if crop.size == 0:
                return
            ri = self._recognition_items[row]
            slot = ri['slot']
            # Constrain matcher to names valid for this slot (see _on_bbox_changed
            # for rationale — prevents cross-domain/career embedder leakage).
            cand = set(self._build_search_candidates(slot))
            _matcher = SETSIconMatcher(self._sets)
            name, conf, thumb, _used_sess = _matcher.match(
                crop, candidate_names=cand if cand else None)
            from warp.debug import log as _slog
            st = dict(getattr(_matcher, '_last_stage_scores', {}) or {})
            src = getattr(_matcher, '_last_match_src', '')
            ml = st.get('embed', 0) or st.get('soft', 0)
            _slog.info(
                f"WARP CORE: rematch row={row} slot='{slot}' "
                f"bbox={bbox} cand={len(cand)} src={src} "
                f"stages[embed={ml:.2f} sess={st.get('session',0):.2f} "
                f"tmpl={st.get('template',0):.2f} knldg={st.get('knowledge',0):.2f}] "
                f"→ ('{name}',{conf:.2f}) state={ri.get('state','')}"
            )

            _cross_check = False
            try:
                if name:
                    from warp.warp_importer import WarpImporter
                    _cross_check = not WarpImporter(sets_app=self._sets)._item_valid_for_slot(name, slot)
            except: pass

            if slot in NON_ICON_SLOTS:
                if self._ship_type_combo.count() == 0: self._populate_ship_type_combo()
                v_tiers = [self._tier_combo.itemText(i) for i in range(self._tier_combo.count())]
                v_types = [self._ship_type_combo.itemText(i) for i in range(self._ship_type_combo.count())]
                worker = OCRWorker(row, crop, slot, v_tiers, v_types, parent=self)
                worker.finished.connect(self._on_ocr_finished)
                worker.start()
                if not hasattr(self, '_ocr_workers'): self._ocr_workers = []
                self._ocr_workers.append(worker)
                return

            ri.update({'name': name, 'conf': conf, 'thumb': thumb, 'crop_bgr': crop,
                       'cross_check_failed': _cross_check, 'src': src})
            self._name_edit.setText(name)
            litem = self._review_list.item(row)
            if litem:
                self._populate_review_item(
                    litem, name, ri['slot'], conf,
                    confirmed=False, cross_check_failed=_cross_check,
                    auto_confirmed=False, conflict_disk_name='',
                    group_label=ri.get('_group_label'),
                )
            # Auto-accept if conf >= threshold and checkbox enabled
            if (name and conf > 0
                    and getattr(self, '_chk_auto_accept', None)
                    and self._chk_auto_accept.isChecked()
                    and conf >= self._spin_auto_conf.value()
                    and ri.get('slot', '') not in NON_ICON_SLOTS):
                self._review_list.setCurrentRow(row)
                self._on_accept()
        except:
            pass

    def _suggest_slot_from_position(self, bbox: tuple) -> str:
        """
        P1 — Infer the most likely slot for a newly-drawn bbox based on its
        Y-position, comparing against:
          1. Already confirmed/pending annotations on this screenshot
          2. Learned layouts from anchors.json

        Returns the slot name or '' if no confident match.
        """
        bx, by, bw, bh = bbox
        cy = by + bh // 2  # center Y of the drawn bbox

        stype = 'UNKNOWN'
        if self._current_idx >= 0:
            stype = self._screen_types.get(
                self._screenshots[self._current_idx].name, 'UNKNOWN')
        group_key = SCREEN_TO_SLOT_GROUP.get(stype, 'ALL')
        allowed = set(SLOT_GROUPS.get(group_key, ALL_SLOTS))

        # ── Source 1: existing annotations on this screenshot ────────────────
        # Build a map: slot → list of (cx, cy) — includes NON_ICON_SLOTS so
        # Ship Name/Type/Tier can still be suggested when the user draws near them.
        slot_pos_map: dict[str, list[tuple[int, int]]] = {}
        for ri in self._recognition_items:
            ri_bbox = ri.get('bbox')
            ri_slot = ri.get('slot', '')
            if ri_bbox and ri_slot and ri_slot in allowed:
                ri_cx = ri_bbox[0] + ri_bbox[2] // 2
                ri_cy = ri_bbox[1] + ri_bbox[3] // 2
                slot_pos_map.setdefault(ri_slot, []).append((ri_cx, ri_cy))

        if slot_pos_map:
            from warp.debug import log as _sl
            slot_order = SLOT_GROUPS.get(group_key, ALL_SLOTS)
            bx_center = bx + bw // 2

            # "Next in order" strategy: look at icon slots (not NON_ICON_SLOTS)
            # that are above OR in the same row as the new bbox.
            # Same-row slots (|Δy| < 0.5*bh) count when the new bbox is to the
            # right — handles Body Armor → EV Suit which are side-by-side.
            icon_above = []
            for slot, positions in slot_pos_map.items():
                if slot in NON_ICON_SLOTS:
                    continue
                avg_cx = sum(p[0] for p in positions) / len(positions)
                avg_cy = sum(p[1] for p in positions) / len(positions)
                same_row = abs(avg_cy - cy) < bh * 0.5
                if avg_cy < cy or (same_row and avg_cx < bx_center):
                    icon_above.append((slot, avg_cy, avg_cx))

            if icon_above:
                last_slot, last_cy, last_cx = max(icon_above, key=lambda x: x[1])
                vertically_below = cy > last_cy + bh * 0.4
                same_row_right   = abs(cy - last_cy) < bh * 0.5 and bx_center > last_cx + bw * 0.5
                # Same-row-right: Y = slot group, X = index within group.
                # Stay in the same slot unless it is already at capacity.
                # BOFF slots have no fixed max — always keep same slot.
                _SAME_ROW_MULTI: dict[str, int] = {
                    'Fore Weapons': 5, 'Aft Weapons': 5, 'Devices': 6,
                    'Kit Modules': 6, 'Weapons': 2, 'Ground Devices': 3,
                    'Engineering Consoles': 5, 'Science Consoles': 5,
                    'Tactical Consoles': 5, 'Universal Consoles': 3, 'Hangars': 4,
                    'Personal Space Traits': 10, 'Starship Traits': 7,
                    'Space Reputation': 5, 'Active Space Rep': 5,
                    'Personal Ground Traits': 10, 'Ground Reputation': 5,
                    'Active Ground Rep': 5,
                }
                # X gap limit for same-slot same-row returns.
                # Use max_cx (rightmost confirmed item) not avg_cx — consecutive
                # items in a row are ~1×bw apart; a different panel is 5-10×bw away.
                # Threshold bw*2 allows one slot gap, rejects cross-panel jumps.
                _last_max_cx = max(
                    (p[0] for p in slot_pos_map.get(last_slot, [])),
                    default=last_cx,
                )
                same_row_close = same_row_right and bx_center - _last_max_cx < bw * 1.5
                if same_row_close:
                    if last_slot.startswith('Boff '):
                        _sl.info(f'slot_suggest: bbox cy={cy} → {last_slot!r} '
                                 f'(same-row BOFF — keep slot, source=slot_order)')
                        return last_slot
                    if last_slot in _SAME_ROW_MULTI:
                        current = len(slot_pos_map.get(last_slot, []))
                        cap = _SAME_ROW_MULTI[last_slot]
                        if current < cap:
                            _sl.info(f'slot_suggest: bbox cy={cy} → {last_slot!r} '
                                     f'(same-row index {current + 1}/{cap}, source=slot_order)')
                            return last_slot

                # Next-in-order via same_row_right only for slots that are genuinely
                # side-by-side horizontally (ground layout: Body Armor → EV Suit).
                # Vertical equipment column slots (Deflector, Engines, Sec-Def…) must
                # only advance via vertically_below — they share a column, not a row.
                _HORIZONTAL_ADVANCE = frozenset({'Body Armor', 'EV Suit', 'Personal Shield'})
                advance = vertically_below or (same_row_right and last_slot in _HORIZONTAL_ADVANCE)
                if advance and last_slot in slot_order:
                    last_idx = slot_order.index(last_slot)
                    for candidate in slot_order[last_idx + 1:]:
                        if candidate in allowed and candidate not in NON_ICON_SLOTS:
                            reason = 'below' if vertically_below else 'same-row-right'
                            _sl.info(f'slot_suggest: bbox cy={cy} → {candidate!r} '
                                     f'(next-in-order after {last_slot!r}@{last_cy:.0f}, '
                                     f'{reason}, source=slot_order)')
                            return candidate

            # Nearest fallback: 2D distance (Y dominant, X at 0.4 weight).
            # Skip single-instance slots already in slot_pos_map — they are
            # already confirmed and a new bbox at a different position belongs
            # to a different slot (e.g. traits to the right of Engines).
            best_slot = ''
            best_dist = float('inf')
            for slot, positions in slot_pos_map.items():
                if slot in SINGLE_INSTANCE_SLOTS:
                    continue
                avg_cx = sum(p[0] for p in positions) / len(positions)
                avg_cy = sum(p[1] for p in positions) / len(positions)
                dist = abs(cy - avg_cy) + abs(bx_center - avg_cx) * 0.4
                if dist < best_dist:
                    best_dist = dist
                    best_slot = slot
            threshold = bh * 0.6
            if best_dist <= threshold:
                _sl.info(f'slot_suggest: bbox cy={cy} → {best_slot!r} (dist={best_dist:.0f}, '
                         f'threshold={threshold:.0f}, source=annotations)')
                return best_slot

        # ── Source 2: learned layouts from anchors.json ───────────────────────
        # Skip for MIXED screen types — their layout differs from pure EQ screens,
        # so SPACE/GROUND anchors would suggest wrong slots for trait/boff regions.
        if self._current_idx >= 0 and stype not in ('SPACE_MIXED', 'GROUND_MIXED'):
            try:
                import cv2
                from warp.recognition.layout_detector import LayoutDetector
                build_type = self._STYPE_TO_BUILD.get(stype)
                if build_type:
                    img = cv2.imread(str(self._screenshots[self._current_idx]))
                    if img is not None:
                        h, w = img.shape[:2]
                        detector = LayoutDetector()
                        cal = detector._calibration
                        if cal and 'learned' in cal:
                            aspect = round(w / h, 3)
                            candidates = [
                                e for e in cal['learned']
                                if e['type'] == build_type
                                and abs(e['aspect'] - aspect) < 0.05
                            ]
                            if candidates:
                                layout = candidates[-1]  # most recent
                                best_slot = ''
                                best_dist = float('inf')
                                for slot_name, geo in layout['slots'].items():
                                    if slot_name not in allowed or slot_name in NON_ICON_SLOTS:
                                        continue
                                    if isinstance(geo, (int, float)):
                                        slot_cy = int(geo * h)
                                    else:
                                        slot_cy = int(geo['y_rel'] * h)
                                    dist = abs(cy - slot_cy)
                                    if dist < best_dist:
                                        best_dist = dist
                                        best_slot = slot_name
                                threshold = bh * 0.8
                                if best_slot and best_dist <= threshold:
                                    from warp.debug import log as _sl
                                    _sl.info(f'slot_suggest: bbox cy={cy} → {best_slot!r} '
                                             f'(dist={best_dist:.0f}, threshold={threshold:.0f}, '
                                             f'source=anchors.json)')
                                    return best_slot
            except Exception as e:
                from warp.debug import log as _sl
                _sl.debug(f'slot_suggest: anchors lookup failed: {e}')

        return ''

    def _rematch_with_slot(self, row: int, slot: str, crop_bgr):
        """Re-run icon matching for an existing crop when the user changes the slot."""
        try:
            if slot in NON_ICON_SLOTS:
                if self._ship_type_combo.count() == 0:
                    self._populate_ship_type_combo()
                v_tiers = [self._tier_combo.itemText(i) for i in range(self._tier_combo.count())]
                v_types = [self._ship_type_combo.itemText(i) for i in range(self._ship_type_combo.count())]

                ri = self._recognition_items[row]
                ri['name'] = ''
                ri['slot'] = slot
                # Wipe any stale icon-matcher name still showing in the name field
                # (e.g. matcher returned a junk label on the ship-name crop before
                # the user switched the slot to Ship Name/Type/Tier).
                self._name_edit.blockSignals(True)
                self._name_edit.clear()
                self._name_edit.blockSignals(False)
                litem = self._review_list.item(row)
                if litem:
                    # Slot change kicks off a fresh OCR pass; show a quick
                    # "scanning" cue until `_on_ocr_finished` repopulates.
                    # Children leave col 0 blank (parent owns Slot); only
                    # standalone items get the label written there.
                    litem.setData(0, Qt.ItemDataRole.UserRole, slot)
                    if litem.parent() is None:
                        litem.setText(0, _pretty_slot(slot))
                    litem.setText(2, '[Scanning…]')
                    litem.setText(3, '')
                    litem.setText(4, 'Scanning')
                    fg = QBrush(QColor('#aaaaaa'))
                    for c in range(5):
                        litem.setForeground(c, fg)
                    # Combo-driven slot change resets any prior seat-aware
                    # grouping — the new slot is the new group key.
                    ri['_group_label'] = slot
                    self._review_list.reparent_item(
                        litem, slot, _pretty_slot(slot))
                    # NON_ICON_SLOTS (Ship Name/Type/Tier) are single-
                    # instance per screenshot — no within-group spatial
                    # sort needed, and skipping the resort here avoids
                    # invalidating the `row` index that the pending
                    # OCRWorker still holds (worker completion looks up
                    # `_recognition_items[row]`, which would point at a
                    # different item after a resort permutation).
                worker = OCRWorker(row, crop_bgr, slot, v_tiers, v_types, parent=self)
                worker.finished.connect(self._on_ocr_finished)
                worker.start()
                if not hasattr(self, '_ocr_workers'): self._ocr_workers = []
                self._ocr_workers.append(worker)
                return

            from warp.recognition.icon_matcher import SETSIconMatcher
            from warp.debug import log as _sl
            candidates = set(self._build_search_candidates(slot)) or None
            name, conf, thumb, _used_sess = SETSIconMatcher(self._sets).match(crop_bgr, candidate_names=candidates)
            _sl.info(f'rematch_slot slot={slot!r} candidates={len(candidates) if candidates else "all"} → name={name!r} conf={conf:.2f}')
            # No global fallback — if slot-scoped search can't match, show unmatched.
            # A global fallback would return items from wrong categories.
            if conf < 0.40:
                name, conf, thumb = '', 0.0, None
            _cross_check = False
            try:
                if name:
                    from warp.warp_importer import WarpImporter
                    _cross_check = not WarpImporter(sets_app=self._sets)._item_valid_for_slot(name, slot)
            except: pass

            ri = self._recognition_items[row]
            old_slot = ri.get('slot', '') or ''
            ri.update({'name': name, 'conf': conf, 'thumb': thumb, 'slot': slot, 'cross_check_failed': _cross_check})
            # Preserve seat membership when the user corrects a slot that
            # was originally part of a physical BOFF seat. Without this,
            # picking the seat's spec profession from the combo (e.g.
            # 'Boff Miracle Worker' for a T+MW seat) would split the row
            # out of its sibling seat group ('Boff Tactical+Miracle
            # Worker') into a standalone profession parent. The seat is a
            # physical coordinate-anchored group; the user's correction
            # only changes the ability identity, not the seat it sits in.
            from warp.recognition.boff_keys import (
                is_seat_keyed as _is_seat_keyed,
                group_items_by_seat as _group_by_seat,
            )
            group_label = slot
            if slot.startswith('Boff ') and slot not in NON_ICON_SLOTS:
                seat_key = ri.get('seat_key') or ''
                if not seat_key and _is_seat_keyed(old_slot):
                    seat_key = old_slot
                    ri['seat_key'] = seat_key
                if seat_key and _is_seat_keyed(seat_key):
                    siblings = [
                        it for it in self._recognition_items
                        if it is not ri and (it.get('seat_key') or '') == seat_key
                    ]
                    existing_label = next(
                        (it.get('_group_label') for it in siblings
                         if it.get('_group_label')),
                        '',
                    )
                    if existing_label:
                        group_label = existing_label
                    else:
                        grouped = _group_by_seat([ri, *siblings])
                        if grouped:
                            group_label = grouped[0][0]
            ri['_group_label'] = group_label
            self._name_edit.blockSignals(True)
            self._name_edit.setText(name)
            self._name_edit.blockSignals(False)
            litem = self._review_list.item(row)
            if litem:
                self._populate_review_item(
                    litem, name, slot, conf,
                    confirmed=False, cross_check_failed=_cross_check,
                    auto_confirmed=False, conflict_disk_name='',
                    group_label=group_label,
                )
                self._review_list.reparent_item(
                    litem, group_label, group_label)
                self._resort_group_of(litem)
                # If the slot didn't have a group yet, reparent_item
                # created the parent at the bottom — re-place parents
                # canonically so the row lands in SLOT_ORDER position.
                self._resort_parents_canonical()
            # Auto-accept if threshold met after rematch. `row` is stale
            # after the resort permuted `_recognition_items`; re-fetch
            # from the QTreeWidgetItem so Accept targets the right row.
            if (name and conf >= 0.40
                    and getattr(self, '_chk_auto_accept', None)
                    and self._chk_auto_accept.isChecked()
                    and conf >= self._spin_auto_conf.value()
                    and slot not in NON_ICON_SLOTS):
                fresh_row = self._review_list.row(litem) if litem else row
                self._review_list.setCurrentRow(fresh_row)
                self._on_accept()
        except Exception as e:
            from warp.debug import log as _sl
            _sl.warning(f'rematch_with_slot failed: {e}')

    def _on_canvas_deselected(self):
        """Canvas click on already-selected bbox or empty area → deselect everything."""
        self._review_list.blockSignals(True)
        self._review_list.setCurrentRow(-1)
        self._review_list.blockSignals(False)
        # blockSignals also silences itemSelectionChanged, which drives the
        # bold-on-selected refresh — run it manually so the old row's bold
        # font is cleared.
        self._review_list._refresh_bold_selected()
        self._set_review_buttons_enabled(False)
        self._ann_widget.clear_highlight()

    def _on_bbox_changed(self, row: int, new_bbox: tuple):
        """Shift+LMB move/resize finished — persist the new bbox and re-classify."""
        if row < 0 or row >= len(self._recognition_items): return
        ri = self._recognition_items[row]
        ri['bbox'] = new_bbox
        
        # Update underlying annotation if it exists
        ann_id = ri.get('ann_id', '')
        if ann_id and self._current_idx >= 0:
            path = self._screenshots[self._current_idx]
            for ann in self._data_mgr.get_annotations(path):
                if ann.ann_id == ann_id:
                    self._data_mgr.update_annotation(path, ann, bbox=new_bbox)
                    self._data_mgr.save()
                    break
        
        # Re-run recognition for this specific crop to update confidence
        if self._current_idx >= 0:
            try:
                import cv2
                from warp.warp_importer import WarpImporter
                path = self._screenshots[self._current_idx]
                img = cv2.imread(str(path))
                if img is not None:
                    stype = self._screen_types.get(path.name, 'UNKNOWN')
                    importer_type = {'SPACE_EQ': 'SPACE', 'GROUND_EQ': 'GROUND', 'TRAITS': 'SPACE_TRAITS',
                                     'BOFFS': 'BOFFS', 'SPECIALIZATIONS': 'SPEC',
                                     'SPACE_MIXED': 'SPACE', 'GROUND_MIXED': 'GROUND'}.get(stype, 'SPACE_EQ')
                    importer = WarpImporter(sets_app=self._sets, build_type=importer_type, from_trainer=True)
                    matcher = importer._get_matcher()
                    
                    x, y, w, h = new_bbox
                    # Ensure bbox is within image bounds
                    y = max(0, min(y, img.shape[0]-1))
                    x = max(0, min(x, img.shape[1]-1))
                    h = max(1, min(h, img.shape[0]-y))
                    w = max(1, min(w, img.shape[1]-x))
                    
                    crop = img[y:y+h, x:x+w]
                    if crop.size > 0:
                        # Refresh the cached crop so a later _on_accept (or the
                        # _contribute below for already-confirmed rows) sends
                        # the corrected crop to the backend, not the stale
                        # pre-edit one.
                        ri['crop_bgr'] = crop.copy()
                        old_name = ri.get('name', '')
                        old_conf = ri.get('conf', 0.0)
                        # Constrain matcher to names valid for the current slot
                        # (domain + career for BOFFs, item type for traits, etc.)
                        # so the embedder cannot return e.g. space-tactical
                        # 'Kemocite-Laced Weaponry' on a ground Science seat.
                        cand = set(self._build_search_candidates(ri.get('slot', '')))
                        name, conf, thumb, _used_sess = matcher.match(
                            crop, candidate_names=cand if cand else None)
                        from warp.debug import log as _slog
                        st = dict(getattr(matcher, '_last_stage_scores', {}) or {})
                        src = getattr(matcher, '_last_match_src', '')
                        ml = st.get('embed', 0) or st.get('soft', 0)
                        _slog.info(
                            f"WARP CORE: bbox_changed row={row} slot='{ri.get('slot','')}' "
                            f"bbox={new_bbox} cand={len(cand)} src={src} "
                            f"stages[embed={ml:.2f} sess={st.get('session',0):.2f} "
                            f"tmpl={st.get('template',0):.2f} knldg={st.get('knowledge',0):.2f}] "
                            f"old=('{old_name}',{old_conf:.2f}) → "
                            f"new=('{name}',{conf:.2f}) "
                            f"state={ri.get('state','')}"
                        )

                        if ri.get('state') != 'confirmed':
                            ri['name'] = name
                            ri['conf'] = conf
                            # Refresh visual row
                            self._review_list.takeItem(row)
                            self._add_review_row(name, ri['slot'], conf, confirmed=False,
                                                 group_label=ri.get('_group_label'))
                        else:
                            ri['conf'] = conf
                            ri['orig_name'] = name
                            # Refresh visual row but keep confirmed status and user-selected name
                            self._review_list.takeItem(row)
                            self._add_review_row(ri['name'], ri['slot'], conf, confirmed=True,
                                                 group_label=ri.get('_group_label'))
                            # Bbox-only correction on a confirmed row: resend
                            # the (better) crop with the user-confirmed name
                            # so the backend gets the improved training signal.
                            if ri.get('slot', '') not in NON_ICON_SLOTS and ri.get('name'):
                                self._contribute(ri, ri['name'])

                        self._review_list.insertItem(row, self._review_list.takeItem(self._review_list.count()-1))
                        self._review_list.setCurrentRow(row)
            except Exception as e:
                log.warning(f'Re-classification failed: {e}')

        self._ann_widget.set_review_items(self._recognition_items)

    def _slot_for_combo(self, slot: str) -> str:
        """Map a stored slot value to a user-pickable combo entry.

        Internal slot values may be dynamic seat keys (`Boff Seat L[U]_478`)
        or the 'Boff Universal' sentinel — neither is in SLOT_GROUPS, so
        `setCurrentText` would silently fail and leave the combo stale.
        Resolve to a real profession label that exists in the dropdown.
        """
        from warp.recognition.boff_keys import (
            parse_seat_profession, parse_seat_spec, is_seat_keyed,
        )
        if is_seat_keyed(slot):
            prof = parse_seat_profession(slot)
            if prof:
                return f'Boff {prof}'
            # Universal seat: vote from sibling abilities, then spec, then
            # default to Tactical so combo always lands on a pickable label.
            voted = self._vote_universal_profession(slot)
            spec  = parse_seat_spec(slot)
            return f'Boff {voted or spec or "Tactical"}'
        if slot == 'Boff Universal':
            # Legacy/sentinel — dropdown has no 'Universal' entry.
            return 'Boff Tactical'
        return slot

    def _on_item_selected(self, ann: dict):
        """Canvas bbox clicked → sync review list selection + fill slot/name fields."""
        slot = ann.get('slot', '')
        name = ann.get('name', '')

        # Sync review list selection to match canvas click
        bbox = ann.get('bbox')
        if bbox is not None:
            for row, ri in enumerate(self._recognition_items):
                if ri.get('bbox') == bbox:
                    self._review_list.blockSignals(True)
                    self._review_list.setCurrentRow(row)
                    self._review_list.blockSignals(False)
                    # blockSignals also silences itemSelectionChanged,
                    # which drives the bold-on-selected refresh — run it
                    # manually so the new row gets bold and the previous
                    # one loses it.
                    self._review_list._refresh_bold_selected()
                    self._ann_widget.set_highlighted_row(row)
                    self._set_review_buttons_enabled(True)
                    break

        # Map dynamic BOFF seat keys + 'Boff Universal' sentinel to a
        # user-pickable combo entry. The combo only contains static
        # SLOT_GROUPS names; seat keys / 'Boff Universal' must never
        # leak into it.
        combo_slot = self._slot_for_combo(slot)

        # Ensure slot is visible in combo (confirmed NON_ICON_SLOTS may be hidden)
        if self._current_idx >= 0:
            _stype = self._screen_types.get(
                self._screenshots[self._current_idx].name, 'UNKNOWN')
            self._refresh_slot_combo(_stype, keep_slot=combo_slot)
        # Set slot without triggering _on_slot_changed's clear() on name_edit
        self._slot_combo.blockSignals(True)
        self._slot_combo.setCurrentText(combo_slot)
        self._slot_combo.blockSignals(False)
        if slot not in NON_ICON_SLOTS:
            self._populate_name_completer(slot)

        # Set name fields
        self._configure_name_field(slot)
        if slot == 'Ship Tier':
            idx = self._tier_combo.findText(name)
            if idx >= 0:
                self._tier_combo.setCurrentIndex(idx)
        elif slot == 'Ship Type':
            self._populate_ship_type_combo()
            idx = self._ship_type_combo.findText(name)
            if idx >= 0:
                self._ship_type_combo.setCurrentIndex(idx)
            else:
                self._ship_type_combo.lineEdit().setText(name)
        else:
            self._name_edit.blockSignals(True)
            self._name_edit.setText(name)
            self._name_edit.blockSignals(False)

    # ── Auto-accept ───────────────────────────────────────────────────────────

    def _apply_auto_accept(self):
        """Auto-accept pending items with conf >= threshold.
        Called before populating review list — items are marked confirmed
        in-place so _add_review_row renders them as confirmed directly."""
        from warp.debug import log as _sl
        if not getattr(self, '_chk_auto_accept', None): return
        if not self._chk_auto_accept.isChecked(): return
        threshold = self._spin_auto_conf.value()
        accepted = 0
        path = self._screenshots[self._current_idx] if self._current_idx >= 0 else None
        for ri in self._recognition_items:
            if ri.get('state') == 'community_conflict':
                _sl.info(
                    f"auto-accept: SKIP community_conflict slot={ri.get('slot')!r} "
                    f"disk={ri.get('disk_name')!r} community={ri.get('name')!r} "
                    f"— requires manual re-verification"
                )
                continue
            if ri.get('state') != 'pending': continue
            slot = ri.get('slot', '')
            conf = ri.get('conf', 0.0)
            if conf < threshold: continue
            name = ri.get('name', '') or ri.get('orig_name', '')
            # Ship Name is position-only — no content stored, but bbox alone is enough
            # to auto-confirm (OCR found the anchor). Other NON_ICON_SLOTS need a name.
            if not slot: continue
            if slot != 'Ship Name' and not name: continue
            # Poison guard: never auto-accept a virtual label (__empty__/
            # __inactive__) that came from session. Session-virtual at high
            # conf is the self-poisoning vector — a real virtual gets
            # written, becomes a session example, and self-matches forever.
            # User must confirm virtuals manually.
            if name in VIRTUAL_ITEM_NAMES and ri.get('src') == 'session':
                _sl.info(
                    f"auto-accept: SKIP poison vector slot={slot!r} "
                    f"name={name!r} conf={conf:.2f} src='session' — "
                    f"virtual labels from session require manual confirmation"
                )
                continue
            ri['state'] = 'confirmed'
            ri['auto_confirmed'] = True
            if ri.get('bbox') and path:
                _saved = self._data_mgr.add_annotation(
                    image_path=path, bbox=ri['bbox'], slot=slot, name=name,
                    state=AnnotationState.CONFIRMED,
                    ml_conf=conf, ml_name=name,
                    auto_confirmed=True,
                    seat_key=ri.get('seat_key', '') or '',
                    slot_index=ri.get('slot_index')
                        if isinstance(ri.get('slot_index'), int) else -1,
                )
                ri['ann_id'] = _saved.ann_id
            # Seed icon-matcher session examples only for ML icon slots —
            # NON_ICON_SLOTS (Ship Name/Type/Tier) are text, not classifiable icons.
            if ri.get('crop_bgr') is not None and slot not in NON_ICON_SLOTS:
                from warp.recognition.icon_matcher import SETSIconMatcher
                # Default 'session' origin (NOT 'user'): auto-accept is the
                # model trusting itself above a threshold the user once set,
                # not an explicit human review. Tagging as 'user' would
                # (a) propagate model errors into WARP with a ✓ user badge
                # that claims "you confirmed this" when you didn't, and
                # (b) create a self-amplification loop where a wrong
                # auto-confirm becomes pixel-perfect session ground truth
                # for the next slot the same icon appears in. Only manual
                # _on_accept clicks earn the 'user' tag.
                SETSIconMatcher.add_session_example(ri['crop_bgr'], name)
            accepted += 1
        if accepted:
            self._data_mgr.save()
            _sl.info(f'TrainerWindow: auto-accepted {accepted} items '
                     f'(conf>={threshold:.2f})')

    def _run_auto_accept(self):
        """Legacy: called after panel is drawn. Now a no-op since
        _apply_auto_accept runs before _add_review_row."""
        pass

    def _on_accept(self):
        # Locked-screenshot guard. Mirrors _on_auto_detect / _on_remove_item /
        # _on_clear_all_bboxes / the Delete key handler — all destructive
        # actions are gated on `_is_current_locked()`. Without this, Enter,
        # the Accept button, autocomplete pick, Ship Type/Tier combo
        # activation and the auto-accept path could still mutate items on
        # a screenshot the user has marked Done, even though the canvas is
        # visually locked and the button reads "↩ Back to Edit".
        if self._is_current_locked():
            self.statusBar().showMessage(
                'Accept blocked: screenshot is marked Done — '
                'press ↩ Back to Edit to modify.', 6000)
            return
        slot = self._slot_combo.currentText()
        if slot == 'Ship Tier':
            name = self._tier_combo.currentText()
        elif slot == 'Ship Type':
            name = self._ship_type_combo.currentText().strip()
        else:
            name = self._name_edit.text().strip()
        # NON_ICON_SLOTS guard: if the user clicks Accept while the Ship
        # Type / Tier editor is empty (combo was blanked because OCR
        # hadn't run yet, or _on_item_selected fed it an empty `name`),
        # don't silently wipe whatever the row already had — fall back
        # to the row's own `name` / `orig_name`. Stops the "after Confirm
        # the Ship Type disappears from the bbox" footgun. The user can
        # still clear it deliberately by deleting the text and re-picking.
        if slot in ('Ship Type', 'Ship Tier') and not name:
            _row = self._review_list.currentRow()
            if 0 <= _row < len(self._recognition_items):
                _ri = self._recognition_items[_row]
                fallback = (_ri.get('name') or _ri.get('orig_name') or '').strip()
                if fallback:
                    log.info(
                        f'accept: {slot} editor was empty — keeping prior '
                        f'value {fallback!r} from recognition row'
                    )
                    name = fallback
                    if slot == 'Ship Tier':
                        idx = self._tier_combo.findText(name)
                        if idx >= 0:
                            self._tier_combo.setCurrentIndex(idx)
                    else:
                        idx = self._ship_type_combo.findText(name)
                        if idx >= 0:
                            self._ship_type_combo.setCurrentIndex(idx)
                        else:
                            self._ship_type_combo.lineEdit().setText(name)
        # Strict name validation for icon slots: only allow exact matches
        # against the slot's candidate list (or empty = Unknown). NON_ICON_SLOTS
        # have their own widgets/use cases (Ship Name = free text, Ship Type/Tier
        # via combos) so they bypass.
        if slot not in NON_ICON_SLOTS and name:
            allowed = set(self._build_search_candidates(slot)) | set(VIRTUAL_ITEM_NAMES)
            if name not in allowed:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.warning(self, 'Invalid item name',
                    f'{name!r} is not in the allowed list for slot {slot!r}.\n'
                    f'Pick from the dropdown or type an exact match.')
                self._name_edit.setFocus()
                self._name_edit.selectAll()
                return
        row = self._review_list.currentRow()
        if 0 <= row < len(self._recognition_items):
            ri = self._recognition_items[row]
            # Check for overlapping bbox with different slot (likely user error)
            if ri.get('bbox') and self._current_idx >= 0:
                path = self._screenshots[self._current_idx]
                existing = self._data_mgr.get_annotations(path)
                new_bbox = ri['bbox']
                for ann in existing:
                    if ann.state.value != 'confirmed': continue
                    if ann.slot == slot: continue  # same slot = ok
                    if ann.ann_id == ri.get('ann_id', ''): continue
                    # Ship Type and Ship Tier intentionally overlap (tier is part of type line)
                    pair = {ann.slot, slot}
                    if pair == {'Ship Type', 'Ship Tier'}: continue
                    # Check overlap
                    ox, oy, ow, oh = ann.bbox
                    nx, ny, nw, nh = new_bbox
                    ix = max(0, min(ox+ow, nx+nw) - max(ox, nx))
                    iy = max(0, min(oy+oh, ny+nh) - max(oy, ny))
                    overlap = ix * iy
                    area = min(ow*oh, nw*nh)
                    if area > 0 and overlap / area > 0.7:
                        from PySide6.QtWidgets import QMessageBox
                        ans = QMessageBox.warning(self, 'Possible duplicate',
                            f'This bbox overlaps {ann.slot!r} → {ann.name!r}\n'
                            f'Are you sure you want to confirm as {slot!r}?',
                            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel)
                        if ans != QMessageBox.StandardButton.Yes:
                            return
                        break
            prev_name = ri.get('name', '')
            was_conflict = ri.get('state') == 'community_conflict'
            # Community proposal to remember as rejected: only when user is
            # resolving a conflict AND picks a name different from what
            # community proposed (prev_name is the community-pushed value at
            # the moment the conflict surfaced). If they pick the community
            # value after all, the conflict resolves naturally — nothing to
            # mark as rejected.
            community_rejected = ''
            if was_conflict:
                log.info(
                    f'accept: resolving community conflict slot={slot!r} '
                    f'bbox={ri.get("bbox")} disk={ri.get("disk_name")!r} '
                    f'community={prev_name!r} → user picked {name!r}'
                )
                ri.pop('disk_name', None)
                if prev_name and prev_name != name:
                    community_rejected = prev_name
            ri['name'] = name
            ri['slot'] = slot
            ri['state'] = 'confirmed'
            ri['auto_confirmed'] = False  # user override → green, not yellow
            ri['community_rejected'] = community_rejected
            if ri.get('bbox') and self._current_idx >= 0:
                path = self._screenshots[self._current_idx]
                log.debug(f'accept: row={row} slot={slot!r} name={name!r} bbox={ri["bbox"]}')
                saved = self._data_mgr.add_annotation(
                    image_path=path, bbox=ri['bbox'], slot=slot, name=name,
                    state=AnnotationState.CONFIRMED,
                    ml_conf=ri.get('conf', 0.0),
                    ml_name=ri.get('ocr_raw', '') or ri.get('orig_name', ''),
                    auto_confirmed=False,
                    community_rejected=community_rejected,
                    seat_key=ri.get('seat_key', '') or '',
                    slot_index=ri.get('slot_index')
                        if isinstance(ri.get('slot_index'), int) else -1,
                )
                ri['ann_id'] = saved.ann_id  # track for future edits on this bbox
                self._ann_widget.refresh_annotations(path)
                self._ann_widget.update()  # repaint review-layer bbox in new color
            litem = self._review_list.item(row)
            if litem:
                self._populate_review_item(
                    litem, name, slot, ri.get('conf', 0.0),
                    confirmed=True, cross_check_failed=False,
                    auto_confirmed=False, conflict_disk_name='',
                    group_label=ri.get('_group_label'),
                )
            if name and ri.get('crop_bgr') is not None and slot not in NON_ICON_SLOTS:
                from warp.recognition.icon_matcher import SETSIconMatcher
                # origin='user' lets the entry survive WARP's reset_ml_session
                # filter so a subsequent WARP detection run picks it up.
                SETSIconMatcher.add_session_example(
                    ri['crop_bgr'], name, origin='user')
                self._contribute(ri, name)
            elif name and slot in TEXT_LEARNING_SLOTS:
                ocr_raw = ri.get('ocr_raw', '')
                if ocr_raw and ocr_raw != name:
                    from warp.debug import log as _slog
                    _slog.info(f'OCR correction: {ocr_raw!r} → {name!r} (queued for HF upload)')
                    from warp.recognition.text_extractor import TextExtractor
                    TextExtractor._corrections[ocr_raw] = name
        else:
            self._ann_widget.confirm_current(slot=slot, name=name)
        # Keep name_edit showing the accepted value — don't clear after accept
        self._update_progress()
        self._advance_to_next_unconfirmed(row)
        if self._current_idx >= 0:
            fname = self._screenshots[self._current_idx].name
            self._recognition_cache[fname] = list(self._recognition_items)
        self._ann_widget.clear_pending()
        self._ann_widget.set_review_items(self._recognition_items)
        self._data_mgr.save()
        self._auto_sync()
        # Refresh slot combo: hide confirmed NON_ICON_SLOTS except the one
        # currently displayed (after advance, so keep_slot reflects new row)
        if self._current_idx >= 0:
            _stype = self._screen_types.get(self._screenshots[self._current_idx].name, 'UNKNOWN')
            _cur = self._review_list.currentRow()
            _keep = (self._recognition_items[_cur]['slot']
                     if 0 <= _cur < len(self._recognition_items) else '')
            self._refresh_slot_combo(_stype, keep_slot=_keep)
        self._refresh_mark_done_btn()
        # Deferred focus — after all signals settle, return focus to list
        from PySide6.QtCore import QTimer
        QTimer.singleShot(0, self._review_list.setFocus)

    # cache.equipment item['type'] → trainer slot name
    # Mirrors EQUIPMENT_TYPES in src/constants.py + SLOT_TO_CACHE_KEY above.
    _ITEM_TYPE_TO_SLOT: dict[str, str] = {
        'Ship Fore Weapon':         'Fore Weapons',
        'Ship Aft Weapon':          'Aft Weapons',
        'Ship Weapon':              'Fore Weapons',   # generic weapon → fore by default
        'Experimental Weapon':      'Experimental',
        'Ship Deflector Dish':      'Deflector',
        'Ship Secondary Deflector': 'Sec-Def',
        'Impulse Engine':           'Engines',
        'Warp Engine':              'Warp Core',
        'Singularity Engine':       'Warp Core',
        'Ship Shields':             'Shield',
        'Ship Device':              'Devices',
        'Universal Console':        'Universal Consoles',
        'Ship Engineering Console': 'Engineering Consoles',
        'Ship Science Console':     'Science Consoles',
        'Ship Tactical Console':    'Tactical Consoles',
        'Hangar Bay':               'Hangars',
        'Body Armor':               'Body Armor',
        'EV Suit':                  'EV Suit',
        'Personal Shield':          'Personal Shield',
        'Ground Weapon':            'Weapons',
        'Kit':                      'Kit',
        'Kit Module':               'Kit Modules',
        'Ground Device':            'Ground Devices',
    }

    def _infer_slot_from_name(self, item_name: str, allowed_slots: list[str] | None = None) -> str:
        """
        Given a recognised item name, returns the most appropriate slot name
        by looking up the item's type in cache.

        allowed_slots: if provided, only slots in this list are considered.
        This enforces screen type restrictions — e.g. on a TRAITS screenshot
        only trait slots are valid, never Hangars or equipment slots.

        Returns '' if the item is not found or inferred slot is not allowed.
        """
        if not self._sets or not item_name:
            return ''

        def _allowed(slot: str) -> str:
            """Return slot if allowed, else empty string."""
            if allowed_slots is None or slot in allowed_slots:
                return slot
            return ''

        # Build reverse map: cache_key → slot name (canonical, non-cross-populated)
        canonical_cache_keys = {v: k for k, v in self._SLOT_TO_CACHE_KEY.items()}

        try:
            # First pass: look in canonical (non-cross-populated) buckets only
            for cache_key, slot_name in canonical_cache_keys.items():
                bucket = self._sets.cache.equipment.get(cache_key, {})
                entry = bucket.get(item_name)
                if entry:
                    item_type = entry.get('type', '')
                    # Use item type for most precise slot (handles Universal Console)
                    if item_type in self._ITEM_TYPE_TO_SLOT:
                        return _allowed(self._ITEM_TYPE_TO_SLOT[item_type])
                    return _allowed(slot_name)

            # Second pass: traits — check all buckets with correct cache structure
            # cache.traits[environment][trait_type][name]
            # environment: 'space' | 'ground'
            # trait_type:  'personal' | 'rep' | 'active_rep'  (migrated by datafunctions.py at load)
            if hasattr(self._sets.cache, 'starship_traits') and item_name in self._sets.cache.starship_traits:
                return _allowed('Starship Traits')
            if hasattr(self._sets.cache, 'traits'):
                t = self._sets.cache.traits
                trait_slot_map = [
                    ('space',  'rep',        'Space Reputation'),
                    ('space',  'active_rep', 'Active Space Rep'),
                    ('ground', 'rep',        'Ground Reputation'),
                    ('ground', 'active_rep', 'Active Ground Rep'),
                    ('space',  'personal',   'Personal Space Traits'),
                    ('ground', 'personal',   'Personal Ground Traits'),
                ]
                for env, ttype, slot_name in trait_slot_map:
                    try:
                        if item_name in t[env][ttype]:
                            return _allowed(slot_name)
                    except (KeyError, TypeError):
                        pass

            # Third pass: boff abilities — build reverse map from cache structure
            # cache.boff_abilities[env][career][rank_idx] = {ability_name: desc}
            if hasattr(self._sets.cache, 'boff_abilities'):
                boff_cache = self._sets.cache.boff_abilities
                for env in ('space', 'ground'):
                    env_data = boff_cache.get(env, {})
                    if not isinstance(env_data, dict):
                        continue
                    for career, rank_list in env_data.items():
                        if not isinstance(rank_list, list):
                            continue
                        for rank_dict in rank_list:
                            if isinstance(rank_dict, dict) and item_name in rank_dict:
                                return _allowed(f'Boff {career}')

        except Exception:
            pass

        return ''

    # Mapping: trainer slot name → cache.equipment key
    # Must stay in sync with EQUIPMENT_TYPES in src/constants.py and SLOT_GROUPS above.
    _SLOT_TO_CACHE_KEY: dict[str, str] = {
        'Fore Weapons':          'fore_weapons',
        'Aft Weapons':           'aft_weapons',
        'Experimental':          'experimental',
        'Deflector':             'deflector',
        'Sec-Def':               'sec_def',
        'Engines':               'engines',
        'Warp Core':             'core',
        'Shield':                'shield',
        'Devices':               'devices',
        'Universal Consoles':    'uni_consoles',
        'Engineering Consoles':  'eng_consoles',
        'Science Consoles':      'sci_consoles',
        'Tactical Consoles':     'tac_consoles',
        'Hangars':               'hangars',
        'Body Armor':            'armor',
        'EV Suit':               'ev_suit',
        'Personal Shield':       'personal_shield',
        'Weapons':               'weapons',
        'Kit':                   'kit',
        'Kit Modules':           'kit_modules',
        'Ground Devices':        'ground_devices',
    }

    def _vote_universal_profession(self, seat_key: str) -> str | None:
        """Universal BOFF seats (`Boff Seat L[U]_*` / `Boff Seat L[U+spec]_*`)
        carry no inherent profession — the player decides which boff sits
        there. Vote on profession from abilities already recognized in this
        seat by looking each name up in `cache.boff_abilities[env][prof]`.
        Returns the most-frequent profession or None if no votes yet.
        """
        if not self._sets or not seat_key.startswith('Boff Seat'):
            return None
        # Pick environment from current screen type (space/ground)
        domain = 'space'
        if self._current_idx >= 0:
            stype = self._screen_types.get(
                self._screenshots[self._current_idx].name, 'UNKNOWN')
            if 'GROUND' in stype:
                domain = 'ground'
        try:
            env_abilities = self._sets.cache.boff_abilities.get(domain, {})
        except Exception:
            return None
        votes: dict[str, int] = {}
        for ri in self._recognition_items:
            if ri.get('slot') != seat_key:
                continue
            name = ri.get('name') or ri.get('orig_name') or ''
            if not name or name in VIRTUAL_ITEM_NAMES:
                continue
            for prof, rank_lists in env_abilities.items():
                hit = any(isinstance(rd, dict) and name in rd for rd in rank_lists)
                if hit:
                    votes[prof] = votes.get(prof, 0) + 1
                    break
        if not votes:
            return None
        return max(votes.items(), key=lambda kv: kv[1])[0]

    def _build_search_candidates(self, slot: str = '') -> list[str]:
        candidates: list[str] = []
        if not self._sets:
            return candidates

        stype = 'UNKNOWN'
        if self._current_idx >= 0:
            path = self._screenshots[self._current_idx]
            stype = self._screen_types.get(path.name, 'UNKNOWN')
        target_domain = 'Ground' if 'GROUND' in stype else 'Space'

        if slot.startswith('Boff'):
            from warp.recognition.boff_keys import parse_seat_profession, is_seat_keyed
            parsed_prof = parse_seat_profession(slot)
            if parsed_prof:
                target_career: str | None = parsed_prof
            elif slot == 'Boff Universal' or is_seat_keyed(slot):
                # Universal seat (with or without spec) — abilities of any
                # profession are valid; the spec marker only constrains
                # which abilities CAN sit there for the build, but for the
                # candidate pool we accept all professions.
                target_career = None
            else:
                target_career = slot.replace('Boff ', '').strip()

            # Determine domain(s) to search
            try:
                if 'GROUND' in stype:
                    domains = ['ground']
                elif 'SPACE' in stype:
                    domains = ['space']
                else:
                    domains = ['space', 'ground']

                # cache.boff_abilities = {env: {career: [rank0_dict, ...]}}
                for domain_key in domains:
                    env = self._sets.cache.boff_abilities.get(domain_key, {})
                    if target_career is None:
                        # Universal — pull every profession's abilities
                        for rank_lists in env.values():
                            for rank_dict in rank_lists:
                                if isinstance(rank_dict, dict):
                                    candidates.extend(rank_dict.keys())
                    else:
                        career_ranks = env.get(target_career, [])
                        for rank_dict in career_ranks:
                            if isinstance(rank_dict, dict):
                                candidates.extend(rank_dict.keys())
            except Exception:
                pass

            # Last resort: all abilities
            if not candidates:
                try:
                    candidates.extend(self._sets.cache.boff_abilities.get('all', {}).keys())
                except Exception:
                    pass
        elif slot in ('Primary Specialization', 'Secondary Specialization'):
            candidates.extend(SPECIALIZATION_NAMES)
        elif 'Starship Trait' in slot:
            # cache.starship_traits = {name: {...}} flat dict
            try:
                candidates.extend(self._sets.cache.starship_traits.keys())
            except Exception:
                pass
        elif slot == 'Active Space Rep':
            try:
                candidates.extend(self._sets.cache.traits['space']['active_rep'].keys())
            except Exception:
                pass
        elif slot == 'Space Reputation':
            try:
                candidates.extend(self._sets.cache.traits['space']['rep'].keys())
            except Exception:
                pass
        elif slot == 'Active Ground Rep':
            try:
                candidates.extend(self._sets.cache.traits['ground']['active_rep'].keys())
            except Exception:
                pass
        elif slot == 'Ground Reputation':
            try:
                candidates.extend(self._sets.cache.traits['ground']['rep'].keys())
            except Exception:
                pass
        elif slot == 'Personal Space Traits':
            try:
                candidates.extend(self._sets.cache.traits['space']['personal'].keys())
            except Exception:
                pass
        elif slot == 'Personal Ground Traits':
            try:
                candidates.extend(self._sets.cache.traits['ground']['personal'].keys())
            except Exception:
                pass
        else:
            cache_key = self._SLOT_TO_CACHE_KEY.get(slot)
            try:
                if cache_key:
                    candidates.extend(self._sets.cache.equipment.get(cache_key, {}).keys())
                else:
                    for cat_items in self._sets.cache.equipment.values():
                        candidates.extend(cat_items.keys())
            except Exception:
                pass

        return sorted(set(candidates))

    def _populate_name_completer(self, slot: str):
        """Pre-populate the completer model for the given slot (called on slot change)."""
        all_names = self._build_search_candidates(slot)
        self._completer_model.clear()
        # Virtual names always appear at the top of the dropdown
        for vname in sorted(VIRTUAL_ITEM_NAMES):
            self._completer_model.appendRow(QStandardItem(vname))
        for name in all_names:
            self._completer_model.appendRow(QStandardItem(name))

    def _configure_name_field(self, slot: str) -> None:
        """Single point of truth for name-input widget state.

        Controls visibility, editability, label and placeholder based on slot.
        Call from both _on_slot_changed and _on_item_selected so the rules
        are never duplicated.
        """
        is_tier      = (slot == 'Ship Tier')
        is_ship_type = (slot == 'Ship Type')
        is_non_icon  = slot in NON_ICON_SLOTS  # Ship Name / Type / Tier

        self._tier_combo.setVisible(is_tier)
        self._ship_type_combo.setVisible(is_ship_type)
        self._name_edit.setVisible(not is_tier and not is_ship_type)
        # When the current screenshot is Mark Done-locked the field must
        # stay disabled regardless of slot — otherwise selecting another
        # item silently re-enables typing on a locked screenshot.
        self._name_edit.setEnabled(not is_non_icon and not self._is_current_locked())

        if is_tier:
            self._name_label.setText('Tier:')
        elif is_ship_type:
            self._name_label.setText('Ship Type:')
        elif is_non_icon:
            self._name_label.setText('Ship Name:')
            self._name_edit.setPlaceholderText('Position only — OCR reads this automatically')
        else:
            self._name_label.setText('Item name:')
            self._name_edit.setPlaceholderText("Item name (or leave blank for 'Unknown')")

    def _on_slot_changed(self, slot: str):
        self._configure_name_field(slot)
        if slot == 'Ship Type':
            self._populate_ship_type_combo()
        # Clear item name field and reset completer state whenever slot changes
        self._suppress_next_focus_popup = True
        self._name_edit.blockSignals(True)
        self._name_edit.clear()
        self._name_edit.blockSignals(False)
        self._suppress_next_focus_popup = False
        # CRITICAL: reset QCompleter's internal completionPrefix so it doesn't
        # filter the new slot's list using the old slot's search text
        self._completer.setCompletionPrefix('')
        # Pre-populate completer with new slot's candidates
        if slot not in NON_ICON_SLOTS:
            self._populate_name_completer(slot)
        # Re-run icon matching or OCR with new slot's candidates (user-initiated change only)
        if not self._loading_row:
            row = self._review_list.currentRow()
            if 0 <= row < len(self._recognition_items):
                ri = self._recognition_items[row]
                crop_bgr = ri.get('crop_bgr')
                
                # If bbox was loaded from saved annotations, crop_bgr is initially None. Fetch it now.
                if crop_bgr is None and self._current_idx >= 0:
                    import cv2
                    img = cv2.imread(str(self._screenshots[self._current_idx]))
                    if img is not None:
                        bbox = ri.get('bbox')
                        if bbox:
                            x, y, w, h = bbox
                            crop_bgr = img[y:y+h, x:x+w].copy()
                            ri['crop_bgr'] = crop_bgr
                            
                if crop_bgr is not None:
                    if ri.get('state') == 'confirmed':
                        ri['state'] = 'pending'
                    self._rematch_with_slot(row, slot, crop_bgr)

    def _populate_ship_type_combo(self):
        if self._ship_type_combo.count() > 0:
            return
        names: list[str] = []
        if self._sets:
            try:
                names = sorted(self._sets.cache.ships.keys())
            except:
                pass
        if not names:
            self._ship_type_combo.lineEdit().setPlaceholderText('Cache not loaded')
            return
        for n in names:
            self._ship_type_combo.addItem(n)
        self._ship_type_combo.setCurrentIndex(-1)
        self._ship_type_combo.lineEdit().clear()

    def _on_name_focus_in(self, event):
        """On first focus (field was not focused before): open dropdown unless suppressed.
        Qt fires focusInEvent THEN mousePressEvent on the same click.
        We open here only when focus came from keyboard (tab) or programmatic setFocus.
        Mouse click is handled entirely by _on_name_mouse_press to avoid double-firing.
        """
        QLineEdit.focusInEvent(self._name_edit, event)
        if self._suppress_next_focus_popup:
            self._suppress_next_focus_popup = False
            return
        from PySide6.QtCore import Qt as _Qt
        if event.reason() == _Qt.FocusReason.MouseFocusReason:
            # Will be handled by _on_name_mouse_press — skip here to avoid double-open
            return
        self._show_name_dropdown()

    def _on_name_mouse_press(self, event):
        """Open/close dropdown on every mouse click in the field."""
        from PySide6.QtWidgets import QLineEdit as _QLE
        _QLE.mousePressEvent(self._name_edit, event)
        if self._suppress_next_focus_popup:
            self._suppress_next_focus_popup = False
            return
        self._show_name_dropdown()

    def _show_name_dropdown(self):
        """Toggle completer popup for the current slot."""
        slot = self._slot_combo.currentText()
        if slot in NON_ICON_SLOTS:
            return
        popup = self._completer.popup()
        if popup and popup.isVisible():
            popup.hide()
            return
        if self._completer_model.rowCount() == 0:
            self._populate_name_completer(slot)
        if not self._completer_model.rowCount():
            return
        # PopupCompletion + MatchContains can silently skip showing the popup
        # when the field is empty (e.g. right after selecting an unmatched
        # item). Switch to Unfiltered for the empty case so the full slot
        # list always appears on click; _on_name_edited flips back to
        # PopupCompletion as soon as the user starts typing.
        if self._name_edit.text():
            self._completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
            self._completer.setCompletionPrefix(self._name_edit.text())
        else:
            self._completer.setCompletionMode(QCompleter.CompletionMode.UnfilteredPopupCompletion)
        self._completer.complete()

    def _on_name_edited(self, text: str):
        slot = self._slot_combo.currentText()
        if slot in NON_ICON_SLOTS:
            self._completer_model.clear()
            return
        # User started typing — restore filtered popup mode (Unfiltered was
        # set in _show_name_dropdown for empty-field clicks).
        self._completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        query = text.strip().lower()
        all_names = self._build_search_candidates(slot)
        if not query:
            # Empty field — show full slot list (already in model from _populate_name_completer)
            # Just trigger the popup if model has items
            if self._completer_model.rowCount():
                self._completer.complete()
            return
        matches = [n for n in all_names if query in n.lower()][:60]
        self._completer_model.clear()
        if matches:
            for name in matches:
                self._completer_model.appendRow(QStandardItem(name))
        else:
            # No match for typed query — fall back to the full allowed list
            # so the user can still pick from the dropdown instead of staring
            # at an empty popup. Virtuals first, then all candidates.
            for vname in sorted(VIRTUAL_ITEM_NAMES):
                self._completer_model.appendRow(QStandardItem(vname))
            for name in all_names:
                self._completer_model.appendRow(QStandardItem(name))
        if self._completer_model.rowCount():
            self._completer.complete()

    def _on_completer_activated(self, text: str):
        self._name_edit.setText(text)
        
        try:
            if self._sets and hasattr(self._sets.cache, 'boff_abilities'):
                boff_cache = self._sets.cache.boff_abilities
                found_career = None
                for env in ['space', 'ground']:
                    for career, rank_list in boff_cache.get(env, {}).items():
                        if not isinstance(rank_list, list): continue
                        for rank_dict in rank_list:
                            if isinstance(rank_dict, dict) and text in rank_dict:
                                found_career = career
                                break
                        if found_career: break
                    if found_career: break
                
                if found_career:
                    self._slot_combo.setCurrentText(f'Boff {found_career}')
        except Exception:
            pass
        # Selection from dropdown = immediate confirm, no need to click Accept
        self._on_accept()
        self._review_list.setFocus()

    def _auto_sync(self):
        """Upload is now handled by SyncManager (app-level background timer). No-op."""
        pass

    def _on_sync_timer(self):
        """Called every 5 minutes — refreshes community knowledge and checks for a newer model.
        Crop upload is handled by SyncManager (started at app launch in warp_button.py)."""
        # Refresh community knowledge (pHash overrides)
        if self._sync_client:
            try:
                self._sync_client.refresh_knowledge()
            except Exception as e:
                log.debug(f'WARP CORE: knowledge refresh error: {e}')

        # Check for newer central model (rate-limited to once per 15 min internally)
        try:
            from warp.trainer.model_updater import ModelUpdater
            ModelUpdater().check_and_update()
        except Exception as e:
            log.debug(f'WARP CORE: model update check error: {e}')

    # Screen type → importer build_type mapping (same as RecognitionWorker)
    _STYPE_TO_BUILD: dict[str, str] = {
        'SPACE_EQ':        'SPACE',
        'GROUND_EQ':       'GROUND',
        'TRAITS':          'SPACE_TRAITS',
        'BOFFS':           'BOFFS',
        'SPECIALIZATIONS': 'SPEC',
        'SPACE_MIXED':     'SPACE',
        'GROUND_MIXED':    'GROUND',
    }

    # ── Done state ────────────────────────────────────────────────────────────

    def _load_done(self) -> set[str]:
        try:
            p = self._data_mgr._dir / 'screenshots_done.json'
            if p.exists():
                data = json.loads(p.read_text(encoding='utf-8'))
                return set(data) if isinstance(data, list) else set()
        except Exception:
            pass
        return set()

    def _save_done(self):
        try:
            p = self._data_mgr._dir / 'screenshots_done.json'
            p.write_text(json.dumps(sorted(self._screenshots_done), indent=2), encoding='utf-8')
        except Exception as e:
            log.warning(f'WarpCore: failed to save done state: {e}')

    def _file_item_color(self, path: Path):
        """Return foreground color for a file list item based on done/annotation state."""
        if path.name in self._screenshots_done:
            return QColor('#7effc8')   # green — done
        if self._data_mgr.has_annotations(path):
            return QColor('#7ec8ff')   # light blue — in progress
        return QColor(Qt.GlobalColor.white)

    _MARK_DONE_TAIL_SEP = '  ·  Mark Done: '

    def _refresh_mark_done_btn(self):
        """Visually gate the Mark Done button on the current screenshot.

        Disabled (greyed out) whenever any review row is not yet
        user-confirmed — i.e. anything that is not (state=='confirmed'
        AND auto_confirmed is False). Covers Match / Low / Unmatched /
        Type ✕ / Conflict / Auto rows alike. When the screenshot is
        already Done, the button stays enabled so the user can toggle
        Back to Edit.

        Also updates the review summary with how many items remain to
        confirm before Mark Done un-greys.
        """
        # Strip any prior Mark Done tail before recomputing — every
        # caller of setText writes a fresh base string, so we manage
        # the suffix here in one place.
        base = self._review_summary.text()
        cut = base.find(self._MARK_DONE_TAIL_SEP)
        if cut != -1:
            base = base[:cut]
        suffix = ''

        if self._current_idx < 0:
            self._btn_done.setEnabled(False)
            self._btn_done.setToolTip('')
            self._review_summary.setText(base)
            return
        path = self._screenshots[self._current_idx]
        if path.name in self._screenshots_done:
            self._btn_done.setEnabled(True)
            self._btn_done.setToolTip('')
            self._review_summary.setText(base)
            return
        pending = sum(
            1 for ri in self._recognition_items
            if not (ri.get('state') == 'confirmed'
                    and not ri.get('auto_confirmed'))
        )
        if pending:
            self._btn_done.setEnabled(False)
            self._btn_done.setToolTip(
                f'Mark Done blocked: {pending} item(s) still not '
                f'confirmed — confirm them first.')
            suffix = f'{self._MARK_DONE_TAIL_SEP}{pending} to confirm'
        else:
            self._btn_done.setEnabled(True)
            self._btn_done.setToolTip('')
            if self._recognition_items:
                suffix = f'{self._MARK_DONE_TAIL_SEP}ready'
        self._review_summary.setText(base + suffix)

    def _on_done_toggle(self, checked: bool):
        if self._current_idx < 0:
            self._btn_done.setChecked(False)
            return
        path = self._screenshots[self._current_idx]
        if checked:
            pending = [
                ri for ri in self._recognition_items
                if not (ri.get('state') == 'confirmed'
                        and not ri.get('auto_confirmed'))
            ]
            if pending:
                self._btn_done.blockSignals(True)
                self._btn_done.setChecked(False)
                self._btn_done.blockSignals(False)
                msg = (f'Mark Done blocked: {len(pending)} item(s) still '
                       f'not confirmed — confirm them first.')
                self.statusBar().showMessage(msg, 6000)
                log.info(f'mark_done: blocked for {path.name} — '
                         f'{len(pending)} unconfirmed item(s)')
                return
            self._screenshots_done.add(path.name)
            self._learn_layout_for(path)
            self._btn_done.setText('↩ Back to Edit')
        else:
            self._screenshots_done.discard(path.name)
            self._remove_layout_for(path)
            self._btn_done.setText('✓ Mark Done')
        self._save_done()
        self._ann_widget.set_locked(checked)
        # Re-gate the annotate panel (Slot combo, Item field, Accept, Tier /
        # Ship Type combos): locked → disabled, unlocked → re-enabled. Without
        # this the panel stays editable after Mark Done is toggled on, and the
        # user can keep typing into the Name field on a locked screenshot.
        stype = self._screen_types.get(path.name, 'UNKNOWN')
        self._update_screen_type_ui(stype)
        self._update_file_list_color(self._current_idx)
        self._update_add_bbox_btn()
        self._refresh_mark_done_btn()

    def _on_send_to_warp(self):
        """Build an ImportResult from confirmed annotations and emit
        `send_to_warp` so the launcher can install it into WARP and switch
        tabs.

        Single-screenshot training mode: sends only the currently loaded
        screenshot (must be Mark Done — see `_update_add_bbox_btn`).

        Fast Correction Mode: sends *every* file in the ephemeral batch.
        The toolbar gates the button on all-files-done, so we don't
        re-check per-file here — we just iterate `self._screenshots` and
        merge each file's confirmed annotations into one ImportResult.
        """
        if self._mode == 'fast_correction':
            files = list(self._screenshots)
        else:
            if self._current_idx < 0 or not self._is_current_locked():
                return
            files = [self._screenshots[self._current_idx]]
        if not files:
            return
        from warp.warp_importer import ImportResult, RecognisedItem
        result = ImportResult(build_type='')
        total_items = 0
        for path in files:
            anns = [a for a in self._data_mgr.get_annotations(path)
                    if a.state == AnnotationState.CONFIRMED]
            if not anns:
                continue
            stype = self._screen_types.get(path.name, 'UNKNOWN')
            build_type = 'GROUND' if stype.startswith('GROUND') else 'SPACE'
            # FC: map staged → orig path so WARP keys by the user's real file.
            emit_path = path
            if self._mode == 'fast_correction' and self._fast_session is not None:
                orig = self._fast_session.orig_for(path)
                if orig is not None:
                    emit_path = orig
            src_str = str(emit_path)
            by_slot: dict[str, list] = {}
            file_ship = {'name': '', 'type': '', 'tier': ''}
            for a in anns:
                if a.slot == 'Ship Name':
                    file_ship['name'] = a.name or file_ship['name']
                    continue
                if a.slot == 'Ship Type':
                    file_ship['type'] = a.name or file_ship['type']
                    continue
                if a.slot == 'Ship Tier':
                    file_ship['tier'] = a.name or file_ship['tier']
                    continue
                by_slot.setdefault(a.slot, []).append(a)
            for slot, group in by_slot.items():
                group.sort(key=lambda a: (a.bbox[0] if a.bbox else 0,
                                          a.bbox[1] if a.bbox else 0))
                for idx, a in enumerate(group):
                    result.items.append(RecognisedItem(
                        slot=slot,
                        slot_index=idx,
                        name=a.name or '',
                        confidence=a.ml_conf or 1.0,
                        thumbnail=None,
                        source_file=src_str,
                        bbox=a.bbox or (),
                        # Preserve the original seat marker (Boff Seat
                        # L[U]_93 etc.) so build_writer's cluster→seat
                        # matcher can route U-marker'd clusters to the
                        # ship's Universal seat instead of competing for
                        # explicit-prof seats via content voting.
                        seat_key=a.seat_key or '',
                        src='user',
                        match_origin='user',
                    ))
                    total_items += 1
            result.per_file[src_str] = build_type
            result.per_file_screen_type[src_str] = stype
            # Best-score ship selection (tier > type > name) — mirrors
            # `WarpImporter.process_folder` so multi-file batches pick the
            # most informative source instead of last-write-wins.
            new_score = (bool(file_ship['tier']),
                         bool(file_ship['type']),
                         bool(file_ship['name']))
            cur_score = (bool(result.ship_tier),
                         bool(result.ship_type),
                         bool(result.ship_name))
            if new_score > cur_score:
                result.ship_name = file_ship['name'] or result.ship_name
                result.ship_type = file_ship['type']
                result.ship_tier = file_ship['tier']
                result.build_type = build_type
            elif not result.build_type:
                result.build_type = build_type
        if total_items == 0:
            self.statusBar().showMessage(
                'Nothing to send — no confirmed annotations.', 5000)
            return
        if not result.build_type:
            result.build_type = 'SPACE'
        log.info(
            f'send_to_warp: files={len(files)} '
            f'items={total_items} build_type={result.build_type} '
            f'ship={result.ship_name!r}/{result.ship_type!r}/{result.ship_tier!r}'
            f'{" (fast-mode → orig)" if self._mode == "fast_correction" else ""}'
        )
        self.send_to_warp.emit(result)

    def _remove_layout_for(self, path: Path):
        from warp.recognition.layout_detector import LayoutDetector
        LayoutDetector().remove_layout(path.name)
        log.info(f'Layout learn: removed entries for {path.name}')

    def _update_file_list_color(self, row: int):
        item = self._file_list.item(row)
        if item is not None and row < len(self._screenshots):
            item.setForeground(self._file_item_color(self._screenshots[row]))

    def _learn_layout_for(self, path: Path) -> bool:
        """Save confirmed layout for one screenshot to anchors.json. Returns True if saved."""
        try:
            anns = self._data_mgr.get_annotations(path)
            confirmed = [{'bbox': a.bbox, 'slot': a.slot} for a in anns
                         if a.state == AnnotationState.CONFIRMED and not a.auto_confirmed]
            if not confirmed:
                return False
            stype = self._screen_types.get(path.name, 'UNKNOWN')
            build_type = self._STYPE_TO_BUILD.get(stype)
            if not build_type:
                log.debug(f'Layout learn: {path.name} — stype={stype!r} unknown, skipping')
                return False
            import cv2
            from warp.recognition.layout_detector import LayoutDetector
            img = cv2.imread(str(path))
            if img is None:
                return False
            LayoutDetector().learn_layout(build_type, img.shape[:2], confirmed, source_file=path.name)
            log.info(f'Layout learn: {path.name} [{build_type}] — {len(confirmed)} slots saved to anchors.json')
            return True
        except Exception as e:
            log.warning(f'Layout learn: error for {path.name}: {e}')
            return False

    def _learn_all_layouts(self):
        """Save confirmed layouts for all screenshots to anchors.json."""
        import cv2
        from warp.recognition.layout_detector import LayoutDetector
        detector = LayoutDetector()
        learned_count = 0
        skipped_unknown = 0
        for path in self._screenshots:
            anns = self._data_mgr.get_annotations(path)
            confirmed = [{'bbox': a.bbox, 'slot': a.slot} for a in anns
                         if a.state == AnnotationState.CONFIRMED and not a.auto_confirmed]
            if not confirmed:
                continue
            stype = self._screen_types.get(path.name, 'UNKNOWN')
            build_type = self._STYPE_TO_BUILD.get(stype)
            if not build_type:
                skipped_unknown += 1
                log.debug(f'Layout learn: {path.name} — stype={stype!r} unknown, skipping')
                continue
            img = cv2.imread(str(path))
            if img is not None:
                detector.learn_layout(build_type, img.shape[:2], confirmed)
                log.info(f'Layout learn: {path.name} [{build_type}] — {len(confirmed)} slots')
                learned_count += 1
        log.info(f'Layout learning: saved {learned_count} layouts'
                 + (f', skipped {skipped_unknown} (unknown screen type)' if skipped_unknown else '')
                 + ' → anchors.json')

    def _update_progress(self):
        total = len(self._screenshots)
        annotated = sum(1 for p in self._screenshots if self._data_mgr.has_annotations(p))
        confirmed_types = self._data_mgr.get_screen_type_counts()
        confirmed_total = sum(confirmed_types.values())
        # Build compact per-type summary for status bar
        if confirmed_types:
            type_summary = '  |  Screen types: ' + '  '.join(
                f'{SCREEN_TYPE_ICONS.get(k,"?")}{v}' for k, v in sorted(confirmed_types.items()))
        else:
            type_summary = '  |  No confirmed screen types yet'
        self._prog_lbl.setText(
            f'{annotated}/{total} annotated  ·  {confirmed_total} confirmed screen types{type_summary}')
        self._prog_bar.setValue(int(100 * annotated / max(1, total)))
        self._file_list.blockSignals(True)
        for row, p in enumerate(self._screenshots):
            item = self._file_list.item(row)
            if item:
                stype = self._screen_types.get(p.name, 'UNKNOWN')
                icon = SCREEN_TYPE_ICONS.get(stype, '?')
                label = SCREEN_TYPE_LABELS.get(stype, 'Unknown')
                is_user = p.name in self._screen_types_manual
                is_ml   = p.name in self._screen_types_ml_auto
                item.setText(f'{icon} {label}\n  {_disp_name(p.name)}')
                # Checkbox tracks user-confirmation only — see `_make_file_list_item`.
                item.setCheckState(
                    Qt.CheckState.Checked if is_user else Qt.CheckState.Unchecked)
                item.setIcon(_get_user_icon() if is_user
                             else (_get_ml_icon() if is_ml else QIcon()))
                item.setForeground(self._file_item_color(p))
        self._file_list.blockSignals(False)

    def _find_sets_root(self) -> Path:
        p = Path(__file__).resolve()
        for _ in range(8):
            if (p / 'pyproject.toml').exists():
                return p
            p = p.parent
        return Path('.')

    def closeEvent(self, event):
        try:
            from PySide6.QtWidgets import QApplication
            QApplication.instance().removeEventFilter(self)
        except Exception:
            pass
        if hasattr(self, '_sync_timer') and self._sync_timer:
            self._sync_timer.stop()

        # Ask all live workers to bail at their next checkpoint, but do
        # not block the UI thread on them — the close button must feel
        # instantaneous. Anything mid-flight gets reaped at interpreter
        # exit. Order: cheapest first so a quick worker actually finishes.
        for attr in ('_ocr_workers',):
            for w in getattr(self, attr, None) or []:
                if w.isRunning():
                    w.requestInterruption()
        for attr in ('_detect_worker', '_recog_worker'):
            w = getattr(self, attr, None)
            if w is not None and w.isRunning():
                w.requestInterruption()
                w.quit()
        # Single short grace window covering all of them combined.
        for attr in ('_detect_worker', '_recog_worker'):
            w = getattr(self, attr, None)
            if w is not None and w.isRunning():
                w.wait(200)

        if hasattr(self, '_data_mgr') and self._data_mgr:
            self._data_mgr.save()
        event.accept()
