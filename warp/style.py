# warp/style.py
# Theme-aware QSS / button helpers for WARP and WARP CORE windows.
# Color values are NOT defined here — they come from `warp.themes` so a
# new palette can be added by editing one file and (optionally) setting
# the `WARP_THEME` env var. The chrome and semantic-color names below are
# re-exported as module attributes for backwards compat with callers that
# do `from warp.style import BG, FG, ACCENT, …`.

from PySide6.QtGui import QPalette, QColor
from PySide6.QtWidgets import QWidget

from warp.themes import get_active as _get_theme

_T = _get_theme()

# ── Chrome ───────────────────────────────────────────────────────────────────
BG     = _T.BG
MBG    = _T.MBG
LBG    = _T.LBG
ACCENT = _T.ACCENT
FG     = _T.FG
MFG    = _T.MFG
BC     = _T.BC
TAB_SEL_BG = _T.TAB_SEL_BG

# ── Semantic state colors ────────────────────────────────────────────────────
C_CONFIRMED  = _T.C_CONFIRMED
C_ERROR      = _T.C_ERROR
C_CONF_HIGH  = _T.C_CONF_HIGH
C_CONF_MED   = _T.C_CONF_MED
C_WARNING    = _T.C_WARNING
C_SUCCESS    = _T.C_SUCCESS
C_FAILURE    = _T.C_FAILURE

# ── Button style helpers ─────────────────────────────────────────────────────

def primary_btn_style() -> str:
    """Heavy action button — gold background, dark text (matches SETS heavy_button)."""
    return (
        f'QPushButton {{'
        f'background:{ACCENT};color:#1a1a1a;'
        f'border:1px solid {ACCENT};border-radius:3px;'
        f'padding:5px 14px;font-weight:bold;}}'
        f'QPushButton:hover{{background:#d4a030;}}'
        f'QPushButton:disabled{{background:{LBG};color:{BC};border-color:{LBG};}}'
    )

def secondary_btn_style(checked_border: bool = False) -> str:
    """Regular button — transparent background, neutral gray border.
    Hover lifts to the accent color so the button still feels interactive
    without shouting gold on every toolbar entry."""
    checked = f'QPushButton:checked{{background:{MBG};border:1px solid {ACCENT};}}' if checked_border else ''
    return (
        f'QPushButton {{'
        f'background:transparent;color:{FG};'
        f'border:1px solid {BC};border-radius:3px;padding:4px 10px;}}'
        f'QPushButton:hover{{border-color:{ACCENT};}}'
        f'QPushButton:disabled{{color:{BC};border-color:{LBG};}}'
        + checked
    )

def warning_btn_style(checked_border: bool = False) -> str:
    """Button with amber/warning accent — for Add BBox and similar attention actions."""
    checked = f'QPushButton:checked{{background:{MBG};border:2px solid {C_WARNING};}}' if checked_border else ''
    return (
        f'QPushButton {{'
        f'background:transparent;color:{C_WARNING};'
        f'border:1px solid {C_WARNING};border-radius:3px;padding:4px 8px;}}'
        f'QPushButton:hover{{background:{MBG};}}'
        + checked
    )
def toggle_yellow_btn_style() -> str:
    """Toggle button that fills yellow/amber (C_WARNING) when checked."""
    return (
        f'QPushButton {{'
        f'background:transparent;color:{C_WARNING};'
        f'border:1px solid {C_WARNING};border-radius:3px;padding:4px 10px;}}'
        f'QPushButton:hover{{background:{MBG};}}'
        f'QPushButton:checked{{background:{C_WARNING};color:{BG};font-weight:bold;}}'
    )


def danger_btn_style() -> str:
    """Destructive action button — red accent."""
    return (
        f'QPushButton {{'
        f'background:transparent;color:{C_CONF_HIGH};'
        f'border:1px solid #884040;border-radius:3px;padding:4px 8px;}}'
        f'QPushButton:hover{{background:#2a1a1a;}}'
    )

# ── Accent overlays ─────────────────────────────────────────────────────────

def accent_qss(name: str) -> str:
    """Return a QSS overlay for a named theme accent variant.

    Designed to be applied via `widget.setStyleSheet(accent_qss(name))`
    on a specific widget subtree so only that subtree is recoloured —
    the rest of the QApplication keeps the base theme.

    Returns an empty string when the active theme does not declare the
    requested accent, which makes it safe to apply unconditionally.

    Targets object-name-scoped widgets so the overlay does NOT cascade
    onto every QWidget / QPushButton inside the subtree:
      - `QMainWindow` background tint (subtle warm wash on the chrome)
      - widgets with `objectName == 'accent_banner'` — prominent banner
      - widgets with `objectName == 'accent_exit_btn'`  — exit action
    """
    theme = _get_theme()
    a = theme.accents.get(name) if hasattr(theme, 'accents') else None
    if not a:
        return ''
    bg     = a.get('BG',     theme.BG)
    accent = a.get('ACCENT', theme.ACCENT)
    border = a.get('BORDER', accent)
    return (
        f'QMainWindow {{ background-color: {bg}; }}'
        f'QFrame#accent_banner {{'
        f'  background-color: {bg};'
        f'  color: {accent};'
        f'  border: 2px solid {border};'
        f'  border-radius: 4px;'
        f'  padding: 6px 12px;'
        f'  font-weight: bold;'
        f'}}'
        f'QLabel#accent_banner_text {{'
        f'  color: {accent};'
        f'  background: transparent;'
        f'  font-weight: bold;'
        f'}}'
        f'QPushButton#accent_exit_btn {{'
        f'  background-color: {accent};'
        f'  color: #1a1a1a;'
        f'  border: none;'
        f'  border-radius: 3px;'
        f'  padding: 4px 12px;'
        f'  font-weight: bold;'
        f'}}'
        f'QPushButton#accent_exit_btn:hover {{'
        f'  background-color: #e0a060;'
        f'}}'
    )

# ── Global stylesheet ────────────────────────────────────────────────────────

WARP_QSS = f"""
QWidget {{
    background-color: {BG};
    color: {FG};
    font-size: 11px;
}}
QMainWindow, QDialog {{
    background-color: {BG};
}}
QGroupBox {{
    color: {FG};
    border: 1px solid {BC};
    border-radius: 3px;
    margin-top: 8px;
    padding-top: 6px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 8px;
    color: {MFG};
}}
QLabel {{
    background-color: transparent;
    color: {FG};
}}
QPushButton, QToolButton {{
    background-color: transparent;
    color: {FG};
    border: 1px solid {BC};
    border-radius: 3px;
    padding: 4px 10px;
}}
QPushButton:hover, QToolButton:hover {{
    border-color: {ACCENT};
}}
QPushButton:disabled, QToolButton:disabled {{
    color: {BC};
    border-color: {LBG};
}}
QPushButton:checked, QToolButton:checked {{
    background-color: {MBG};
    border: 1px solid {ACCENT};
}}
QListWidget {{
    background-color: {MBG};
    color: {FG};
    border: 1px solid {BC};
    border-radius: 2px;
    outline: 0;
}}
QListWidget::item:selected {{
    background-color: {LBG};
}}
QListWidget::item:hover {{
    background-color: {LBG};
}}
QLineEdit {{
    background-color: {MBG};
    color: {FG};
    border: 1px solid {BC};
    border-radius: 2px;
    padding: 2px 5px;
    selection-background-color: {LBG};
}}
QLineEdit:focus {{
    border-color: {ACCENT};
}}
QPlainTextEdit {{
    background-color: {MBG};
    color: {FG};
    border: 1px solid {BC};
    border-radius: 2px;
    padding: 3px;
    selection-background-color: {LBG};
}}
QPlainTextEdit:focus {{
    border-color: {ACCENT};
}}
QComboBox {{
    background-color: {BG};
    color: {FG};
    border: 1px solid {BC};
    border-radius: 2px;
    padding: 2px 5px;
}}
QComboBox:disabled {{
    background-color: {MBG};
    color: {BC};
    border-color: {LBG};
}}
QComboBox QAbstractItemView {{
    background-color: {MBG};
    color: {FG};
    border: 1px solid {BC};
    selection-background-color: {LBG};
    outline: 0;
}}
QComboBox::drop-down {{
    border-style: none;
}}
QComboBox::drop-down:disabled {{
    background-color: {MBG};
}}
QRadioButton {{
    color: {FG};
    spacing: 6px;
}}
QRadioButton::indicator {{
    width: 14px;
    height: 14px;
    border: 1px solid {BC};
    border-radius: 7px;
    background-color: {LBG};
}}
QRadioButton::indicator:checked {{
    background-color: {ACCENT};
    border-color: {ACCENT};
}}
QRadioButton::indicator:hover {{
    border-color: {ACCENT};
}}
QCheckBox {{
    color: {FG};
    spacing: 6px;
}}
QCheckBox::indicator {{
    width: 14px;
    height: 14px;
    border: 1px solid {BC};
    border-radius: 2px;
    background-color: {LBG};
}}
QCheckBox::indicator:checked {{
    background-color: {ACCENT};
    border-color: {ACCENT};
}}
QCheckBox::indicator:hover {{
    border-color: {ACCENT};
}}
QDoubleSpinBox, QSpinBox {{
    background-color: {MBG};
    color: {FG};
    border: 1px solid {BC};
    border-radius: 2px;
    padding: 2px;
}}
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button,
QSpinBox::up-button, QSpinBox::down-button {{
    background-color: {LBG};
    border: none;
    width: 16px;
}}
QScrollArea {{
    background-color: {BG};
    border: none;
}}
QScrollBar:vertical {{
    background: none;
    border: none;
    width: 8px;
    margin: 0;
}}
QScrollBar:horizontal {{
    background: none;
    border: none;
    height: 8px;
    margin: 0;
}}
QScrollBar::handle {{
    background-color: rgba(100,100,100,0.75);
    border-radius: 4px;
    border: none;
}}
QScrollBar::add-page, QScrollBar::sub-page {{
    background: none;
}}
QScrollBar::add-line, QScrollBar::sub-line {{
    height: 0;
    width: 0;
}}
QProgressBar {{
    background-color: {LBG};
    color: {FG};
    border: none;
    border-radius: 3px;
    text-align: center;
}}
QProgressBar::chunk {{
    background-color: {ACCENT};
    border-radius: 3px;
}}
QSplitter::handle {{
    background-color: {LBG};
}}
QTabWidget::pane {{
    border: 1px solid {BC};
    border-radius: 2px;
    top: -1px;
}}
QTabBar::tab {{
    background-color: {MBG};
    color: {MFG};
    border: 1px solid {BC};
    border-bottom: none;
    border-top-left-radius: 3px;
    border-top-right-radius: 3px;
    padding: 4px 12px;
    margin-right: 2px;
}}
QTabBar::tab:hover {{
    color: {FG};
}}
QTabBar::tab:selected {{
    background-color: {TAB_SEL_BG};
    color: {FG};
}}
QTabBar::tab:disabled {{
    color: {BC};
}}
QStatusBar {{
    background-color: {MBG};
    color: {MFG};
    border-top: 1px solid {BC};
}}
QToolBar {{
    background-color: {MBG};
    border-bottom: 1px solid {BC};
    spacing: 4px;
    padding: 2px;
}}
QMenuBar {{
    background-color: {MBG};
    color: {FG};
}}
QMenuBar::item:selected {{
    background-color: {LBG};
}}
QMenu {{
    background-color: {MBG};
    color: {FG};
    border: 1px solid {BC};
    border-radius: 2px;
}}
QMenu::item:selected {{
    background-color: {LBG};
    border: 1px solid {ACCENT};
}}
QFrame[frameShape="4"],
QFrame[frameShape="5"] {{
    color: {BC};
    background-color: {BC};
    border: none;
    max-height: 1px;
}}
QToolTip {{
    background-color: {MBG};
    color: {FG};
    border: 1px solid {LBG};
    padding: 2px;
}}
/* System dialogs (QFileDialog, QMessageBox, QFontDialog, QColorDialog)
   are Qt widget trees parented under our QApplication, so they inherit
   the global QPushButton/QToolButton rules above. That strips icons
   from the nav-toolbar (back/forward/up/new-folder) because our
   border+padding overrides the platform style's icon geometry. Reset
   to palette-driven defaults so the platform style draws those buttons
   natively (including the standard icons). */
QFileDialog QPushButton, QMessageBox QPushButton,
QFontDialog QPushButton, QColorDialog QPushButton {{
    background-color: palette(button);
    color: palette(button-text);
    border: 1px solid palette(mid);
    border-radius: 2px;
    padding: 3px 8px;
}}
QFileDialog QPushButton:hover, QMessageBox QPushButton:hover,
QFontDialog QPushButton:hover, QColorDialog QPushButton:hover {{
    background-color: palette(midlight);
}}
QFileDialog QPushButton:disabled, QMessageBox QPushButton:disabled,
QFontDialog QPushButton:disabled, QColorDialog QPushButton:disabled {{
    color: palette(mid);
}}
QFileDialog QToolButton, QMessageBox QToolButton,
QFontDialog QToolButton, QColorDialog QToolButton {{
    background-color: transparent;
    border: none;
    padding: 2px;
}}
QFileDialog QToolButton:hover, QMessageBox QToolButton:hover,
QFontDialog QToolButton:hover, QColorDialog QToolButton:hover {{
    background-color: palette(midlight);
}}
"""


def apply_dark_style(target) -> None:
    """Apply the active theme's palette + QSS.

    `target` may be a QWidget (just that widget + its children) or a
    QApplication (every widget that gets constructed afterwards). Both
    classes expose `setPalette` / `setStyleSheet`, so the call works
    via duck typing — call once on QApplication at startup to skin the
    whole app, or per-window if you need to skin a popup that lives
    outside the app's stylesheet."""
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window,          QColor(BG))
    palette.setColor(QPalette.ColorRole.WindowText,      QColor(FG))
    palette.setColor(QPalette.ColorRole.Base,            QColor(MBG))
    palette.setColor(QPalette.ColorRole.AlternateBase,   QColor(LBG))
    palette.setColor(QPalette.ColorRole.Text,            QColor(FG))
    palette.setColor(QPalette.ColorRole.BrightText,      QColor(FG))
    palette.setColor(QPalette.ColorRole.ButtonText,      QColor(FG))
    palette.setColor(QPalette.ColorRole.Button,          QColor(MBG))
    palette.setColor(QPalette.ColorRole.Highlight,       QColor(LBG))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(FG))
    palette.setColor(QPalette.ColorRole.ToolTipBase,     QColor(MBG))
    palette.setColor(QPalette.ColorRole.ToolTipText,     QColor(FG))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(MFG))
    target.setPalette(palette)
    target.setStyleSheet(WARP_QSS)
