"""Theme registry — color palettes consumed by `warp.style`.

Themes are immutable `Theme` dataclasses. The active theme is chosen at
import time from the `WARP_THEME` environment variable (default
`sets_dark`); `warp.style` reads from `get_active()` to build its QSS,
button helpers, and palette.

Adding a new theme:
    1. Define a `Theme(...)` at module scope with all fields populated.
    2. Register it: `THEMES[my_theme.name] = my_theme`.
    3. Launch with `WARP_THEME=<name>` to activate, or call
       `set_active('<name>')` before any GUI window constructs.

The split between chrome colors (BG / MBG / LBG / ACCENT / FG / MFG / BC)
and semantic colors (C_*) is deliberate: chrome can be re-skinned freely,
but semantic colors must stay distinct from chrome so users can still
read state-at-a-glance (confirmed / error / warning / success).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Theme:
    name: str
    # ── Window chrome ────────────────────────────────────────────────
    BG:     str   # background
    MBG:    str   # medium background (panels, lists, inputs)
    LBG:    str   # light background (selection / hover)
    ACCENT: str   # primary accent (buttons, focus ring, progress chunk)
    FG:     str   # primary text
    MFG:    str   # secondary text / hints
    BC:     str   # borders
    TAB_SEL_BG: str  # selected QTabBar tab background (muted, not accent)
    # ── Semantic state colors ────────────────────────────────────────
    C_CONFIRMED:  str   # green  — confirmed annotation / file done
    C_ERROR:      str   # red    — unmatched / hard error
    C_CONF_HIGH:  str   # soft red — high-confidence pending
    C_CONF_MED:   str   # medium red — medium-confidence pending
    C_WARNING:    str   # amber  — warning / attention
    C_SUCCESS:    str   # green  — training complete
    C_FAILURE:    str   # red    — training failed
    # ── Accent variants ───────────────────────────────────────────────
    # Named overlays consumed by `warp.style.accent_qss(name)`. Each entry
    # is a small dict of overrides; consumers fall back to chrome colors
    # when a key is missing. Apply via `widget.setStyleSheet(accent_qss(...))`
    # so the overlay scopes to that widget subtree only.
    accents: dict = field(default_factory=dict)


SETS_DARK = Theme(
    name='sets_dark',
    BG='#1a1a1a',  MBG='#242424',  LBG='#404040',
    ACCENT='#c59129',
    FG='#eeeeee',  MFG='#bbbbbb',  BC='#888888',
    TAB_SEL_BG='#2d4a6b',  # muted slate blue — visible but not loud
    C_CONFIRMED='#7effc8', C_ERROR='#ff5555',
    C_CONF_HIGH='#ffaaaa', C_CONF_MED='#ff8888',
    C_WARNING='#e8c060',   C_SUCCESS='#7effc8', C_FAILURE='#ff7e7e',
    accents={
        # WARP Fast Correction Mode — warm amber-brown overlay so the user
        # can see at a glance they are in the ephemeral correction tab,
        # not the regular training-data review.
        'fast_correction': {
            'BG':     '#2a1f12',   # warm tint behind the window chrome
            'ACCENT': '#d49050',   # banner / exit-button accent
            'BORDER': '#d49050',
        },
    },
)


THEMES: dict[str, Theme] = {SETS_DARK.name: SETS_DARK}

_active: Theme = THEMES.get(
    os.environ.get('WARP_THEME', '').strip() or SETS_DARK.name,
    SETS_DARK,
)


def get_active() -> Theme:
    return _active


def set_active(name: str) -> None:
    """Switch the active theme. Call BEFORE the first window is built —
    `style.WARP_QSS` and the color constants are evaluated at import
    time and won't refresh after the fact."""
    global _active
    if name not in THEMES:
        raise KeyError(f'unknown theme {name!r}; known: {sorted(THEMES)}')
    _active = THEMES[name]
