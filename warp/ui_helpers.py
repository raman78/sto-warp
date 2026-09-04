# warp/ui_helpers.py
# Shared UI helper utilities for WARP dialogs.

from __future__ import annotations

import time as _time_mod

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QLabel


def window_title(name: str, suffix: str = '') -> str:
    """`"<name> V<version>"`, with an optional trailing `" — <suffix>"`.

    Every window builds its title through here rather than holding a literal,
    because the trainer alone sets its own title in three places — entering
    Fast Correction Mode and leaving it again — and a version pasted into some
    of them disappears the moment the user switches modes.

    Matches the format STO-CLARE uses, so the two programs read alike when
    both are open.

    The number comes from the installed package's build metadata, so it is
    the version that is *running* rather than the version in the working tree
    — an editable install keeps showing the release it was built from until
    it is reinstalled, and a checkout with no metadata at all reports
    `0.0.0+unknown`. Shown as-is either way: a title claiming nothing is
    worse than one admitting the build is old or unknown, and for the case
    this exists for — a user reporting what they are running — the installed
    version is the right answer.
    """
    from warp import __version__

    base = f'{name} V{__version__}'
    return f'{base} — {suffix}' if suffix else base


def time_spent_counter(
    parent,
    prefix: str = 'Time: ',
    style: str = 'color:#bbbbbb;font-size:10px;',
) -> tuple[QLabel, QTimer]:
    """
    Create a self-updating elapsed-time label and QTimer pair.

    Returns (label, timer).
      - Call timer.start(1000) to begin counting from now.
      - Call timer.stop()  to freeze the display.

    Customise appearance for the whole app by changing the defaults here:
      prefix — text before the M:SS value  (e.g. 'Time: ', 'Elapsed: ')
      style  — Qt stylesheet string applied to the label
    """
    _start = [_time_mod.monotonic()]

    lbl = QLabel(f'{prefix}0:00', parent)
    lbl.setStyleSheet(style)

    def _tick():
        elapsed = int(_time_mod.monotonic() - _start[0])
        m, s = divmod(elapsed, 60)
        lbl.setText(f'{prefix}{m}:{s:02d}')

    timer = QTimer(parent)
    timer.timeout.connect(_tick)
    return lbl, timer
