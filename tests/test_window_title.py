"""Every window title carries the running version.

The trainer alone sets its own title in three places — on construction, on
entering Fast Correction Mode, and on leaving it — so a version pasted into
some of them disappears the moment the user switches modes. One builder,
called from all of them, is what stops that.

Run standalone:
    python -m pytest tests/test_window_title.py -v
"""
from __future__ import annotations

import pytest

pytest.importorskip('PySide6')

from warp.ui_helpers import window_title


def test_the_version_is_in_the_title():
    """`display_version`, not `__version__`: the title says what is running,
    which for a working tree past its last release is not what was built."""
    from warp import display_version

    assert window_title('sto-warp') == f'sto-warp V{display_version()}'


def test_a_suffix_follows_the_version():
    """The version sits with the program name, not tacked on the end, so it
    stays readable when the window manager truncates a long title."""
    from warp import display_version

    title = window_title('WARP CORE', 'ML Trainer')

    assert title == f'WARP CORE V{display_version()} — ML Trainer'


def test_the_format_matches_the_sibling_program():
    """STO-CLARE renders `STO-CLARE V1.2.3`; the two read alike when both are
    open, which is the point of matching it."""
    assert window_title('X', '').startswith('X V')


def test_no_dash_when_there_is_no_suffix():
    assert '—' not in window_title('sto-warp')


def test_every_main_window_builds_its_title_through_the_helper():
    """The failure this guards is a literal creeping back into one of the
    places the trainer re-titles itself, silently dropping the version there.
    """
    import inspect
    from warp.gui import launcher, warp_window
    from warp.trainer import trainer_window

    for mod in (launcher, warp_window, trainer_window):
        src = inspect.getsource(mod)
        for line in src.splitlines():
            if 'setWindowTitle(' not in line:
                continue
            # Dialogs and message boxes name themselves; only the program's
            # own windows carry the version.
            if line.lstrip().startswith(('box.', 'msg.', 'dlg.')):
                continue
            if 'self.setWindowTitle(' in line:
                assert 'window_title(' in line, f'{mod.__name__}: {line.strip()}'
