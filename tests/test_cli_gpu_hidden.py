"""The windows keep off the graphics card.

Recognition runs on the CPU, but EasyOCR's recogniser pins host memory, and on
a CUDA build of torch that opens a CUDA context. With a game holding the
card's memory the OCR read failed, returned no text, and the layout fell back
to a two-minute full scan that found no equipment. `sto-warp` now hides the
card before any window starts.

Run standalone:
    python -m pytest tests/test_cli_gpu_hidden.py -v
"""
from __future__ import annotations

import pytest

pytest.importorskip('PySide6')

from warp import cli


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    monkeypatch.setenv('WARP_LOG_DIR', str(tmp_path / 'logs'))
    # setenv first so monkeypatch records the original state and restores it:
    # cli writes os.environ directly, which would otherwise outlive the test.
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'placeholder')
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES')


def _seen_by_window(monkeypatch, module: str, cmd: str | None):
    """Run `sto-warp <cmd>` with the window stubbed; return the variable as
    the window saw it when it started."""
    import importlib
    import os
    mod = importlib.import_module(module)
    seen = {}
    monkeypatch.setattr(mod, 'main', lambda argv=None: seen.setdefault(
        'v', os.environ.get('CUDA_VISIBLE_DEVICES')) and 0 or 0)
    cli.main([cmd] if cmd else [])
    return seen['v']


@pytest.mark.parametrize('module,cmd', [
    ('warp.gui.launcher', None),
    ('warp.gui.launcher', 'launcher'),
    ('warp.gui.warp_window', 'gui'),
])
def test_a_window_starts_with_the_card_hidden(monkeypatch, module, cmd):
    assert _seen_by_window(monkeypatch, module, cmd) == ''


def test_a_value_the_user_set_is_left_alone(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')

    assert _seen_by_window(monkeypatch, 'warp.gui.launcher', 'launcher') == '0'


def test_a_console_command_does_not_touch_it(monkeypatch):
    import os
    cli.main(['check'])

    assert 'CUDA_VISIBLE_DEVICES' not in os.environ
