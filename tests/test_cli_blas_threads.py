"""The windows keep numpy's OpenBLAS to one thread.

Every icon match runs the embedder in torch and then a numpy product against
the gallery. The OpenBLAS threads left spinning after that product starved
the next convolution: 47.9 ms per match instead of 9.1, and a full scan of
one screenshot took 74 s instead of 11. `sto-warp` now sets
`OPENBLAS_NUM_THREADS=1` before any window starts.

Run standalone:
    python -m pytest tests/test_cli_blas_threads.py -v
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

pytest.importorskip('PySide6')

from warp import cli


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    monkeypatch.setenv('WARP_LOG_DIR', str(tmp_path / 'logs'))
    # setenv first so monkeypatch records the original state and restores it:
    # cli writes os.environ directly, which would otherwise outlive the test.
    for name in ('OPENBLAS_NUM_THREADS', 'CUDA_VISIBLE_DEVICES'):
        monkeypatch.setenv(name, 'placeholder')
        monkeypatch.delenv(name)


def _seen_by_window(monkeypatch, module: str, cmd: str | None):
    """Run `sto-warp <cmd>` with the window stubbed; return the variable as
    the window saw it when it started."""
    import importlib
    mod = importlib.import_module(module)
    seen = {}
    monkeypatch.setattr(mod, 'main', lambda argv=None: seen.setdefault(
        'v', os.environ.get('OPENBLAS_NUM_THREADS')) and 0 or 0)
    cli.main([cmd] if cmd else [])
    return seen['v']


@pytest.mark.parametrize('module,cmd', [
    ('warp.gui.launcher', None),
    ('warp.gui.launcher', 'launcher'),
    ('warp.gui.warp_window', 'gui'),
])
def test_a_window_starts_with_one_blas_thread(monkeypatch, module, cmd):
    assert _seen_by_window(monkeypatch, module, cmd) == '1'


def test_a_value_the_user_set_is_left_alone(monkeypatch):
    monkeypatch.setenv('OPENBLAS_NUM_THREADS', '4')

    assert _seen_by_window(monkeypatch, 'warp.gui.launcher', 'launcher') == '4'


def test_a_console_command_does_not_touch_it():
    cli.main(['check'])

    assert 'OPENBLAS_NUM_THREADS' not in os.environ


def test_importing_the_entry_point_does_not_load_numpy(tmp_path):
    # OpenBLAS reads the variable once, when numpy loads it. If importing
    # warp.cli ever pulls numpy in, the setting arrives too late and does
    # nothing — silently. A fresh interpreter is the only clean place to ask.
    env = dict(os.environ, WARP_LOG_DIR=str(tmp_path / 'logs'))
    out = subprocess.run(
        [sys.executable, '-c',
         "import sys, warp.cli; print('numpy' in sys.modules)"],
        capture_output=True, text=True, env=env, check=True)

    assert out.stdout.strip() == 'False'
