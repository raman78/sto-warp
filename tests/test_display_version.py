"""What a person is shown is what is running, not what was built.

`__version__` is written into the package once, at build time, so an editable
install keeps reporting the release it was installed from however far the
working tree has moved. Measured while writing this: a checkout sitting on
v1.0.36 with local edits announced v1.0.32.dev7 — a number that is worse than
useless in a bug report, because it is confidently wrong.

Run standalone:
    python -m pytest tests/test_display_version.py -v
"""
from __future__ import annotations

import subprocess

import pytest

import warp


@pytest.fixture(autouse=True)
def _clear_cache():
    """The result is cached for the process; each test needs a clean one."""
    warp._display_version = None
    yield
    warp._display_version = None


def _fake_git(monkeypatch, stdout: str, returncode: int = 0):
    def _run(cmd, **kw):
        return subprocess.CompletedProcess(cmd, returncode, stdout, '')
    monkeypatch.setattr(subprocess, 'run', _run)


def test_a_checkout_reports_what_git_says(monkeypatch, tmp_path):
    _fake_git(monkeypatch, 'v1.0.36-1-g3a46caf\n')

    assert warp.display_version() == '1.0.36-1-g3a46caf'


def test_local_edits_are_visible_in_the_version(monkeypatch):
    """The whole point: running modified code says so, with no reinstall."""
    _fake_git(monkeypatch, 'v1.0.36-1-g3a46caf-dirty\n')

    assert warp.display_version().endswith('-dirty')


def test_the_tag_prefix_is_dropped():
    """`git describe` prints `v1.0.36`; the title adds its own V, and
    `sto-warp Vv1.0.36` would be the result of keeping both."""
    import re

    from warp.ui_helpers import window_title
    warp._display_version = '1.0.36'

    assert not re.search(r'Vv', window_title('sto-warp'))


# ── Falling back ───────────────────────────────────────────────────────────

def test_an_installed_copy_reports_the_built_version(monkeypatch):
    """No repository beside the package — a user sees a plain release."""
    monkeypatch.setattr('pathlib.Path.exists', lambda self: False)

    assert warp.display_version() == warp.__version__


def test_a_git_failure_falls_back_rather_than_raising(monkeypatch):
    _fake_git(monkeypatch, '', returncode=128)

    assert warp.display_version() == warp.__version__


def test_git_missing_entirely_does_not_break_a_window(monkeypatch):
    """Version reporting must never be the thing that stops a window opening."""
    def _boom(*a, **k):
        raise FileNotFoundError('git')
    monkeypatch.setattr(subprocess, 'run', _boom)

    assert warp.display_version() == warp.__version__


def test_empty_output_is_not_treated_as_a_version(monkeypatch):
    _fake_git(monkeypatch, '   \n')

    assert warp.display_version() == warp.__version__


# ── Cost ───────────────────────────────────────────────────────────────────

def test_git_is_consulted_once_per_process(monkeypatch):
    """It is read on every window title, and the trainer re-titles itself
    three times; a subprocess per title would be absurd."""
    calls = []

    def _run(cmd, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, 'v1.0.36\n', '')
    monkeypatch.setattr(subprocess, 'run', _run)

    warp.display_version()
    warp.display_version()
    warp.display_version()

    assert len(calls) == 1


def test_the_wire_version_carries_no_git_description():
    """`WARP_VERSION` goes into upload payloads and User-Agent headers, and
    is resolved separately — from the installed distribution's metadata
    rather than from the generated version file, so the two can and do
    disagree. Whether a dirty marker belongs on the wire is a separate
    decision from what a window says, and was not taken; this pins that it
    was not taken by accident."""
    from warp.knowledge import sync_client

    assert '-dirty' not in sync_client.WARP_VERSION
    assert '-g' not in sync_client.WARP_VERSION
