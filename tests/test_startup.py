"""Startup integration test — runs ``sto-warp check`` in a subprocess.

Verifies that the console entry point imports the recognition pipeline
cleanly and exits 0.  The subprocess inherits QT_QPA_PLATFORM=offscreen
so no display server is required.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest


def _has_recognition_deps() -> bool:
    try:
        import cv2, easyocr, torch  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(
    not _has_recognition_deps(),
    reason='recognition dependencies (opencv, easyocr, torch) not installed',
)
def test_check_command_exits_zero():
    """``sto-warp check`` must import foundation modules and print OK."""
    env = os.environ.copy()
    env['QT_QPA_PLATFORM'] = 'offscreen'
    result = subprocess.run(
        [sys.executable, '-m', 'warp.cli', 'check'],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f'sto-warp check failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}'
    )
    assert 'OK' in result.stdout


def test_version_flag_prints_version():
    """``sto-warp --version`` must print a version string and exit 0."""
    env = os.environ.copy()
    env['QT_QPA_PLATFORM'] = 'offscreen'
    result = subprocess.run(
        [sys.executable, '-m', 'warp.cli', '--version'],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert 'sto-warp' in result.stdout
