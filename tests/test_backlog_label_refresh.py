"""The "not yet shared" count is recomputed when a sync cycle ends.

Only two things call `_refresh_upload_backlog`: the trainer's constructor and
its own five-minute timer. That timer is armed only when the window is NOT
embedded, because under the launcher the coordinator owns the cycle — so in
the launcher the number was taken once as the window was built and never
again. It read 129 for days while uploads ran every hour, which looked exactly
like uploads being refused, and was not.

The launcher now recounts on `busy_changed(False)`, which the coordinator
emits from `_on_finished`, after the upload step has had its bounded wait.

Offline: no Qt event loop, no network — the handler is called directly.
"""
from __future__ import annotations

import types

import pytest

pytest.importorskip('PySide6')

from warp.gui.launcher import LauncherWindow


def _launcher(calls):
    w = LauncherWindow.__new__(LauncherWindow)
    w._core_win = types.SimpleNamespace(
        _refresh_upload_backlog=lambda: calls.append('counted'))
    return w


def test_the_end_of_a_cycle_recounts():
    calls: list[str] = []
    _launcher(calls)._on_sync_busy_for_backlog(False)
    assert calls == ['counted']


def test_the_start_of_a_cycle_does_not():
    """Mid-cycle the store is still being written; the number would be wrong
    and would then sit there looking authoritative."""
    calls: list[str] = []
    _launcher(calls)._on_sync_busy_for_backlog(True)
    assert calls == []


def test_a_trainer_that_is_not_ready_does_not_break_the_launcher():
    w = LauncherWindow.__new__(LauncherWindow)
    w._core_win = None
    w._on_sync_busy_for_backlog(False)        # must not raise


def test_a_failing_recount_does_not_break_the_launcher():
    def _boom():
        raise RuntimeError('store unreadable')

    w = LauncherWindow.__new__(LauncherWindow)
    w._core_win = types.SimpleNamespace(_refresh_upload_backlog=_boom)
    w._on_sync_busy_for_backlog(False)        # must not raise


def test_the_embedded_trainer_still_has_no_timer_of_its_own():
    """Two refreshes per cycle is what the embed gate exists to prevent; this
    fix must not undo it."""
    import inspect

    from warp.trainer.trainer_window import WarpCoreWindow
    src = inspect.getsource(WarpCoreWindow.__init__)
    assert 'if not self._embed:' in src
    assert '_sync_timer' in src
