"""Standalone WARP CORE uploads what it has confirmed.

It did not. The five-minute timer refreshed the community knowledge, checked
for a newer model and refreshed the pending counter — and sent nothing, so the
counter reported the same number for ever. Measured on the maintainer's install
2026-09-06: 129 screenshots pending, every one of them a screen type the user
had corrected, and not one `HF Sync` line in any log generation.

The comment explaining the absence said upload was "started at app launch in
warp_button.py". That file is the SETS bridge and does not exist in this
repository — it stayed in sets-warp when this one was split out. Only the
launcher's `SyncCoordinator` still started an upload, so opening the trainer
directly meant confirmations accumulated with nowhere to go.

Offscreen Qt; the sync manager is replaced, so nothing touches the network.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from warp.trainer import trainer_window as tw


@pytest.fixture
def win(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'cfg'))
    monkeypatch.setenv('XDG_DATA_HOME', str(tmp_path / 'data'))
    w = tw.WarpCoreWindow()
    yield w
    w.close()


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication([])


class _Manager:
    """Stands in for SyncManager — records that the upload was asked for."""
    made = 0

    def __init__(self, sets_app, parent=None):
        _Manager.made += 1
        self.sets_app = sets_app
        self.calls = 0

    def check_and_upload(self):
        self.calls += 1


@pytest.fixture
def fake_manager(monkeypatch):
    _Manager.made = 0
    monkeypatch.setattr('warp.trainer.sync.SyncManager', _Manager)
    return _Manager


def test_the_timer_tick_sends(win, fake_manager):
    """The behaviour that was missing entirely."""
    win._on_sync_timer()
    assert win._sync_mgr.calls == 1


def test_the_manager_is_built_once_and_reused(win, fake_manager):
    """It owns a QThread; one per tick would pile them up."""
    win._on_sync_timer()
    win._on_sync_timer()
    win._on_sync_timer()
    assert fake_manager.made == 1
    assert win._sync_mgr.calls == 3


def test_the_manager_gets_the_app_shim_it_looks_the_window_up_through(win, fake_manager):
    """`SyncManager._data_manager` reaches the trainer through
    `sets_app._warp_core_window`, which the constructor sets on the shim."""
    win._upload_now()
    assert win._sync_mgr.sets_app is win._sets
    assert getattr(win._sets, '_warp_core_window', None) is win


def test_a_failing_upload_does_not_take_the_window_down(win, monkeypatch, capfd):
    """But it is not swallowed either — a silent failure here is exactly how a
    backlog sits still for weeks."""
    class _Boom:
        def __init__(self, *a, **kw):
            raise RuntimeError('no network')

    monkeypatch.setattr('warp.trainer.sync.SyncManager', _Boom)
    win._upload_now()                      # must not raise
    assert 'upload failed' in capfd.readouterr().err


def test_the_counter_is_refreshed_after_sending_not_before(win, fake_manager,
                                                            monkeypatch):
    """So the number shown is what is left after this tick."""
    order: list[str] = []
    monkeypatch.setattr(win, '_upload_now',
                        lambda: order.append('upload'))
    monkeypatch.setattr(win, '_refresh_upload_backlog',
                        lambda: order.append('count'))
    win._on_sync_timer()
    assert order == ['upload', 'count']


def test_the_legacy_entry_point_still_sends(win, fake_manager):
    """`_auto_sync` was a no-op with the same stale justification."""
    win._auto_sync()
    assert win._sync_mgr.calls == 1
