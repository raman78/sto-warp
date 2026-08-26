"""One refused batch must not hold back the ones behind it.

`_upload_anchors_grid` used to `break` out of the batch loop on any HTTP
error. The backend validates each grid's `build_type` against a whitelist, so
a grid it will never accept sat at the head of the queue and stopped every
grid behind it — on every sync, for good. A 4xx is a verdict on those grids;
5xx and network failures are the server's problem and still stop the pass.
"""
from __future__ import annotations

import json
import types
import urllib.error

import pytest


def _has_pyside() -> bool:
    try:
        import PySide6  # noqa: F401
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(not _has_pyside(), reason='PySide6 not installed')


def _make_worker(tmp_path, monkeypatch, n_grids: int):
    """A SyncWorker with `n_grids` learned layouts waiting to go out."""
    from warp.trainer import sync
    w = sync.SyncWorker.__new__(sync.SyncWorker)
    w._mgr = types.SimpleNamespace(_dir=tmp_path)
    w._url = 'http://test'
    monkeypatch.setattr(sync, '_get_install_id', lambda: 'testinstall12345')
    # One grid per batch, so batch boundaries are easy to reason about.
    monkeypatch.setattr(sync, 'BULK_ANCHORS_BATCH', 1)

    learned = [{
        'type':   f'TYPE_{i}',
        'aspect': 1.777,
        'res':    '1920x1080',
        'slots':  {f'Slot {j}': {'y_rel': 0.1 * j} for j in range(3)},
    } for i in range(n_grids)]
    (tmp_path / 'anchors.json').write_text(json.dumps({'learned': learned}))
    return sync, w


def _http_error(code: int) -> urllib.error.HTTPError:
    import io
    return urllib.error.HTTPError(
        'http://test', code, 'nope', {},
        io.BytesIO(b'{"detail":"All 1 grids rejected"}'))


def test_a_refused_batch_does_not_stop_the_next_one(tmp_path, monkeypatch):
    sync, w = _make_worker(tmp_path, monkeypatch, n_grids=3)
    seen: list[str] = []

    def _post(path, payload):
        build_type = payload['grids'][0]['build_type']
        seen.append(build_type)
        if build_type == 'TYPE_0':
            raise _http_error(400)
        return {'accepted': 1}

    monkeypatch.setattr(w, '_post', _post)
    w._upload_anchors_grid()

    assert seen == ['TYPE_0', 'TYPE_1', 'TYPE_2']


def test_a_refused_grid_is_offered_again_next_run(tmp_path, monkeypatch):
    """The verdict is the backend's whitelist, which changes without us."""
    sync, w = _make_worker(tmp_path, monkeypatch, n_grids=2)

    def _post(path, payload):
        if payload['grids'][0]['build_type'] == 'TYPE_0':
            raise _http_error(400)
        return {'accepted': 1}

    monkeypatch.setattr(w, '_post', _post)
    w._upload_anchors_grid()

    cached = json.loads((tmp_path / '.sync_uploaded_grids.json').read_text())
    assert len(cached) == 1, 'only the accepted grid may be remembered'

    seen: list[str] = []
    monkeypatch.setattr(w, '_post',
                        lambda p, payload: seen.append(
                            payload['grids'][0]['build_type']) or {'accepted': 1})
    w._upload_anchors_grid()

    assert seen == ['TYPE_0']


def test_a_server_error_still_stops_the_pass(tmp_path, monkeypatch):
    """A 503 says nothing about the data — retrying the rest just hammers."""
    sync, w = _make_worker(tmp_path, monkeypatch, n_grids=3)
    seen: list[str] = []

    def _post(path, payload):
        seen.append(payload['grids'][0]['build_type'])
        raise _http_error(503)

    monkeypatch.setattr(w, '_post', _post)
    w._upload_anchors_grid()

    assert seen == ['TYPE_0']


def test_a_network_failure_still_stops_the_pass(tmp_path, monkeypatch):
    sync, w = _make_worker(tmp_path, monkeypatch, n_grids=3)
    seen: list[str] = []

    def _post(path, payload):
        seen.append(payload['grids'][0]['build_type'])
        raise OSError('connection reset')

    monkeypatch.setattr(w, '_post', _post)
    w._upload_anchors_grid()

    assert seen == ['TYPE_0']
