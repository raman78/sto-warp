"""Tests for the screen-type upload oversized-PNG guard in warp/trainer/sync.py.

The backend caps each screen-type PNG at `MAX_SCREEN_PNG_B64` base64 chars
(main.py _ScreenTypeItem.png_b64 max_length). One oversized item makes the
backend 422 the ENTIRE batch, so the client must drop it before batching.
These tests lock that behaviour without a live network or Qt event loop.
"""
from __future__ import annotations

import base64
import json
import types

import pytest


def _has_pyside() -> bool:
    try:
        import PySide6  # noqa: F401
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(not _has_pyside(), reason='PySide6 not installed')


def _make_worker(tmp_path, monkeypatch):
    """A SyncWorker built without QThread.__init__ (no Qt loop needed)."""
    from warp.trainer import sync
    w = sync.SyncWorker.__new__(sync.SyncWorker)
    w._mgr = types.SimpleNamespace(_dir=tmp_path)
    w._url = 'http://test'
    monkeypatch.setattr(sync, '_get_install_id', lambda: 'testinstall12345')
    # No network: pretend nothing is on HF yet.
    monkeypatch.setattr(w, '_fetch_staging_screen_hashes', lambda d: set())
    return sync, w


def test_oversized_png_excluded_but_normal_still_sent(tmp_path, monkeypatch):
    sync, w = _make_worker(tmp_path, monkeypatch)

    sdir = tmp_path / 'screen_types' / 'space_build'
    sdir.mkdir(parents=True)
    # Oversized: raw byte count chosen so its base64 length exceeds the cap.
    big = sdir / 'big.png'
    big.write_bytes(b'\x89PNG' + b'\x00' * sync.MAX_SCREEN_PNG_B64)
    small = sdir / 'small.png'
    small.write_bytes(b'\x89PNG' + b'\x00' * 200)

    # Sanity: the big file really is over the cap once base64-encoded.
    assert len(base64.b64encode(big.read_bytes())) > sync.MAX_SCREEN_PNG_B64

    posted: list[dict] = []
    monkeypatch.setattr(
        w, '_post',
        lambda path, payload: posted.append(payload) or {'accepted': len(payload['items'])})

    w._upload_screen_types()

    # Exactly one POST, carrying only the small screenshot; big never batched.
    assert len(posted) == 1
    assert len(posted[0]['items']) == 1


def test_oversized_png_cached_so_it_is_not_retried(tmp_path, monkeypatch):
    sync, w = _make_worker(tmp_path, monkeypatch)

    sdir = tmp_path / 'screen_types' / 'ground_build'
    sdir.mkdir(parents=True)
    big = sdir / 'huge.png'
    big.write_bytes(b'\x89PNG' + b'\x00' * sync.MAX_SCREEN_PNG_B64)

    monkeypatch.setattr(w, '_post',
                        lambda path, payload: {'accepted': len(payload['items'])})

    w._upload_screen_types()

    # No POST happened (only an oversized file), but its sha is persisted to
    # the hash cache so the next sync tick skips it without re-reading it.
    cache = tmp_path / '.sync_uploaded_screen_hashes.json'
    assert cache.exists()
    cached = set(json.loads(cache.read_text()))
    assert sync.SyncWorker._file_sha256(big) in cached


def test_batch_split_by_cumulative_size(tmp_path, monkeypatch):
    """Screens are split into POSTs bounded by cumulative base64 size, not
    just item count, so a few large originals never build a request body the
    backend/ingress would reject."""
    sync, w = _make_worker(tmp_path, monkeypatch)

    # Tiny budget so size (not the 20-item count cap) drives the split.
    monkeypatch.setattr(sync, 'MAX_SCREEN_BATCH_B64', 600)

    sdir = tmp_path / 'screen_types' / 'space_build'
    sdir.mkdir(parents=True)
    # 5 distinct ~200-byte PNGs → ~272 b64 chars each; 2 fit per 600-char
    # budget, so 5 items must split across 3 POSTs.
    for i in range(5):
        (sdir / f's{i}.png').write_bytes(b'\x89PNG' + bytes([i]) * 200)

    posted: list[dict] = []
    monkeypatch.setattr(
        w, '_post',
        lambda path, payload: posted.append(payload) or {'accepted': len(payload['items'])})

    w._upload_screen_types()

    # More than one POST (count cap alone would have sent all 5 in one).
    assert len(posted) > 1
    # Every item delivered exactly once, and no POST exceeds the budget.
    assert sum(len(p['items']) for p in posted) == 5
    for p in posted:
        total = sum(len(it['png_b64']) for it in p['items'])
        assert len(p['items']) == 1 or total <= sync.MAX_SCREEN_BATCH_B64
