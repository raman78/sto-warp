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
    monkeypatch.setattr(w, '_fetch_staging_screen_types', lambda d: {})
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
    # Since 2026-09-05 the cache maps sha -> screen type, so a screenshot
    # re-typed in WARP CORE is recognised as needing a fresh upload. Reading
    # it as a bare set would pass here and silently drop every type.
    cached = json.loads(cache.read_text())
    assert isinstance(cached, dict)
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


# ── The two faults found on 2026-09-05 ─────────────────────────────────────
#
# Both were invisible to every test that existed, because both live in what
# the loop *remembers* rather than in what it computes.


def test_a_retyped_screenshot_is_sent_again(tmp_path, monkeypatch):
    """Changing a screenshot's type in WARP CORE does not change the file, so
    the sha still matches. Until the cache recorded the type, the correction
    could not reach the backend at all — 26 of 27 locally `SPACE_BOFFS`
    screenshots are published as `BOFFS` because of it.
    """
    sync, w = _make_worker(tmp_path, monkeypatch)
    png = tmp_path / 'screen_types' / 'BOFFS' / 'a.png'
    png.parent.mkdir(parents=True)
    png.write_bytes(b'\x89PNG' + b'\x00' * 200)

    posted: list[dict] = []
    monkeypatch.setattr(w, '_post', lambda p_, payload: posted.append(payload)
                        or {'accepted': len(payload['items']), 'rejected': 0})
    w._upload_screen_types()
    assert [p_['screen_type'] for p_ in posted] == ['BOFFS']

    # The user re-types it: same bytes, new folder.
    moved = tmp_path / 'screen_types' / 'SPACE_BOFFS' / 'a.png'
    moved.parent.mkdir(parents=True)
    moved.write_bytes(png.read_bytes())
    png.unlink()

    posted.clear()
    w._upload_screen_types()

    assert [p_['screen_type'] for p_ in posted] == ['SPACE_BOFFS']


def test_an_unchanged_screenshot_is_not_sent_twice(tmp_path, monkeypatch):
    """The saving the cache exists for must survive the fix."""
    sync, w = _make_worker(tmp_path, monkeypatch)
    png = tmp_path / 'screen_types' / 'BOFFS' / 'a.png'
    png.parent.mkdir(parents=True)
    png.write_bytes(b'\x89PNG' + b'\x00' * 200)

    posted: list[dict] = []
    monkeypatch.setattr(w, '_post', lambda p_, payload: posted.append(payload)
                        or {'accepted': len(payload['items']), 'rejected': 0})
    w._upload_screen_types()
    posted.clear()
    w._upload_screen_types()

    assert posted == []


def test_a_refused_batch_is_not_remembered_as_sent(tmp_path, monkeypatch):
    """The backend answers with counts, not identities, so a batch holding a
    refusal is not cached at all — re-sending an accepted screenshot costs a
    request, forgetting a refused one loses it for good. Ten `DISCARD`
    screenshots were lost this way.
    """
    sync, w = _make_worker(tmp_path, monkeypatch)
    d = tmp_path / 'screen_types' / 'DISCARD'
    d.mkdir(parents=True)
    for i in range(2):
        (d / f'{i}.png').write_bytes(b'\x89PNG' + bytes([i]) + b'\x00' * 200)

    monkeypatch.setattr(w, '_post', lambda p_, payload:
                        {'accepted': 0, 'rejected': len(payload['items'])})
    w._upload_screen_types()

    cache = tmp_path / '.sync_uploaded_screen_hashes.json'
    cached = json.loads(cache.read_text()) if cache.exists() else {}
    assert cached == {}


def test_a_refused_batch_is_retried_on_the_next_sync(tmp_path, monkeypatch):
    sync, w = _make_worker(tmp_path, monkeypatch)
    d = tmp_path / 'screen_types' / 'DISCARD'
    d.mkdir(parents=True)
    (d / 'a.png').write_bytes(b'\x89PNG' + b'\x00' * 200)

    calls: list[dict] = []
    monkeypatch.setattr(w, '_post', lambda p_, payload: calls.append(payload)
                        or {'accepted': 0, 'rejected': len(payload['items'])})
    w._upload_screen_types()
    w._upload_screen_types()

    assert len(calls) == 2


def test_the_persisted_cache_keeps_the_type(tmp_path, monkeypatch):
    """Guards the write side: the mapping used to be persisted with
    `sorted(...)`, which on a dict writes the keys and drops every type —
    reintroducing the fault through the back door."""
    sync, w = _make_worker(tmp_path, monkeypatch)
    png = tmp_path / 'screen_types' / 'SPACE_TRAITS' / 'a.png'
    png.parent.mkdir(parents=True)
    png.write_bytes(b'\x89PNG' + b'\x00' * 200)

    monkeypatch.setattr(w, '_post', lambda p_, payload:
                        {'accepted': len(payload['items']), 'rejected': 0})
    w._upload_screen_types()

    cached = json.loads((tmp_path / '.sync_uploaded_screen_hashes.json').read_text())

    assert cached == {sync.SyncWorker._file_sha256(png): 'SPACE_TRAITS'}


# ── Self-diagnosis ─────────────────────────────────────────────────────────
#
# Every upload fault found so far was silent in the same way: the client
# believed it had sent something it had not. One number — confirmed here but
# not sent — would have shown it, so it is computed after every upload pass.


def test_nothing_pending_after_a_clean_upload(tmp_path, monkeypatch):
    sync, w = _make_worker(tmp_path, monkeypatch)
    png = tmp_path / 'screen_types' / 'BOFFS' / 'a.png'
    png.parent.mkdir(parents=True)
    png.write_bytes(b'\x89PNG' + b'\x00' * 200)
    monkeypatch.setattr(w, '_post', lambda p_, payload:
                        {'accepted': len(payload['items']), 'rejected': 0})

    w._upload_screen_types()

    assert w._diagnose_upload_backlog() == {}


def test_a_refused_screenshot_is_reported_as_pending(tmp_path, monkeypatch):
    """The `DISCARD` case: refused at the door, and until now invisible."""
    sync, w = _make_worker(tmp_path, monkeypatch)
    d = tmp_path / 'screen_types' / 'DISCARD'
    d.mkdir(parents=True)
    d.joinpath('a.png').write_bytes(b'\x89PNG' + b'\x00' * 200)
    monkeypatch.setattr(w, '_post', lambda p_, payload:
                        {'accepted': 0, 'rejected': len(payload['items'])})

    w._upload_screen_types()

    assert w._diagnose_upload_backlog() == {'screen_types': 1}


def test_a_retyped_screenshot_counts_as_pending(tmp_path, monkeypatch):
    """The fault that motivated this: the file was sent, its correction was
    not, and no count anywhere said so."""
    sync, w = _make_worker(tmp_path, monkeypatch)
    png = tmp_path / 'screen_types' / 'BOFFS' / 'a.png'
    png.parent.mkdir(parents=True)
    png.write_bytes(b'\x89PNG' + b'\x00' * 200)
    monkeypatch.setattr(w, '_post', lambda p_, payload:
                        {'accepted': len(payload['items']), 'rejected': 0})
    w._upload_screen_types()

    moved = tmp_path / 'screen_types' / 'SPACE_BOFFS' / 'a.png'
    moved.parent.mkdir(parents=True)
    moved.write_bytes(png.read_bytes())
    png.unlink()

    assert w._diagnose_upload_backlog() == {'screen_types': 1}


def test_the_unclassified_folder_is_not_counted_as_pending(tmp_path, monkeypatch):
    """`UNKNOWN` is a sentinel the backend refuses by design, so counting it
    would report a permanent backlog nobody can clear."""
    sync, w = _make_worker(tmp_path, monkeypatch)
    d = tmp_path / 'screen_types' / 'UNKNOWN'
    d.mkdir(parents=True)
    d.joinpath('a.png').write_bytes(b'\x89PNG' + b'\x00' * 200)

    assert w._diagnose_upload_backlog() == {}


def test_the_check_never_raises_when_there_is_no_store(tmp_path, monkeypatch):
    """Diagnosis must not be the thing that breaks a sync."""
    sync, w = _make_worker(tmp_path, monkeypatch)

    assert w._diagnose_upload_backlog() == {}
