"""One shared daily budget for everything that POSTs to the backend.

The backend caps **requests** per UTC day — `MAX_REQ_PER_INSTALL` in
`sets-warp-backend/main.py`, plus a second bucket of the same size per IP.
Every counter this replaced measured something else: the trainer counted crops
it queued, the knowledge client counted contributions the server accepted. So
a day of refusals moved neither, and both kept sending — which is how 127
corrected screen types sat unshared while every POST came back 429.

Offline: no network, no Qt loop, budget files under `tmp_path`.
"""
from __future__ import annotations

import io
import json
import types
import urllib.error

import pytest

from warp.backend_budget import (MAX_DAILY_REQUESTS, BackendBudgetExhausted,
                                 DailyBudget)


def _budget(tmp_path, name='b.json'):
    return DailyBudget(tmp_path / name)


def _429(body=b'{"detail":"Rate limit exceeded. Try again tomorrow."}'):
    return urllib.error.HTTPError('http://test/x', 429, 'Too Many Requests',
                                  {}, io.BytesIO(body))


# ── The counter ───────────────────────────────────────────────────────────

def test_nothing_spent_means_nothing_blocked(tmp_path):
    b = _budget(tmp_path)
    assert b.spent == 0
    assert b.blocked_reason() == ''
    b.check()                       # must not raise


def test_requests_are_counted_and_survive_a_restart(tmp_path):
    b = _budget(tmp_path)
    for _ in range(3):
        b.note_request()
    assert _budget(tmp_path).spent == 3


def test_the_cap_is_the_request_count_not_the_item_count(tmp_path):
    b = _budget(tmp_path)
    for _ in range(MAX_DAILY_REQUESTS):
        b.note_request()
    assert str(MAX_DAILY_REQUESTS) in b.blocked_reason()
    with pytest.raises(BackendBudgetExhausted):
        b.check()


def test_a_refusal_blocks_even_far_under_our_own_cap(tmp_path):
    """The server keeps a second bucket per IP, shared with everyone behind
    the same address. This install cannot see it, so its own count can say
    'plenty left' while every request is refused."""
    b = _budget(tmp_path)
    b.note_request()
    b.note_refusal('Rate limit exceeded')
    assert b.remaining() > 0
    assert 'tomorrow' in b.blocked_reason()
    with pytest.raises(BackendBudgetExhausted):
        b.check()


def test_a_refusal_survives_a_restart(tmp_path):
    _budget(tmp_path).note_refusal()
    assert _budget(tmp_path).refused_today()


def test_yesterdays_state_does_not_carry_over(tmp_path):
    p = tmp_path / 'b.json'
    p.write_text(json.dumps({'day': '2000-01-01', 'requests': 9999,
                             'refused_on': '2000-01-01'}))
    b = DailyBudget(p)
    assert b.spent == 0
    assert not b.refused_today()
    assert b.blocked_reason() == ''


def test_an_unreadable_file_reads_as_a_fresh_day(tmp_path):
    p = tmp_path / 'b.json'
    p.write_text('{ not json')
    assert DailyBudget(p).blocked_reason() == ''


# ── The trainer stops the whole run, not one channel ──────────────────────

def _has_pyside() -> bool:
    try:
        import PySide6  # noqa: F401
        return True
    except ImportError:
        return False


pyside = pytest.mark.skipif(not _has_pyside(), reason='PySide6 not installed')


def _worker(tmp_path, monkeypatch):
    from warp.trainer import sync
    w = sync.SyncWorker.__new__(sync.SyncWorker)
    w._mgr = types.SimpleNamespace(_dir=tmp_path)
    w._url = 'http://test'
    w._mode = 'upload'
    w._budget = DailyBudget(tmp_path / 'b.json')
    monkeypatch.setattr(sync, '_get_install_id', lambda: 'testinstall12345')
    monkeypatch.setattr(w, '_fetch_staging_screen_types', lambda d: {})
    return sync, w


@pyside
def test_a_post_counts_against_the_budget(tmp_path, monkeypatch):
    from warp.trainer import sync
    _s, w = _worker(tmp_path, monkeypatch)
    monkeypatch.setattr(sync.urllib.request, 'urlopen',
                        lambda *a, **k: (_ for _ in ()).throw(_429()))
    with pytest.raises(BackendBudgetExhausted):
        w._post('/upload/screen-types', {})
    assert w._budget.spent == 1
    assert w._budget.refused_today()


@pyside
def test_a_refused_request_still_counts(tmp_path, monkeypatch):
    """The server counts a request whether or not it liked the contents, so
    counting only the accepted ones is how a day of refusals stayed invisible."""
    from warp.trainer import sync
    _s, w = _worker(tmp_path, monkeypatch)
    err = urllib.error.HTTPError('http://test/x', 400, 'Bad Request', {},
                                 io.BytesIO(b'nope'))
    monkeypatch.setattr(sync.urllib.request, 'urlopen',
                        lambda *a, **k: (_ for _ in ()).throw(err))
    with pytest.raises(urllib.error.HTTPError):
        w._post('/upload/screen-types', {})
    assert w._budget.spent == 1
    assert not w._budget.refused_today()      # a 400 is about the payload


@pyside
def test_nothing_is_sent_once_the_budget_is_gone(tmp_path, monkeypatch):
    from warp.trainer import sync
    _s, w = _worker(tmp_path, monkeypatch)
    w._budget.note_refusal()
    called: list = []
    monkeypatch.setattr(sync.urllib.request, 'urlopen',
                        lambda *a, **k: called.append(1))
    with pytest.raises(BackendBudgetExhausted):
        w._post('/upload/screen-types', {})
    assert called == []


@pyside
def test_one_refusal_ends_every_channel_for_the_run(tmp_path, monkeypatch):
    """Thirteen screen-type directories, then crops, then anchors: the run
    used to spend about fifteen POSTs learning the same 429, each one counted
    against the very budget that was missing."""
    from warp.trainer import sync
    _s, w = _worker(tmp_path, monkeypatch)
    for i in range(3):
        d = tmp_path / 'screen_types' / f'TYPE_{i}'
        d.mkdir(parents=True)
        (d / 'a.png').write_bytes(b'\x89PNG' + b'\x00' * 200)

    posts: list[str] = []

    def _post(path, payload):
        posts.append(path)
        w._budget.note_request()
        w._budget.note_refusal('Rate limit exceeded')
        raise BackendBudgetExhausted('the backend answered HTTP 429')

    monkeypatch.setattr(w, '_post', _post)
    monkeypatch.setattr(w, '_upload', lambda: None)
    monkeypatch.setattr(w, '_upload_anchors_grid', lambda: None)
    monkeypatch.setattr(w, '_report_upload_backlog', lambda: None)
    monkeypatch.setattr(w, 'finished', types.SimpleNamespace(emit=lambda ok: None))

    w.run()
    assert len(posts) == 1
