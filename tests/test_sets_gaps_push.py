"""The client sends its SETS-gap ledger once a day, and sends names only.

Two properties matter. The push must be capped to one a day, or a user who
restarts WARP twenty times posts twenty times. And it must never carry a crop,
a screenshot or a build — the maintainer only needs to know which item names
SETS refuses and in how many installs.

Run standalone:
    python -m pytest tests/test_sets_gaps_push.py -v
"""
from __future__ import annotations

import json

import pytest


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    monkeypatch.setenv('XDG_DATA_HOME', str(tmp_path / 'data'))
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'config'))
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path / 'cache'))
    monkeypatch.setenv('WARP_LOG_DIR', str(tmp_path / 'logs'))


@pytest.fixture
def pusher(monkeypatch):
    """A sync client with only the pieces `_push_sets_gaps_bg` touches.

    `WARPSyncClient.__init__` starts network threads, so it is bypassed —
    the method under test reads three attributes and nothing else.
    """
    from warp.knowledge.sync_client import WARPSyncClient

    sent: list[dict] = []

    class _Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return json.dumps({'ok': True, 'accepted': 1}).encode()

    def _urlopen(req, timeout=None):
        sent.append({'url': req.full_url,
                     'body': json.loads(req.data.decode('utf-8'))})
        return _Resp()

    import urllib.request
    monkeypatch.setattr(urllib.request, 'urlopen', _urlopen)

    client = WARPSyncClient.__new__(WARPSyncClient)
    client._url = 'http://backend.invalid'
    client._install_id = 'test-install-1234'
    client.sent = sent
    return client


def _record_a_gap():
    from warp import upstream_gaps
    from warp.sets_schema import Violation

    upstream_gaps.record([
        Violation('/ground/ground_weapons[0]', 'not_in_sets',
                  'an item SETS can resolve', "'Colony Rifle' (scraped)"),
    ])


def test_the_ledger_is_sent_once(pusher):
    _record_a_gap()

    pusher._push_sets_gaps_bg()

    assert len(pusher.sent) == 1
    assert pusher.sent[0]['url'].endswith('/upload/sets-gaps')
    assert pusher.sent[0]['body']['items'][0]['name'] == 'Colony Rifle'


def test_a_second_start_the_same_day_does_not_send_again(pusher):
    """Restarting WARP must not post again — otherwise one enthusiastic user
    looks like traffic."""
    _record_a_gap()

    pusher._push_sets_gaps_bg()
    pusher._push_sets_gaps_bg()

    assert len(pusher.sent) == 1


def test_the_next_day_sends_again(pusher):
    from warp import userdata

    _record_a_gap()
    pusher._push_sets_gaps_bg()
    userdata.sets_gaps_push_stamp_file().write_text(
        json.dumps({'date': '2000-01-01'}))

    pusher._push_sets_gaps_bg()

    assert len(pusher.sent) == 2


def test_an_install_with_nothing_to_report_stays_silent(pusher):
    pusher._push_sets_gaps_bg()

    assert pusher.sent == []


def test_an_emptied_ledger_is_still_reported_once_it_has_been_seen(pusher):
    """That is how an entry expires after the gap is fixed upstream."""
    from warp import upstream_gaps

    _record_a_gap()
    pusher._push_sets_gaps_bg()
    upstream_gaps.ledger_path().unlink()
    from warp import userdata
    userdata.sets_gaps_push_stamp_file().write_text(
        json.dumps({'date': '2000-01-01'}))

    pusher._push_sets_gaps_bg()

    assert pusher.sent[-1]['body']['items'] == []


def test_only_names_reasons_and_slots_leave_the_machine(pusher):
    """No crop, no screenshot, no build — and no local export count, which
    would let one user's repeated exports read as many users."""
    _record_a_gap()

    pusher._push_sets_gaps_bg()

    body = pusher.sent[0]['body']
    assert set(body) == {'install_id', 'items'}
    assert set(body['items'][0]) == {'name', 'reason', 'slots'}


def test_a_failed_send_is_retried_on_the_next_start(pusher, monkeypatch):
    """No stamp is written on failure, so nothing is lost to a backend blip."""
    import urllib.request

    _record_a_gap()

    def _boom(req, timeout=None):
        raise OSError('backend down')
    monkeypatch.setattr(urllib.request, 'urlopen', _boom)
    pusher._push_sets_gaps_bg()

    from warp import userdata
    assert not userdata.sets_gaps_push_stamp_file().exists()
