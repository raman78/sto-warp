"""Source fallback in `warp.data.cargo`.

The primary mirror is ours; SETS-Data stands behind it. These tests pin the
behaviour that matters: a dead primary must not stop the app, and an ETag must
never be replayed against a server that did not issue it.
"""
from __future__ import annotations

import urllib.error

import pytest

from warp.data import cargo


class _Resp:
    def __init__(self, payload: bytes, etag: str | None):
        self._payload, self.headers = payload, {'ETag': etag}

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def bases(monkeypatch):
    primary, fallback = 'https://primary.test/cargo', 'https://fallback.test/cargo'
    monkeypatch.setattr(cargo, 'UPSTREAM_BASES', (primary, fallback))
    return primary, fallback


def test_primary_is_used_when_healthy(bases, monkeypatch):
    primary, _ = bases
    seen = []

    def fake_urlopen(req, timeout=0):
        seen.append(req.full_url)
        return _Resp(b'[1]', '"e1"')

    monkeypatch.setattr(cargo.urllib.request, 'urlopen', fake_urlopen)
    payload, etag, base = cargo._fetch('traits.json')

    assert payload == b'[1]'
    assert base == primary
    assert seen == [f'{primary}/traits.json']      # fallback never touched


def test_falls_back_when_primary_fails(bases, monkeypatch):
    primary, fallback = bases

    def fake_urlopen(req, timeout=0):
        if req.full_url.startswith(primary):
            raise urllib.error.HTTPError(req.full_url, 404, 'gone', {}, None)
        return _Resp(b'[2]', '"e2"')

    monkeypatch.setattr(cargo.urllib.request, 'urlopen', fake_urlopen)
    payload, _etag, base = cargo._fetch('traits.json')

    assert payload == b'[2]'
    assert base == fallback


def test_etag_only_replayed_against_its_own_source(bases, monkeypatch):
    """A 304 from the wrong server would strand us on the other mirror's bytes."""
    primary, fallback = bases
    headers = {}

    def fake_urlopen(req, timeout=0):
        if req.full_url.startswith(primary):
            headers['primary'] = req.get_header('If-none-match')
            raise urllib.error.URLError('down')
        headers['fallback'] = req.get_header('If-none-match')
        return _Resp(b'[3]', '"e3"')

    monkeypatch.setattr(cargo.urllib.request, 'urlopen', fake_urlopen)
    cargo._fetch('traits.json', etag='"from-primary"', source=primary)

    assert headers['primary'] == '"from-primary"'
    assert headers['fallback'] is None


def test_not_modified_reports_the_source(bases, monkeypatch):
    primary, _ = bases

    def fake_urlopen(req, timeout=0):
        raise urllib.error.HTTPError(req.full_url, 304, 'nm', {}, None)

    monkeypatch.setattr(cargo.urllib.request, 'urlopen', fake_urlopen)
    payload, etag, base = cargo._fetch('traits.json', etag='"e"', source=primary)

    assert payload is None
    assert etag == '"e"'
    assert base == primary


def test_raises_only_when_every_source_fails(bases, monkeypatch):
    def fake_urlopen(req, timeout=0):
        raise urllib.error.URLError('no network')

    monkeypatch.setattr(cargo.urllib.request, 'urlopen', fake_urlopen)
    with pytest.raises(RuntimeError, match='no source served'):
        cargo._fetch('traits.json')


# --- content validation --------------------------------------------------

def test_corrupt_body_falls_through_to_the_next_source(bases, monkeypatch):
    """A 200 carrying an error page must not be cached as if it were data."""
    primary, fallback = bases

    def fake_urlopen(req, timeout=0):
        if req.full_url.startswith(primary):
            return _Resp(b'<html>504 Gateway Timeout</html>', '"e"')
        return _Resp(b'[{"name": "real"}]', '"e2"')

    monkeypatch.setattr(cargo.urllib.request, 'urlopen', fake_urlopen)
    payload, _etag, base = cargo._fetch('traits.json')

    assert base == fallback
    assert b'real' in payload


def test_empty_document_is_rejected(bases, monkeypatch):
    primary, fallback = bases

    def fake_urlopen(req, timeout=0):
        if req.full_url.startswith(primary):
            return _Resp(b'[]', '"e"')
        return _Resp(b'[{"name": "real"}]', '"e2"')

    monkeypatch.setattr(cargo.urllib.request, 'urlopen', fake_urlopen)
    _payload, _etag, base = cargo._fetch('traits.json')
    assert base == fallback


def test_poisoned_cache_is_discarded_and_refetched(tmp_path, monkeypatch):
    """A bad cache file must not wedge the app permanently."""
    monkeypatch.setenv('WARP_CACHE_DIR', str(tmp_path))
    monkeypatch.setattr(cargo, '_MEMO', {})
    cached = tmp_path / 'traits.json'
    cached.write_text('{ truncated', encoding='utf-8')

    monkeypatch.setattr(cargo, '_fetch',
                        lambda name, **kw: (b'[{"name": "fresh"}]', '"e"', 'src'))
    assert cargo._load_raw('traits.json') == [{'name': 'fresh'}]
    assert not cached.exists() or cached.read_text(encoding='utf-8') != '{ truncated'


# --- the ground-weapon overlay ---------------------------------------------
#
# The publisher drops every row the real cargo tables start carrying, so an
# empty overlay is its designed end state (reached 2026-09-19), not a broken
# download. And a wiki regression can refill it later, which only reaches
# installs that keep refreshing it.

_OVERLAY = 'scraped_ground_weapons.json'


def test_an_empty_overlay_is_accepted(monkeypatch):
    monkeypatch.setattr(cargo.urllib.request, 'urlopen',
                        lambda req, timeout=0: _Resp(b'[]', '"e"'))

    payload, _etag, base = cargo._fetch(_OVERLAY)

    assert payload == b'[]'
    assert base == cargo.OVERLAY_BASE


def test_an_empty_overlay_is_cached_rather_than_refetched(tmp_path, monkeypatch):
    """Rejected, it was never cached: every start fetched it again and
    logged a failure that had not happened."""
    monkeypatch.setenv('WARP_CACHE_DIR', str(tmp_path))
    monkeypatch.setattr(cargo.urllib.request, 'urlopen',
                        lambda req, timeout=0: _Resp(b'[]', '"e"'))

    cargo._resolve_raw(_OVERLAY)

    assert (tmp_path / _OVERLAY).read_bytes() == b'[]'


def test_refresh_replaces_a_stale_overlay(tmp_path, monkeypatch):
    """The overlay was fetched once and then kept for good; an install from
    2026-08-22 still held the 136-row version a month later."""
    monkeypatch.setenv('WARP_CACHE_DIR', str(tmp_path))
    monkeypatch.setattr(cargo, '_MEMO', {})
    monkeypatch.setattr(cargo, 'validate', lambda **kw: None)
    (tmp_path / _OVERLAY).write_bytes(b'[{"name": "stale row"}]')
    monkeypatch.setattr(cargo, '_fetch',
                        lambda name, **kw: (b'[]', '"new"', cargo.OVERLAY_BASE)
                        if name == _OVERLAY else (None, None, 'src'))

    cargo.refresh_all()

    assert (tmp_path / _OVERLAY).read_bytes() == b'[]'
