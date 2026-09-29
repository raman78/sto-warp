"""Second icon source: the pictures SETS-Data never carried."""
from __future__ import annotations

from pathlib import Path

import pytest

from warp.data import asset_sync


@pytest.fixture
def mgr(tmp_path):
    return asset_sync.AssetSyncManager(
        images_dir_=tmp_path / 'icons',
        ship_images_dir_=tmp_path / 'ships',
        cache_dir_=tmp_path / 'cache')


def test_an_overlay_icon_lands_in_the_icon_directory(mgr, monkeypatch, tmp_path):
    """`scraped/icons/Jackal+Mastiff.png` is filed the same as `images/…`."""
    entry = {'path': 'scraped/icons/Jackal+Mastiff.png', 'type': 'blob', 'size': 99}
    monkeypatch.setattr(asset_sync, '_fetch_github_tree',
                        lambda s, url=None: [entry] if url else [])
    written = {}

    def _fake_download(self, e, local_path, session):
        written[e['path']] = local_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(b'x' * 99)
        return True, 1

    monkeypatch.setattr(asset_sync.AssetSyncManager, '_download_one', _fake_download)

    report = mgr.run()

    assert report['updated'] == 1
    assert written['scraped/icons/Jackal+Mastiff.png'] == tmp_path / 'icons' / 'Jackal+Mastiff.png'


def test_overlay_downloads_come_from_our_own_mirror(mgr, monkeypatch):
    entry = {'path': 'scraped/icons/Jackal+Mastiff.png', 'type': 'blob', 'size': 9}
    monkeypatch.setattr(asset_sync, '_fetch_github_tree',
                        lambda s, url=None: [entry] if url else [])
    urls = []

    class _Resp:
        ok = True
        status_code = 200
        content = b'0123456789'

    monkeypatch.setattr(asset_sync.requests.Session, 'get',
                        lambda self, url, **kw: (urls.append(url), _Resp())[1])

    mgr.run()

    assert urls and urls[-1].startswith(asset_sync.OVERLAY_RAW_BASE)


def test_an_unreachable_overlay_does_not_fail_the_sync(mgr, monkeypatch):
    """It is additive: without it those pictures stay missing, nothing else."""
    main = {'path': 'images/Phaser.png', 'type': 'blob', 'size': 10}
    monkeypatch.setattr(asset_sync, '_fetch_github_tree',
                        lambda s, url=None: None if url else [main])
    monkeypatch.setattr(asset_sync.AssetSyncManager, '_download_one',
                        lambda self, e, p, s: (True, 1))

    report = mgr.run()

    assert report['failed'] == 0
    assert report['updated'] == 1


# ── A name both sources carry ──────────────────────────────────────────────
#
# Measured 2026-09-29: ten `Temporal Operative Kit Module - …` icons were in
# both, grey in SETS-Data and gold in the overlay (gold in-game). Each run
# downloaded SETS-Data's, then the overlay's over it: twenty downloads a
# cycle, and every rewritten file changed the icon index's key.

def _blob_sha(data: bytes) -> str:
    import hashlib
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()


_SETS_ART, _OVERLAY_ART = b'grey old art', b'gold wiki art!'


@pytest.fixture
def both_sources(monkeypatch):
    name = 'Temporal+Operative+Kit+Module+-+Chronoplasty.png'
    sets_entry = {'path': f'images/{name}', 'type': 'blob',
                  'size': len(_SETS_ART), 'sha': _blob_sha(_SETS_ART)}
    overlay_entry = {'path': f'scraped/icons/{name}', 'type': 'blob',
                     'size': len(_OVERLAY_ART), 'sha': _blob_sha(_OVERLAY_ART)}
    overlay_up = {'on': True}
    monkeypatch.setattr(
        asset_sync, '_fetch_github_tree',
        lambda s, url=None: ([overlay_entry] if overlay_up['on'] else None)
        if url else [sets_entry])
    downloads = []

    def _download(self, e, local_path, session):
        downloads.append(e['path'])
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(_OVERLAY_ART if e['path'].startswith('scraped/')
                               else _SETS_ART)
        return True, 1
    monkeypatch.setattr(asset_sync.AssetSyncManager, '_download_one', _download)
    return name, downloads, overlay_up


def test_a_shared_name_keeps_the_overlays_picture(mgr, both_sources, tmp_path):
    name, downloads, _ = both_sources
    mgr.run()
    assert (tmp_path / 'icons' / name).read_bytes() == _OVERLAY_ART
    assert not any(p.startswith('images/') for p in downloads)


def test_a_shared_name_is_not_downloaded_again_on_the_next_run(mgr, both_sources, tmp_path):
    _, downloads, _ = both_sources
    mgr.run()
    downloads.clear()
    for f in (tmp_path / 'cache').glob('*tree_cache.json'):
        f.unlink()                      # a fresh manifest, as after an hour
    mgr.run()
    assert downloads == []


def test_without_the_overlay_sets_data_still_supplies_the_picture(mgr, both_sources, tmp_path):
    name, _, overlay_up = both_sources
    overlay_up['on'] = False
    mgr.run()
    assert (tmp_path / 'icons' / name).read_bytes() == _SETS_ART
