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
