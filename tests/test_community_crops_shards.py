"""The community mirror reads a sharded dataset and stays flat on disk.

Upstream had to shard `data/crops/`: HF refuses a push that would leave more
than 10 000 files in one directory, and the folder filled up. The mirror does
not have to follow. A crop's filename is its content sha, so a flat directory
cannot collide, and every reader on this side — the k-NN seed, the review
tools, the mirror scan — already globs it flat.

So downloads land wherever the repo path says and are flattened once, rather
than teaching each reader two layouts.

Run standalone:
    python -m pytest tests/test_community_crops_shards.py -v
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    monkeypatch.setenv('WARP_LOG_DIR', str(tmp_path / 'logs'))
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path / 'cache'))
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'config'))
    monkeypatch.setenv('XDG_DATA_HOME', str(tmp_path / 'data'))


def _client():
    from warp.knowledge.community_crops import CommunityCropsClient
    return CommunityCropsClient()


def _crops_dir():
    from warp.knowledge.community_crops import community_crops_dir
    d = community_crops_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── Flattening ─────────────────────────────────────────────────────────────

def test_a_sharded_download_is_flattened():
    crops = _crops_dir()
    (crops / 'ab').mkdir()
    (crops / 'ab' / 'ab99.png').write_bytes(b'\x89PNG')

    assert _client()._flatten_shards() == 1
    assert (crops / 'ab99.png').exists()


def test_the_empty_shard_directory_is_removed():
    """Left behind, they would accumulate one per shard on every sync."""
    crops = _crops_dir()
    (crops / 'ab').mkdir()
    (crops / 'ab' / 'ab99.png').write_bytes(b'\x89PNG')

    _client()._flatten_shards()

    assert not (crops / 'ab').exists()


def test_a_crop_already_flat_is_not_overwritten():
    """Same sha, same bytes — the sharded copy is redundant, and replacing
    the flat one would churn the mirror on every sync."""
    crops = _crops_dir()
    (crops / 'ab99.png').write_bytes(b'flat')
    (crops / 'ab').mkdir()
    (crops / 'ab' / 'ab99.png').write_bytes(b'sharded')

    _client()._flatten_shards()

    assert (crops / 'ab99.png').read_bytes() == b'flat'
    assert not (crops / 'ab').exists()


def test_a_flat_mirror_is_left_alone():
    crops = _crops_dir()
    (crops / 'ab99.png').write_bytes(b'\x89PNG')

    assert _client()._flatten_shards() == 0
    assert (crops / 'ab99.png').exists()


def test_hidden_directories_are_not_touched():
    """`.trash` holds soft-deleted crops; flattening it would resurrect them."""
    crops = _crops_dir()
    (crops / '.keep').mkdir()
    (crops / '.keep' / 'x.png').write_bytes(b'\x89PNG')

    _client()._flatten_shards()

    assert (crops / '.keep' / 'x.png').exists()


# ── What the sync looks for upstream ───────────────────────────────────────

def test_both_layouts_are_allowed_in_a_snapshot():
    from warp.knowledge import community_crops as cc

    assert 'data/crops/*.png' in cc._ALLOW_PATTERNS
    assert 'data/crops/*/*.png' in cc._ALLOW_PATTERNS


# ── The wiring, not just the helper ────────────────────────────────────────

def test_the_delta_sync_flattens_what_it_downloaded(monkeypatch):
    """Testing `_flatten_shards` alone would pass with the call site deleted,
    and the call site is the part that can vanish unnoticed."""
    from warp.knowledge import community_crops as cc

    crops = _crops_dir()

    class _Api:
        def list_repo_files(self, **k):
            return ['data/annotations.jsonl', 'data/crops/ab/ab99.png']

    def _download(repo_id, repo_type, filename, revision, local_dir):
        dst = cc.community_root() / filename
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b'\x89PNG')
        return str(dst)

    # A non-empty mirror, so the cold-start tarball path is not taken.
    (crops / 'seed.png').write_bytes(b'\x89PNG')

    assert _client()._delta_sync(_Api(), _download, 'rev') is True
    assert (crops / 'ab99.png').exists(), 'downloaded crop was left in its shard'
    assert not (crops / 'ab').exists()
