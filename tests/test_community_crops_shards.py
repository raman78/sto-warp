"""The community mirror shards its crops, the same way the dataset does.

`data/crops/` upstream hit HF's cap of 10 000 files per directory and had to
shard. The mirror follows for its own reasons: it held 12 274 files after one
merge and grows every week, every sync globs it several times, and the client
ships on Windows too. The shard is the first two characters of the filename,
which is the crop's content sha — derivable, so there is no index to keep in
step.

Installs that predate this have their crops flat. Nothing may assume the
migration has already happened: `_shard_local` runs on every sync and every
reader accepts either layout until it has.

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


# ── The rule ───────────────────────────────────────────────────────────────

def test_a_crop_lives_under_the_first_two_characters_of_its_name():
    from warp.knowledge.community_crops import (community_crops_dir,
                                                mirror_crop_path)

    assert mirror_crop_path('ab99.png') == community_crops_dir() / 'ab' / 'ab99.png'


def test_the_shard_matches_what_the_dataset_uses():
    """Both sides derive the path from the sha with the same rule, so a crop
    is findable on either without a lookup table."""
    from warp.knowledge.community_crops import mirror_crop_path

    sha = 'ab12cd34ef56'
    assert mirror_crop_path(f'{sha}.png').parent.name == sha[:2]


# ── Migrating an install that predates it ──────────────────────────────────

def test_a_flat_mirror_is_filed_into_shards():
    crops = _crops_dir()
    (crops / 'ab99.png').write_bytes(b'\x89PNG')

    assert _client()._shard_local() == 1
    assert (crops / 'ab' / 'ab99.png').exists()
    assert not (crops / 'ab99.png').exists()


def test_a_crop_already_in_its_shard_is_left_alone():
    crops = _crops_dir()
    (crops / 'ab').mkdir()
    (crops / 'ab' / 'ab99.png').write_bytes(b'\x89PNG')

    assert _client()._shard_local() == 0
    assert (crops / 'ab' / 'ab99.png').exists()


def test_a_duplicate_is_dropped_rather_than_overwriting_the_shard():
    """Same name means same sha means same bytes, so the flat copy is
    redundant — and replacing the sharded one would churn the mirror."""
    crops = _crops_dir()
    (crops / 'ab').mkdir()
    (crops / 'ab' / 'ab99.png').write_bytes(b'sharded')
    (crops / 'ab99.png').write_bytes(b'flat')

    _client()._shard_local()

    assert (crops / 'ab' / 'ab99.png').read_bytes() == b'sharded'
    assert not (crops / 'ab99.png').exists()


# ── Readers must not assume the migration has run ──────────────────────────

def test_a_not_yet_migrated_mirror_is_still_counted():
    from warp.knowledge.community_crops import CommunityCropsClient

    crops = _crops_dir()
    (crops / 'flat.png').write_bytes(b'\x89PNG')
    (crops / 'ab').mkdir()
    (crops / 'ab' / 'ab99.png').write_bytes(b'\x89PNG')

    snap = CommunityCropsClient()._scan(ok=True)

    assert snap.crops == 2


def test_the_path_guard_allows_a_shard_but_nothing_further_out(tmp_path):
    """The guard is what stops a crafted name from writing outside the
    mirror; adding a level must not have opened it up."""
    from warp.knowledge.community_crops import (_assert_inside_mirror_crops,
                                                community_crops_dir)

    _assert_inside_mirror_crops(community_crops_dir() / 'ab' / 'ab99.png')
    _assert_inside_mirror_crops(community_crops_dir() / 'ab99.png')
    with pytest.raises(RuntimeError):
        _assert_inside_mirror_crops(community_crops_dir().parent / 'elsewhere.png')


# ── What the sync looks for upstream ───────────────────────────────────────

def test_both_upstream_layouts_are_allowed_in_a_snapshot():
    from warp.knowledge import community_crops as cc

    assert 'data/crops/*.png' in cc._ALLOW_PATTERNS
    assert 'data/crops/*/*.png' in cc._ALLOW_PATTERNS


# ── The wiring, not just the helper ────────────────────────────────────────

def test_the_delta_sync_shards_what_it_downloaded(monkeypatch):
    """Testing `_shard_local` alone would pass with the call site deleted,
    and the call site is the part that can vanish unnoticed."""
    from warp.knowledge import community_crops as cc

    crops = _crops_dir()

    class _Api:
        def list_repo_files(self, **k):
            return ['data/annotations.jsonl', 'data/crops/cd99.png']

    def _download(repo_id, repo_type, filename, revision, local_dir):
        dst = cc.community_root() / filename
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b'\x89PNG')
        return str(dst)

    (crops / 'ab' ).mkdir()
    (crops / 'ab' / 'ab99.png').write_bytes(b'\x89PNG')   # non-empty: no cold start

    assert _client()._delta_sync(_Api(), _download, 'rev') is True
    assert (crops / 'cd' / 'cd99.png').exists(), 'downloaded crop was left flat'
