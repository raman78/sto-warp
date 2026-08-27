"""Merging two systems' training data into one shared store.

The second system must not overwrite the first. These lock the merge rules
that decide what happens when both sides know about the same screenshot.
"""
from __future__ import annotations

import json

import pytest

from warp.tools import share_training_data as share


def _entry(*rows):
    return {'filename': 'x.png', 'annotations': list(rows)}


def _row(slot='Fore Weapons', state='confirmed', auto=False):
    return {'slot': slot, 'state': state, 'auto_confirmed': auto}


# ── annotations.json ───────────────────────────────────────────────────

def test_entries_only_one_side_has_are_kept():
    local = {'aaa': _entry(_row())}
    shared = {'bbb': _entry(_row())}

    merged, clashes = share.merge_annotations(local, shared)

    assert set(merged) == {'aaa', 'bbb'}
    assert clashes == []


def test_the_richer_entry_wins_a_clash():
    """Two systems reviewing the same screenshot: more of the user's work
    beats less of it."""
    local = {'aaa': _entry(_row(), _row('Deflector'))}
    shared = {'aaa': _entry(_row())}

    merged, clashes = share.merge_annotations(local, shared)

    assert len(merged['aaa']['annotations']) == 2
    assert clashes and 'local kept' in clashes[0]


def test_a_clash_the_shared_side_wins_is_still_reported():
    """Silence would be the wrong answer — the local rows are about to stop
    being read, and the user should know."""
    local = {'aaa': _entry(_row())}
    shared = {'aaa': _entry(_row(), _row('Deflector'))}

    merged, clashes = share.merge_annotations(local, shared)

    assert len(merged['aaa']['annotations']) == 2
    assert clashes and 'shared kept' in clashes[0]


def test_auto_confirmed_rows_do_not_make_an_entry_look_richer():
    """Same rule as the importer: the detector's own answers are not the
    user's work."""
    local = {'aaa': _entry(_row(auto=True), _row(auto=True), _row(auto=True))}
    shared = {'aaa': _entry(_row())}

    merged, _clashes = share.merge_annotations(local, shared)

    assert len(merged['aaa']['annotations']) == 1


# ── anchors.json ───────────────────────────────────────────────────────

def test_learned_layouts_are_unioned_without_duplicates():
    entry = {'type': 'SPACE', 'aspect': 1.77, 'slots': {}}
    other = {'type': 'GROUND', 'aspect': 1.77, 'slots': {}}

    merged = share.merge_anchors({'learned': [entry, other]},
                                 {'learned': [entry]})

    assert merged['learned'] == [entry, other]


# ── crops / screen_types ───────────────────────────────────────────────

def test_a_file_the_shared_side_lacks_is_copied(tmp_path):
    local, shared = tmp_path / 'l', tmp_path / 's'
    (local / 'sub').mkdir(parents=True)
    (local / 'sub' / 'a.png').write_bytes(b'a')
    shared.mkdir()

    assert share.merge_tree(local, shared, apply=True) == 1
    assert (shared / 'sub' / 'a.png').read_bytes() == b'a'


def test_an_existing_file_is_never_overwritten(tmp_path):
    """Crops are content-addressed; a name collision with different bytes
    means something is already wrong, and clobbering hides it."""
    local, shared = tmp_path / 'l', tmp_path / 's'
    local.mkdir(); shared.mkdir()
    (local / 'a.png').write_bytes(b'local')
    (shared / 'a.png').write_bytes(b'shared')

    assert share.merge_tree(local, shared, apply=True) == 0
    assert (shared / 'a.png').read_bytes() == b'shared'


# ── the move itself ────────────────────────────────────────────────────

@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv('XDG_DATA_HOME', str(tmp_path / 'data'))
    from warp import userdata
    local = userdata.training_data_dir()
    (local / 'annotations.json').write_text(json.dumps({'aaa': _entry(_row())}))
    (local / 'crops').mkdir()
    (local / 'crops' / 'c.png').write_bytes(b'crop')
    return local, tmp_path / 'shared'


def test_a_dry_run_changes_nothing(store):
    local, shared = store

    assert share.run(shared, apply=False) == 0
    assert not shared.exists()
    assert not local.is_symlink()


def test_apply_merges_and_leaves_a_symlink(store):
    local, shared = store

    assert share.run(shared, apply=True) == 0
    assert local.is_symlink()
    assert json.loads((shared / 'annotations.json').read_text()).keys() == {'aaa'}
    assert (shared / 'crops' / 'c.png').read_bytes() == b'crop'


def test_the_local_store_is_kept_not_deleted(store):
    """The only copy of work that cannot be recreated."""
    local, shared = store

    share.run(shared, apply=True)

    backups = list(local.parent.glob('training_data.bak-*'))
    assert len(backups) == 1
    assert (backups[0] / 'annotations.json').exists()


def test_running_it_twice_is_harmless(store):
    local, shared = store
    share.run(shared, apply=True)

    assert share.run(shared, apply=True) == 0
    assert local.is_symlink()
