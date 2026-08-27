"""Merging two systems' training data into one shared store.

The second system must not overwrite the first. These lock the merge rules
that decide what happens when both sides know about the same screenshot.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from warp.tools import share_training_data as share


def _entry(*rows):
    return {'filename': 'x.png', 'annotations': list(rows)}


def _row(slot='Fore Weapons', state='confirmed', auto=False, name='', bbox=None,
         crop=''):
    return {'slot': slot, 'name': name, 'state': state, 'crop_name': crop,
            'auto_confirmed': auto, 'bbox': bbox or [0, 0, 10, 10]}


# ── annotations.json ───────────────────────────────────────────────────

def test_entries_only_one_side_has_are_kept():
    local = {'aaa': _entry(_row())}
    shared = {'bbb': _entry(_row())}

    merged, clashes = share.merge_annotations(local, shared)

    assert set(merged) == {'aaa', 'bbb'}
    assert clashes == []


def test_rows_the_other_system_has_are_added(tmp_path):
    """Row level, not entry level: neither side's work is thrown away."""
    local = {'aaa': _entry(_row(bbox=[0, 0, 10, 10]),
                           _row('Deflector', bbox=[0, 50, 10, 10]))}
    shared = {'aaa': _entry(_row(bbox=[0, 0, 10, 10]))}

    merged, clashes = share.merge_annotations(local, shared)

    slots = [r['slot'] for r in merged['aaa']['annotations']]
    assert slots == ['Fore Weapons', 'Deflector']
    assert clashes and '+1 row' in clashes[0]


def test_an_overlapping_box_is_not_added_twice():
    """Same icon, boxes drawn a pixel apart on two systems."""
    local = {'aaa': _entry(_row(bbox=[0, 0, 10, 10]))}
    shared = {'aaa': _entry(_row(bbox=[1, 1, 10, 10]))}

    merged, _clashes = share.merge_annotations(local, shared)

    assert len(merged['aaa']['annotations']) == 1


def test_a_single_instance_slot_collides_on_the_slot_alone():
    """The ship class box moves between runs; there is still only one."""
    local = {'aaa': _entry(_row('Ship Type', name='A', bbox=[0, 0, 200, 20]))}
    shared = {'aaa': _entry(_row('Ship Type', name='B', bbox=[0, 0, 300, 30]))}

    merged, clashes = share.merge_annotations(local, shared)

    assert len(merged['aaa']['annotations']) == 1
    assert clashes and "kept 'B'" in clashes[0] and "dropped 'A'" in clashes[0]


def test_the_users_row_beats_one_the_detector_confirmed_for_itself():
    local = {'aaa': _entry(_row('Ship Tier', name='T6-X'))}
    shared = {'aaa': _entry(_row('Ship Tier', name='T6-X2', auto=True))}

    merged, _clashes = share.merge_annotations(local, shared)

    assert merged['aaa']['annotations'][0]['name'] == 'T6-X'


def test_two_confirmed_rows_that_disagree_keep_the_shared_one_and_say_so():
    """This tool cannot know which review was right, and a silent drop is
    unrecoverable."""
    local = {'aaa': _entry(_row('Ship Type', name='Legondary Bortasqu'))}
    shared = {'aaa': _entry(_row('Ship Type', name='Legendary Bortasqu'))}

    merged, clashes = share.merge_annotations(local, shared)

    assert merged['aaa']['annotations'][0]['name'] == 'Legendary Bortasqu'
    assert clashes and 'dropped' in clashes[0]


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


# ── Freshness ──────────────────────────────────────────────────────────

def _store_with_crop(root, name: str, when: float) -> Path:
    """A store root holding one crop file with a chosen mtime."""
    (root / 'crops').mkdir(parents=True, exist_ok=True)
    crop = root / 'crops' / name
    crop.write_bytes(b'png')
    os.utime(crop, (when, when))
    return root


def test_the_later_confirmation_wins_a_disagreement(tmp_path):
    """Nothing in the store is timestamped, but the crop a row points at is:
    writing it is what confirming does. The later answer is the one given
    after whatever correction prompted it."""
    lb, sb = tmp_path / 'l', tmp_path / 's'
    _store_with_crop(lb, 'a.png', 2_000_000)       # local: newer
    _store_with_crop(sb, 'b.png', 1_000_000)       # shared: older
    local = {'k': _entry(_row('Ship Type', name='Corrected', crop='crops/a.png'))}
    shared = {'k': _entry(_row('Ship Type', name='Typo', crop='crops/b.png'))}

    merged, clashes = share.merge_annotations(local, shared, lb, sb)

    assert merged['k']['annotations'][0]['name'] == 'Corrected'
    assert clashes and 'newer' in clashes[0]


def test_an_older_local_row_does_not_displace_the_shared_one(tmp_path):
    lb, sb = tmp_path / 'l', tmp_path / 's'
    _store_with_crop(lb, 'a.png', 1_000_000)
    _store_with_crop(sb, 'b.png', 2_000_000)
    local = {'k': _entry(_row('Ship Type', name='Old', crop='crops/a.png'))}
    shared = {'k': _entry(_row('Ship Type', name='New', crop='crops/b.png'))}

    merged, _clashes = share.merge_annotations(local, shared, lb, sb)

    assert merged['k']['annotations'][0]['name'] == 'New'


def test_without_a_date_on_both_sides_nothing_is_displaced(tmp_path):
    """Guessing which review was right is worse than keeping what is there."""
    lb, sb = tmp_path / 'l', tmp_path / 's'
    local = {'k': _entry(_row('Ship Type', name='A'))}
    shared = {'k': _entry(_row('Ship Type', name='B'))}

    merged, clashes = share.merge_annotations(local, shared, lb, sb)

    assert merged['k']['annotations'][0]['name'] == 'B'
    assert clashes and 'undated' in clashes[0]
