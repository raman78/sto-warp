"""Did this machine's contributions reach the dataset, and where is it outvoted.

The published dataset is the reference. A local store holding a different
label is normally the tally working, not a fault — so the split is drawn on
whether this install ever *sent* that label, which its own upload cache
records. Scoring both alike amounts to arguing the consensus should be
corrected to match one machine.

Run standalone:
    python -m pytest tests/test_reconcile_uploads.py -v
"""
from __future__ import annotations

import json

import pytest

pytest.importorskip('PySide6')          # the tool calls SyncWorker for hashing

from warp.tools import reconcile_uploads as rec


# ── Why they differ, not whether ───────────────────────────────────────────

def test_a_matching_pair_is_not_reported():
    v = rec.compare({'aa': 'BOFFS'}, {'aa': 'BOFFS'}, {'aa': 'BOFFS'})

    assert v == {'unsent': [], 'outvoted': [], 'absent': []}


def test_a_label_never_sent_is_a_transport_fault():
    """The screen-type bug: the correction stayed here, so the dataset never
    had the chance to weigh it."""
    v = rec.compare({'aa': 'SPACE_BOFFS'}, {'aa': 'BOFFS'}, {'aa': 'BOFFS'})

    assert v['unsent'] == [('aa', 'SPACE_BOFFS')]
    assert v['outvoted'] == []


def test_a_label_that_was_sent_and_lost_is_not_a_fault():
    v = rec.compare({'aa': 'SPACE_BOFFS'}, {'aa': 'BOFFS'}, {'aa': 'SPACE_BOFFS'})

    assert v['outvoted'] == [('aa', 'SPACE_BOFFS', 'BOFFS')]
    assert v['unsent'] == []


def test_something_never_sent_and_absent_there_is_unsent():
    v = rec.compare({'aa': 'DISCARD'}, {}, {})

    assert v['unsent'] == [('aa', 'DISCARD')]


def test_something_sent_and_then_dropped_is_not_a_transport_fault():
    """A maintainer rejection reads this way and is legitimate."""
    v = rec.compare({'aa': 'DISCARD'}, {}, {'aa': 'DISCARD'})

    assert v['unsent'] == []
    assert v['outvoted'] == [('aa', 'DISCARD', '<dropped>')]


def test_something_in_the_dataset_and_not_here_is_absent():
    v = rec.compare({}, {'aa': 'BOFFS'}, {})

    assert v['absent'] == ['aa']


# ── Reading this machine ───────────────────────────────────────────────────

def test_the_unclassified_folder_is_not_compared(tmp_path):
    d = tmp_path / 'screen_types' / rec.UNCLASSIFIED
    d.mkdir(parents=True)
    (d / 'a.png').write_bytes(b'\x89PNG' + b'\x00' * 50)

    assert rec.local_screens(tmp_path) == {}


def test_the_hash_comes_from_the_client_not_a_copy(tmp_path):
    """A second definition of "how a sha is truncated" is what turns an empty
    result into 'everything agrees'. 32 asserted literally, so changing the
    client's length fails here rather than silently matching nothing."""
    from warp.trainer.sync import SyncWorker

    d = tmp_path / 'screen_types' / 'BOFFS'
    d.mkdir(parents=True)
    png = d / 'a.png'
    png.write_bytes(b'\x89PNG')

    sha = next(iter(rec.local_screens(tmp_path)))

    assert sha == SyncWorker._file_sha256(png)
    assert len(sha) == 32


def test_a_crop_label_comes_from_the_annotations_not_the_filename(tmp_path):
    """The filename carries the label the file had when it was written, so a
    correction made later would be invisible exactly where it matters."""
    (tmp_path / 'crops').mkdir()
    (tmp_path / 'crops' / 'boff__old__abc123def456.png').write_bytes(
        b'\x89PNG' + b'\x00' * 50)
    (tmp_path / 'annotations.json').write_text(json.dumps({
        'k': {'filename': 's.png', 'annotations': [
            {'ann_id': 'abc123def456', 'slot': 'Boff Tactical',
             'name': 'Corrected Name'}]}}), encoding='utf-8')

    assert list(rec.local_crops(tmp_path).values()) == \
        ['Boff Tactical|Corrected Name']


def test_a_legacy_screen_cache_cannot_support_an_outvoted_claim(tmp_path):
    """It was a bare list of shas and recorded no label."""
    (tmp_path / '.sync_uploaded_screen_hashes.json').write_text(
        json.dumps(['aa']), encoding='utf-8')

    sent = rec.sent_labels(tmp_path, 'screens')

    assert sent == {'aa': ''}
    assert rec.compare({'aa': 'BOFFS'}, {'aa': 'TRAITS'}, sent)['unsent']


def test_a_missing_cache_leaves_everything_unproven(tmp_path):
    """No cache means no claim about what was sent, so a difference falls back
    to the fault reading rather than being excused."""
    assert rec.sent_labels(tmp_path, 'crops') == {}
