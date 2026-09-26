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


def _store(tmp_path, entries):
    """A store as the trainer writes it: real PNGs plus crops/crop_index.json.
    `entries` is [(filename, slot, name, state, pixel_value)]."""
    import numpy as np
    cv2 = pytest.importorskip('cv2')
    (tmp_path / 'crops').mkdir()
    index = {}
    for fname, slot, name, state, value in entries:
        img = np.full((40, 30, 3), value, dtype=np.uint8)
        cv2.imwrite(str(tmp_path / 'crops' / fname), img)
        index[fname] = {'slot': slot, 'name': name, 'state': state}
    (tmp_path / 'crops' / 'crop_index.json').write_text(json.dumps(index))
    return tmp_path


def test_a_crop_label_comes_from_the_store_not_the_filename(tmp_path):
    """The filename carries the label the file had when it was written, so a
    correction made later would be invisible exactly where it matters."""
    store = _store(tmp_path, [('boff__old__k-abc123def456.png', 'Boff Tactical',
                               'Corrected Name', 'confirmed', 90)])

    assert list(rec.local_crops(store).values()) == ['Boff Tactical|Corrected Name']


def test_an_auto_accepted_crop_is_not_counted_as_unsent(tmp_path):
    """The uploader keeps it pending until a person confirms it; listing it
    as a transport fault reported 230 deliberate non-sends as faults."""
    store = _store(tmp_path, [('x__y__k-1.png', 'Devices', 'Item', 'pending', 90)])

    assert rec.local_crops(store) == {}


def test_one_picture_counts_once_under_the_label_the_uploader_sends(tmp_path):
    store = _store(tmp_path, [
        ('a__x__k-1.png', 'Boff Temporal', '__inactive__', 'confirmed', 60),
        ('b__x__k-2.png', 'Boff Tactical', '__inactive__', 'confirmed', 60),
    ])

    assert list(rec.local_crops(store).values()) == ['Boff Tactical|__inactive__']


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

