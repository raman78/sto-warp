"""A screenshot has one screen type, and the folder says so too.

`screen_types.json` holds one label per screenshot and is right.
`screen_types/<TYPE>/<filename>` was only ever written to, because
`set_screen_type` copied instead of moving — so every type a screenshot had
ever been given kept a file. The first of those is usually not the user's
answer but the classifier's guess: auto-classification writes a copy before
anybody looks at it.

Measured on the maintainer's store 2026-09-11: 534 files against 287 labels,
110 screenshots filed under two or three mutually exclusive types at once.
The uploader walks those folders, so both labels went up; the upload cache is
keyed on the content hash alone and cannot hold two, so the copies overwrote
each other's entry and were re-sent every cycle for ever; and the backend
takes one vote per (install, sha) from whichever copy its file walk reaches
first, so the vote was an arbitrary pick between the user's correction and
the guess they had corrected.

Offline: a training store under `tmp_path`, no image decoding and no network.
"""
from __future__ import annotations

import json

import pytest

from warp.tools.reconcile_screen_types import reconcile
from warp.trainer.training_data import TrainingDataManager

PNG = b'\x89PNG\r\n\x1a\n' + b'screenshot bytes'
OTHER = b'\x89PNG\r\n\x1a\n' + b'a different screenshot'


@pytest.fixture
def store(tmp_path):
    return TrainingDataManager(tmp_path / 'training_data')


def _shot(tmp_path, name='shot.png', data=PNG):
    p = tmp_path / name
    p.write_bytes(data)
    return p


def _files(store) -> dict[str, list[str]]:
    root = store._dir / 'screen_types'
    return {d.name: sorted(f.name for f in d.glob('*.png'))
            for d in sorted(root.iterdir()) if d.is_dir()}


# ── Setting a type ────────────────────────────────────────────────────────

def test_the_screenshot_lands_in_the_type_folder(store, tmp_path):
    store.set_screen_type(_shot(tmp_path), 'BOFFS')
    assert _files(store) == {'BOFFS': ['shot.png']}


def test_relabelling_moves_it_rather_than_leaving_a_second_copy(store, tmp_path):
    """The defect this exists to stop: the classifier guesses `SPACE_BOFFS`,
    the user corrects it to `BOFFS`, and both files are uploaded for ever."""
    shot = _shot(tmp_path)
    store.set_screen_type(shot, 'SPACE_BOFFS', user_confirmed=False)
    store.set_screen_type(shot, 'BOFFS', user_confirmed=True)
    assert _files(store) == {'BOFFS': ['shot.png'], 'SPACE_BOFFS': []}


def test_the_label_and_the_folder_agree_after_several_changes(store, tmp_path):
    shot = _shot(tmp_path)
    for stype in ('TRAITS', 'GROUND_MIXED', 'SPACE_TRAITS', 'SPACE_MIXED'):
        store.set_screen_type(shot, stype)
    placed = [t for t, names in _files(store).items() if names]
    assert placed == ['SPACE_MIXED']
    assert store.get_screen_type(shot) == 'SPACE_MIXED'


def test_a_different_screenshot_sharing_a_filename_is_not_touched(store, tmp_path):
    """A filename is not an identity. Deleting on the name alone would lose
    training data outright."""
    (tmp_path / 'a').mkdir()
    (tmp_path / 'b').mkdir()
    mine = _shot(tmp_path / 'a', 'shot.png', data=PNG)
    theirs = _shot(tmp_path / 'b', 'shot.png', data=OTHER)

    store.set_screen_type(theirs, 'SPACE_BOFFS')
    store.set_screen_type(mine, 'BOFFS')
    assert _files(store) == {'BOFFS': ['shot.png'],
                             'SPACE_BOFFS': ['shot.png']}


def test_setting_the_same_type_twice_keeps_the_file(store, tmp_path):
    shot = _shot(tmp_path)
    store.set_screen_type(shot, 'BOFFS')
    store.set_screen_type(shot, 'BOFFS')
    assert _files(store) == {'BOFFS': ['shot.png']}


# ── The source is already the training copy ───────────────────────────────
#
# Reachable whenever a caller hands over a path inside the store — pointing
# WARP CORE at the training folder itself is enough. `shutil.copy2` raises
# `SameFileError` rather than doing nothing, and the caller is a click
# handler, so the exception would surface as a dead button.

def test_confirming_the_training_copy_itself_does_not_raise(store, tmp_path):
    shot = _shot(tmp_path)
    placed = store.set_screen_type(shot, 'BOFFS')
    assert store.set_screen_type(placed, 'BOFFS') == placed
    assert _files(store) == {'BOFFS': ['shot.png']}
    assert placed.read_bytes() == PNG


def test_the_stale_sweep_still_runs_when_there_is_nothing_to_copy(store,
                                                                 tmp_path):
    """The early return is only for the copy. This screenshot may still sit
    under other types, and those copies are what the sweep exists to remove."""
    shot = _shot(tmp_path)
    store.set_screen_type(shot, 'SPACE_BOFFS')
    placed = store.set_screen_type(shot, 'BOFFS')

    stale = store._dir / 'screen_types' / 'SPACE_BOFFS' / 'shot.png'
    stale.write_bytes(PNG)                    # it came back, e.g. from a sync
    store.set_screen_type(placed, 'BOFFS')    # source *is* the destination
    assert not stale.exists()
    assert _files(store) == {'BOFFS': ['shot.png'], 'SPACE_BOFFS': []}


# ── Reconciling what is already on disk ───────────────────────────────────

def _stale_store(store, tmp_path):
    """A store in the broken state: labelled one way, filed two ways."""
    shot = _shot(tmp_path)
    store.set_screen_type(shot, 'BOFFS')
    stale = store._dir / 'screen_types' / 'SPACE_BOFFS'
    stale.mkdir(parents=True, exist_ok=True)
    (stale / 'shot.png').write_bytes(PNG)
    return shot


def test_the_dry_run_reports_without_deleting(store, tmp_path, capsys):
    _stale_store(store, tmp_path)
    assert reconcile(store._dir, apply=False) == 1
    assert (store._dir / 'screen_types' / 'SPACE_BOFFS' / 'shot.png').exists()
    assert 'dry run' in capsys.readouterr().out


def test_apply_removes_the_copy_the_label_contradicts(store, tmp_path, capsys):
    _stale_store(store, tmp_path)
    reconcile(store._dir, apply=True)
    assert _files(store) == {'BOFFS': ['shot.png'], 'SPACE_BOFFS': []}


def test_a_file_with_no_label_is_left_alone(store, tmp_path, capsys):
    """Deleting on a guess is how training data disappears."""
    root = store._dir / 'screen_types'
    for stype in ('BOFFS', 'SPACE_BOFFS'):
        (root / stype).mkdir(parents=True, exist_ok=True)
        (root / stype / 'orphan.png').write_bytes(OTHER)
    reconcile(store._dir, apply=True)
    assert _files(store)['BOFFS'] == ['orphan.png']
    assert _files(store)['SPACE_BOFFS'] == ['orphan.png']


def test_the_upload_cache_forgets_a_type_the_label_contradicts(store, tmp_path):
    """Left in place it would keep the screenshot looking already-shared under
    the wrong label, which is how the count stayed frozen."""
    _stale_store(store, tmp_path)
    sha = store._image_id(tmp_path / 'shot.png')
    cache = store._dir / '.sync_uploaded_screen_hashes.json'
    cache.write_text(json.dumps({sha: 'SPACE_BOFFS'}))
    reconcile(store._dir, apply=True)
    assert json.loads(cache.read_text()) == {}


def test_a_cache_entry_that_agrees_is_kept(store, tmp_path):
    _stale_store(store, tmp_path)
    sha = store._image_id(tmp_path / 'shot.png')
    cache = store._dir / '.sync_uploaded_screen_hashes.json'
    cache.write_text(json.dumps({sha: 'BOFFS'}))
    reconcile(store._dir, apply=True)
    assert json.loads(cache.read_text()) == {sha: 'BOFFS'}


# ── One file per screenshot per type ──────────────────────────────────────
#
# The folder holds two generations: `set_screen_type` used to save a 224x224
# thumbnail and now copies the screenshot whole. The classifier resizes every
# input with a plain `cv2.resize` to 224x224, so where both survive they are
# the *same tensor* — measured over all 74 such pairs in the maintainer's
# store, mean pixel difference 0.00 on every one.

import numpy as np
import pytest

cv2 = pytest.importorskip('cv2')


def _write(path, w, h):
    rng = np.random.default_rng(seed=w * h)
    img = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
    return cv2.imread(str(path))


def test_a_thumbnail_of_a_file_already_there_is_swept(store, tmp_path):
    d = store._dir / 'screen_types' / 'SPACE_EQ'
    whole = _write(d / 'shot.png', 640, 360)
    cv2.imwrite(str(d / 'shot_deadbeef.png'),
                cv2.resize(whole, (224, 224), interpolation=cv2.INTER_AREA))
    reconcile(store._dir, apply=True)
    assert _files(store)['SPACE_EQ'] == ['shot.png']


def test_a_thumbnail_with_no_twin_is_kept(store, tmp_path):
    """It is the only copy of that screenshot, and worth exactly what a
    full-size one would be — the classifier resizes to 224 either way."""
    d = store._dir / 'screen_types' / 'SPACE_EQ'
    _write(d / 'lonely_deadbeef.png', 224, 224)
    reconcile(store._dir, apply=True)
    assert _files(store)['SPACE_EQ'] == ['lonely_deadbeef.png']


def test_a_different_picture_with_a_tag_shaped_name_is_kept(store, tmp_path):
    """The name is how the pair is found; the pixels decide."""
    d = store._dir / 'screen_types' / 'SPACE_EQ'
    _write(d / 'shot.png', 640, 360)
    _write(d / 'shot_deadbeef.png', 224, 224)      # unrelated content
    reconcile(store._dir, apply=True)
    assert _files(store)['SPACE_EQ'] == ['shot.png', 'shot_deadbeef.png']


def test_a_full_size_file_is_never_taken_for_a_thumbnail(store, tmp_path):
    d = store._dir / 'screen_types' / 'SPACE_EQ'
    _write(d / 'shot.png', 640, 360)
    _write(d / 'shot_deadbeef.png', 640, 360)
    reconcile(store._dir, apply=True)
    assert _files(store)['SPACE_EQ'] == ['shot.png', 'shot_deadbeef.png']
