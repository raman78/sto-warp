"""A crop file shows its own box on its own screenshot.

`ann_id` is derived from bbox + slot, so the same slot box on two screenshots
shares it — common, the game UI does not move. Crops were named and looked up
by `ann_id` alone, and the startup sweep renamed one screenshot's crop to the
other's label. Measured 2026-09-25 on the maintainer's store: 98 of 7314
confirmed crops showed a picture other than their box, and a Fragment of AI
Tech icon reached the community labelled Unconventional Systems.

Offline: cv2 only to write PNGs.
"""
from __future__ import annotations

import hashlib
import json

import pytest

pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

import cv2

from warp.trainer.training_data import AnnotationState, TrainingDataManager

BOX = (10, 10, 20, 30)
SLOT = 'Personal Space Traits'


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'cfg'))
    monkeypatch.setenv('WARP_LOG_DIR', str(tmp_path / 'logs'))


def _shot(path, value):
    """A screenshot whose BOX holds a flat patch of `value` — its 'icon'."""
    img = np.zeros((80, 80, 3), dtype=np.uint8)
    img[5:75, 5:75] = (value, 255 - value, 90)
    img[0, 0] = value            # distinct bytes → distinct content hash
    cv2.imwrite(str(path), img)
    return path


def _labelled(mgr, name) -> list:
    """Every crop the uploader would send under `name` — read through
    `get_confirmed_crops`, the uploader's own view, so the check holds for
    any naming scheme."""
    return [cv2.imread(c['path']) for c in mgr.get_confirmed_crops()
            if c.get('name') == name]


def _shows_only(mgr, name, shot) -> bool:
    crops = _labelled(mgr, name)
    return bool(crops) and all(c is not None and np.array_equal(c, _box(shot)) for c in crops)


def _box(shot) -> np.ndarray:
    x, y, w, h = BOX
    return cv2.imread(str(shot))[y:y + h, x:x + w]


@pytest.fixture
def two(tmp_path):
    a = _shot(tmp_path / 'a.png', 40)
    b = _shot(tmp_path / 'b.png', 200)
    mgr = TrainingDataManager(tmp_path / 'store')
    for shot, name in ((a, 'Fragment of AI Tech'), (b, 'Unconventional Systems')):
        mgr.add_annotation(image_path=shot, bbox=BOX, slot=SLOT, name=name,
                           state=AnnotationState.CONFIRMED)
    mgr.save()
    return mgr, a, b


def test_the_same_box_on_two_screenshots_keeps_two_crops(two):
    mgr, a, b = two

    assert _shows_only(mgr, 'Fragment of AI Tech', a)
    assert _shows_only(mgr, 'Unconventional Systems', b)


def test_restarting_does_not_move_a_crop_between_screenshots(two, tmp_path):
    """The startup sweep is where the reported crop changed hands."""
    TrainingDataManager(tmp_path / 'store')
    mgr = TrainingDataManager(tmp_path / 'store')
    _, a, b = two

    assert _shows_only(mgr, 'Fragment of AI Tech', a)
    assert _shows_only(mgr, 'Unconventional Systems', b)


def test_removing_one_screenshots_annotation_keeps_the_others_crop(two):
    mgr, a, b = two
    mgr.remove_annotation(a, mgr.get_annotations(a)[0])

    assert _shows_only(mgr, 'Unconventional Systems', b)


# ── Stores written with the old names ──────────────────────────────────────

def _legacy_store(tmp_path, *, screenshots_in_store: bool):
    """Two screenshots, same box, one legacy crop file holding the wrong one's
    picture — the state the old sweep left behind."""
    a = _shot(tmp_path / 'a.png', 40)
    b = _shot(tmp_path / 'b.png', 200)
    store = tmp_path / 'store'
    mgr = TrainingDataManager(store)
    for shot, name in ((a, 'Fragment of AI Tech'), (b, 'Unconventional Systems')):
        mgr.add_annotation(image_path=shot, bbox=BOX, slot=SLOT, name=name,
                           state=AnnotationState.CONFIRMED)
    mgr.save()
    # Rewrite the crops the way the old code named them.
    for f in list((store / 'crops').glob('*.png')):
        f.unlink()
    ann_id = mgr.get_annotations(a)[0].ann_id
    legacy = f'personal_space_traits__unconventional_systems__{ann_id}.png'
    cv2.imwrite(str(store / 'crops' / legacy), _box(a))      # a's picture, b's label
    index = {legacy: {'slot': SLOT, 'name': 'Unconventional Systems',
                      'state': 'confirmed', 'source': 'b.png'}}
    (store / 'crops' / 'crop_index.json').write_text(json.dumps(index))
    data = json.loads((store / 'annotations.json').read_text())
    for rec in data.values():
        for d in rec['annotations']:
            d['crop_name'] = f'crops/{legacy}'
    (store / 'annotations.json').write_text(json.dumps(data))
    if screenshots_in_store:
        for shot in (a, b):
            dst = store / 'screen_types' / 'SPACE_TRAITS' / shot.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(shot.read_bytes())
    return store, a, b


def test_an_old_store_is_recut_from_its_screenshots(tmp_path):
    store, a, b = _legacy_store(tmp_path, screenshots_in_store=True)
    mgr = TrainingDataManager(store)

    assert _shows_only(mgr, 'Fragment of AI Tech', a)
    assert _shows_only(mgr, 'Unconventional Systems', b)


def test_an_old_crop_that_could_be_either_screenshots_is_dropped(tmp_path):
    """No screenshot to cut from, and the id is shared: the file may show
    either picture, so neither annotation keeps it."""
    store, a, b = _legacy_store(tmp_path, screenshots_in_store=False)
    mgr = TrainingDataManager(store)

    assert not _labelled(mgr, 'Fragment of AI Tech')
    assert not _labelled(mgr, 'Unconventional Systems')
    assert not list((store / 'crops').glob('*.png'))
