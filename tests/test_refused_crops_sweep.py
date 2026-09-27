"""Crops that must not exist are not written, and old ones are swept on start.

Two pictures are refused (`_crop_refused`): a `Ship Name`, which shows the
player's ship name and always has an empty label, and a `Ship Tier` whose box
is the class line. Versions before the rule wrote both, indexed as confirmed.
The uploader then refused every `Ship Name` on each sync for its empty label,
and reported each class-line tier as one picture confirmed under two names.

The annotations are kept; only the pictures and their index entries go.

Offline: no OCR, no network. cv2 is needed only to write PNGs.
"""
from __future__ import annotations

import json

import pytest

pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

import cv2

from warp.trainer.training_data import (AnnotationState, TrainingDataManager,
                                        confirmed_crops_in)


NAME_BOX  = (49, 5, 200, 18)
CLASS_BOX = (49, 25, 292, 20)
BADGE_BOX = (280, 27, 60, 17)


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'cfg'))
    img = np.full((200, 400, 3), 40, dtype=np.uint8)
    shot = tmp_path / 'shot.png'
    cv2.imwrite(str(shot), img)
    return tmp_path / 'store', shot


def _add(mgr, shot, slot, name, bbox):
    return mgr.add_annotation(image_path=shot, bbox=bbox, slot=slot, name=name,
                              state=AnnotationState.CONFIRMED)


def _plant_old_crop(mgr, shot, ann):
    """What a version before the rule left behind: a PNG, a confirmed index
    entry, and the annotation pointing at it."""
    key = mgr._image_id(shot)
    fname = mgr._crop_fname(key, ann.slot, ann.name, ann.ann_id)
    cv2.imwrite(str(mgr._dir / 'crops' / fname),
                np.full((20, 60, 3), 90, dtype=np.uint8))
    mgr._crop_index[fname] = {'slot': ann.slot, 'name': ann.name,
                              'state': AnnotationState.CONFIRMED,
                              'source': shot.name}
    for d in mgr._annotations[key]:
        if d['ann_id'] == ann.ann_id:
            d['crop_name'] = f'crops/{fname}'
    mgr.save()
    return fname


def _index(store_dir):
    return json.loads((store_dir / 'crops' / 'crop_index.json').read_text())


# ── New annotations ───────────────────────────────────────────────────────

def test_a_ship_name_gets_no_crop_and_no_index_entry(store):
    store_dir, shot = store
    mgr = TrainingDataManager(store_dir)
    ann = _add(mgr, shot, 'Ship Name', '', NAME_BOX)
    mgr.save()
    assert ann.crop_name == ''
    assert not list((store_dir / 'crops').glob('ship_name__*.png'))
    assert not any(m['slot'] == 'Ship Name' for m in _index(store_dir).values())


def test_the_ship_name_annotation_itself_survives(store):
    """Its box is what layout learning uses."""
    store_dir, shot = store
    mgr = TrainingDataManager(store_dir)
    _add(mgr, shot, 'Ship Name', '', NAME_BOX)
    names = [a for a in mgr.get_annotations(shot) if a.slot == 'Ship Name']
    assert [tuple(a.bbox) for a in names] == [NAME_BOX]


# ── The startup sweep ─────────────────────────────────────────────────────

@pytest.fixture
def old_store(store):
    """A store as an older version left it."""
    store_dir, shot = store
    mgr = TrainingDataManager(store_dir)
    name = _add(mgr, shot, 'Ship Name', '', NAME_BOX)
    _add(mgr, shot, 'Ship Type', 'Verne Temporal Science Vessel', CLASS_BOX)
    tier = _add(mgr, shot, 'Ship Tier', 'T6-X2', CLASS_BOX)
    planted = [_plant_old_crop(mgr, shot, name), _plant_old_crop(mgr, shot, tier)]
    for f in planted:
        assert (store_dir / 'crops' / f).exists()
    return store_dir, shot, planted


def test_the_sweep_removes_old_pictures_and_entries(old_store):
    store_dir, _shot, planted = old_store
    TrainingDataManager(store_dir)
    idx = _index(store_dir)
    for f in planted:
        assert not (store_dir / 'crops' / f).exists()
        assert f not in idx


def test_the_sweep_keeps_the_annotations_and_clears_their_crop_name(old_store):
    store_dir, shot, _planted = old_store
    mgr = TrainingDataManager(store_dir)
    by_slot = {a.slot: a for a in mgr.get_annotations(shot)}
    assert by_slot['Ship Tier'].name == 'T6-X2'
    assert tuple(by_slot['Ship Name'].bbox) == NAME_BOX
    assert by_slot['Ship Tier'].crop_name == ''
    assert by_slot['Ship Name'].crop_name == ''


def test_the_class_line_crop_of_ship_type_is_kept(old_store):
    store_dir, _shot, _planted = old_store
    TrainingDataManager(store_dir)
    assert any(m['slot'] == 'Ship Type' for m in _index(store_dir).values())
    assert list((store_dir / 'crops').glob('ship_type__*.png'))


def test_nothing_refused_reaches_the_uploader(old_store):
    store_dir, _shot, _planted = old_store
    TrainingDataManager(store_dir)
    sent = {c['slot'] for c in confirmed_crops_in(store_dir, _index(store_dir))}
    assert sent == {'Ship Type'}


def test_a_second_start_finds_nothing_to_sweep(old_store):
    """The index repair must not put back what the sweep removed, or every
    start would remove it again."""
    store_dir, _shot, _planted = old_store
    TrainingDataManager(store_dir)
    mgr = TrainingDataManager(store_dir)
    assert mgr.sweep_refused_crops() == (0, 0)
    assert not any(m['slot'] in ('Ship Name', 'Ship Tier')
                   for m in _index(store_dir).values())


def test_the_sweep_says_what_it_removed(old_store, monkeypatch):
    store_dir, _shot, _planted = old_store
    lines: list[str] = []
    import warp.debug
    monkeypatch.setattr(warp.debug.log, 'info', lambda msg, *a, **k: lines.append(msg))
    TrainingDataManager(store_dir)
    text = '\n'.join(lines)
    assert '1 Ship Name crops' in text and '1 Ship Tier crops' in text
    assert 'shot.png' in text and 'tier badge' in text


def test_a_tier_with_its_own_box_is_not_swept(store):
    store_dir, shot = store
    mgr = TrainingDataManager(store_dir)
    _add(mgr, shot, 'Ship Type', 'Verne Temporal Science Vessel', CLASS_BOX)
    tier = _add(mgr, shot, 'Ship Tier', 'T6-X2', BADGE_BOX)
    mgr.save()
    TrainingDataManager(store_dir)
    assert (store_dir / tier.crop_name).exists()
    assert any(m['slot'] == 'Ship Tier' for m in _index(store_dir).values())
