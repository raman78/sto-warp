"""A `Ship Tier` crop that is really the class line is refused, and said aloud.

The badge now gets a box of its own (`test_badge_box_split`), so this is the
safety net rather than the common path: a screenshot where the badge cannot be
separated still produces a tier row whose rectangle is the ship's class line.
Writing that crop would put a picture of a ship's name into the dataset under a
tier's label, and — because identical pixels give an identical hash — merge the
tier and class ballots on the server.

The annotation is kept. Only the picture is refused, and the refusal is logged
with what to do about it, because a rejection nobody hears about teaches
nothing.

Offline: no OCR, no network. cv2 is needed only to write a PNG.
"""
from __future__ import annotations

import pytest

pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

import cv2

from warp.trainer.training_data import (AnnotationState, TrainingDataManager)


CLASS_BOX = (49, 25, 292, 20)
BADGE_BOX = (280, 27, 60, 17)


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'cfg'))
    img = np.full((200, 400, 3), 40, dtype=np.uint8)
    shot = tmp_path / 'shot.png'
    cv2.imwrite(str(shot), img)
    return TrainingDataManager(tmp_path / 'store'), shot


def _add(mgr, shot, slot, name, bbox):
    return mgr.add_annotation(image_path=shot, bbox=bbox, slot=slot, name=name,
                              state=AnnotationState.CONFIRMED)


def test_a_tier_sharing_the_class_box_gets_no_crop(store):
    mgr, shot = store
    _add(mgr, shot, 'Ship Type', 'Verne Temporal Science Vessel', CLASS_BOX)
    ann = _add(mgr, shot, 'Ship Tier', 'T6-X2', CLASS_BOX)
    assert ann.crop_name == ''
    assert not list((mgr._dir / 'crops').glob('ship_tier__*.png'))


def test_the_annotation_itself_survives(store):
    """The tier is still known, still in the build, still reviewable — only
    its picture is refused."""
    mgr, shot = store
    _add(mgr, shot, 'Ship Type', 'Verne Temporal Science Vessel', CLASS_BOX)
    _add(mgr, shot, 'Ship Tier', 'T6-X2', CLASS_BOX)
    tiers = [a for a in mgr.get_annotations(shot) if a.slot == 'Ship Tier']
    assert [a.name for a in tiers] == ['T6-X2']


def test_a_tier_with_its_own_box_is_cropped_normally(store):
    mgr, shot = store
    _add(mgr, shot, 'Ship Type', 'Verne Temporal Science Vessel', CLASS_BOX)
    ann = _add(mgr, shot, 'Ship Tier', 'T6-X2', BADGE_BOX)
    assert ann.crop_name.startswith('crops/ship_tier__')
    assert (mgr._dir / ann.crop_name).exists()


def test_the_class_line_is_cropped_as_before(store):
    """Only the tier is refused. `Ship Type` labelled with the class name is
    a picture of exactly what its label says."""
    mgr, shot = store
    ann = _add(mgr, shot, 'Ship Type', 'Verne Temporal Science Vessel', CLASS_BOX)
    assert ann.crop_name.startswith('crops/ship_type__')
    assert (mgr._dir / ann.crop_name).exists()


def test_a_tier_alone_on_the_image_is_cropped(store):
    """No `Ship Type` to collide with — nothing to refuse."""
    mgr, shot = store
    ann = _add(mgr, shot, 'Ship Tier', 'T6-X2', CLASS_BOX)
    assert ann.crop_name.startswith('crops/ship_tier__')


def test_the_refusal_is_logged_with_what_to_do(store, caplog):
    mgr, shot = store
    _add(mgr, shot, 'Ship Type', 'Verne Temporal Science Vessel', CLASS_BOX)
    with caplog.at_level('INFO'):
        _add(mgr, shot, 'Ship Tier', 'T6-X2', CLASS_BOX)
    text = caplog.text
    assert 'Ship Tier' in text and 'class line' in text
    assert 'Draw a box' in text


# ── The shared-box test itself ────────────────────────────────────────────

def test_a_box_of_its_own_is_not_called_shared(store):
    mgr, shot = store
    _add(mgr, shot, 'Ship Type', 'Verne Temporal Science Vessel', CLASS_BOX)
    assert mgr.tier_box_is_the_class_line(shot, BADGE_BOX) is False


def test_a_box_off_by_a_pixel_is_still_called_shared(store):
    """Rounding must not let the collision through."""
    mgr, shot = store
    _add(mgr, shot, 'Ship Type', 'Verne Temporal Science Vessel', CLASS_BOX)
    x, y, w, h = CLASS_BOX
    assert mgr.tier_box_is_the_class_line(shot, (x + 1, y, w, h)) is True


def test_no_class_row_means_nothing_to_share_with(store):
    mgr, _shot = store
    assert mgr.tier_box_is_the_class_line(_shot, CLASS_BOX) is False


def test_an_absent_box_is_not_shared(store):
    mgr, shot = store
    _add(mgr, shot, 'Ship Type', 'Verne Temporal Science Vessel', CLASS_BOX)
    assert mgr.tier_box_is_the_class_line(shot, None) is False
