"""A layout is settled when the user says the screenshot is done, not before.

Re-detection used to be skipped as soon as *any* confirmed annotation existed
for a screenshot, on the reasoning that it would overwrite the user's
pixel-perfect bboxes. It does not — `_apply_confirmed` puts them back. What
the old condition actually did was freeze a layout the moment its first row
was confirmed: a row that gained a cell could never gain a box, and a phantom
the user deleted came back on the next run because the profile still asked
for it.

`screenshots_done.json` is the state the user sets and can see. Anything else
is work in progress and gets a fresh scan.

Offline: a temporary training store under `tmp_path`, no image and no ML.
"""
from __future__ import annotations

import json

import pytest

from warp.warp_importer import WarpImporter


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A training store the importer will read `screenshots_done.json` from."""
    from warp import userdata
    d = tmp_path / 'training_data'
    d.mkdir()
    monkeypatch.setattr(userdata, 'training_data_dir', lambda: d)
    return d


def _importer():
    return WarpImporter.__new__(WarpImporter)


def test_a_screenshot_on_the_done_list_is_settled(store):
    (store / 'screenshots_done.json').write_text(json.dumps(['a.png', 'b.png']))
    assert _importer()._screenshot_is_done('/some/where/b.png')


def test_one_not_on_the_list_is_work_in_progress(store):
    (store / 'screenshots_done.json').write_text(json.dumps(['a.png']))
    assert not _importer()._screenshot_is_done('/some/where/b.png')


def test_the_full_path_does_not_have_to_match(store):
    """The list holds bare filenames; the importer is handed a path."""
    (store / 'screenshots_done.json').write_text(json.dumps(['b.png']))
    assert _importer()._screenshot_is_done('/a/completely/other/dir/b.png')


def test_no_list_at_all_means_nothing_is_settled(store):
    assert not _importer()._screenshot_is_done('/some/where/b.png')


def test_an_unreadable_list_means_nothing_is_settled(store):
    """The safe direction: a fresh scan followed by the confirmed merge,
    rather than a layout nothing can correct."""
    (store / 'screenshots_done.json').write_text('{ not json')
    assert not _importer()._screenshot_is_done('/some/where/b.png')


def test_an_unexpected_shape_is_not_read_as_done(store):
    (store / 'screenshots_done.json').write_text(json.dumps({'b.png': True}))
    assert not _importer()._screenshot_is_done('/some/where/b.png')
