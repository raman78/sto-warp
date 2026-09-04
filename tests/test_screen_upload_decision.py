"""What the client decides to upload, and what it remembers afterwards.

Two silent faults lived here, and neither could be caught by a unit test of
anything that existed: the decision was inline in the upload loop, so testing
it meant faking a network.

  1. A screenshot re-typed in WARP CORE never went up again. The check was
     "have I sent this file", and changing a type does not change the file —
     so the sha matched and the loop skipped it. Measured 2026-09-04 against
     the published dataset: 26 of 27 screenshots typed `SPACE_BOFFS` locally
     are published as `BOFFS`, the label they carried the first time; all six
     `GROUND_TRAITS` are published as `GROUND_MIXED`.

  2. Items the backend refused were remembered as sent. The response carries
     counts, not identities, and the whole batch was cached regardless — so a
     refused screenshot was never looked at again. Ten `DISCARD` screenshots
     went that way while the type was outside the backend whitelist.

The crop path has had the equivalent of (1) since it was written, which is why
crop corrections do propagate and screen corrections did not.

Run standalone:
    python -m pytest tests/test_screen_upload_decision.py -v
"""
from __future__ import annotations

import json

import pytest

pytest.importorskip('PySide6')

from warp.trainer.sync import SyncWorker


needs = SyncWorker._screen_needs_upload
load = SyncWorker._load_screen_type_cache


# ── The decision ───────────────────────────────────────────────────────────

def test_an_unseen_screenshot_is_uploaded():
    assert needs('aa', 'BOFFS', {})


def test_a_screenshot_already_sent_under_the_same_type_is_skipped():
    assert not needs('aa', 'BOFFS', {'aa': 'BOFFS'})


def test_a_retyped_screenshot_is_uploaded_again():
    """The fault: the file is unchanged, so the sha matches, but the label
    the community dataset holds is now wrong."""
    assert needs('aa', 'SPACE_BOFFS', {'aa': 'BOFFS'})


def test_a_legacy_cache_entry_forces_one_re_upload():
    """A cache written before the type was recorded cannot prove the stored
    type is right, so every entry in it goes up once."""
    assert needs('aa', 'BOFFS', {'aa': ''})


# ── Reading the cache ──────────────────────────────────────────────────────

def test_a_mapping_cache_is_read_as_written(tmp_path):
    p = tmp_path / 'c.json'
    p.write_text(json.dumps({'aa': 'BOFFS', 'bb': 'TRAITS'}))

    assert load(p) == {'aa': 'BOFFS', 'bb': 'TRAITS'}


def test_the_legacy_list_cache_is_migrated(tmp_path):
    """The format in the field until 2026-09-05. Migrated to an empty type so
    each entry is re-sent once and its real type reaches the dataset."""
    p = tmp_path / 'c.json'
    p.write_text(json.dumps(['aa', 'bb']))

    assert load(p) == {'aa': '', 'bb': ''}


def test_a_missing_cache_is_not_an_error(tmp_path):
    assert load(tmp_path / 'absent.json') == {}


def test_a_corrupt_cache_is_treated_as_empty(tmp_path):
    """Losing the cache costs one re-upload; trusting a broken one loses
    screenshots."""
    p = tmp_path / 'c.json'
    p.write_text('{ not json')

    assert load(p) == {}
