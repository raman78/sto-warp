"""A crop of one flat colour cannot be a session example.

`TM_CCOEFF_NORMED` divides by the template's standard deviation. For a
constant template that is 0/0, and OpenCV's guard resolves it to exactly
1.00 — against every query, colourful or not. Two such crops (pure black,
labelled `__empty__`) reached the community pool, so every real icon in every
screenshot was also offered `__empty__` at 0.80 + 0.20·histogram ≈ 0.80–0.85.

Measured over the 301 rows of recog_runs.jsonl written before this guard: not
one session score fell between 0 and 0.80. No genuine match below that floor
could surface, and the anti-virtual guards had to shoot the false `__empty__`
down slot by slot — which is how a screenshot with three guard warnings per
icon looked like a regression.

Run standalone:
    python -m pytest tests/test_flat_session_example.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip('cv2')


@pytest.fixture(autouse=True)
def _isolate_user_dirs(monkeypatch, tmp_path):
    """Importing the matcher initialises `warp.debug`, which rotates the log
    files under the user's real config dir unless they are redirected."""
    monkeypatch.setenv('WARP_LOG_DIR', str(tmp_path / 'logs'))
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'config'))
    monkeypatch.setenv('XDG_DATA_HOME', str(tmp_path / 'data'))
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path / 'cache'))


@pytest.fixture(autouse=True)
def _empty_pool():
    """The pool is class state — leave it as it was found."""
    from warp.recognition.icon_matcher import SETSIconMatcher
    saved = SETSIconMatcher._session_examples
    SETSIconMatcher._session_examples = []
    yield
    SETSIconMatcher._session_examples = saved


def _icon() -> np.ndarray:
    """A crop with structure, standing in for a real icon."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (51, 40, 3), dtype=np.uint8)


# ── The guard ──────────────────────────────────────────────────────────────

def test_a_flat_crop_is_refused(name='__empty__'):
    from warp.recognition.icon_matcher import SETSIconMatcher

    SETSIconMatcher.add_session_example(
        np.zeros((90, 70, 3), dtype=np.uint8), name, origin='community')

    assert SETSIconMatcher._session_examples == []


def test_a_flat_crop_of_any_colour_is_refused():
    """Black is what turned up in the pool, but the defect is flatness."""
    from warp.recognition.icon_matcher import SETSIconMatcher

    SETSIconMatcher.add_session_example(
        np.full((51, 40, 3), 90, dtype=np.uint8), '__inactive__',
        origin='community')

    assert SETSIconMatcher._session_examples == []


def test_a_crop_with_a_single_differing_pixel_is_kept():
    """The pathology is exactly at zero variance: one pixel out of 4096
    already scores 0.006 against an unrelated query, so the guard needs no
    tolerance and must not grow one."""
    from warp.recognition.icon_matcher import SETSIconMatcher

    crop = np.zeros((51, 40, 3), dtype=np.uint8)
    crop[0, 0] = 255
    SETSIconMatcher.add_session_example(crop, '__empty__', origin='community')

    assert len(SETSIconMatcher._session_examples) == 1


def test_a_real_icon_is_still_seeded():
    from warp.recognition.icon_matcher import SETSIconMatcher

    SETSIconMatcher.add_session_example(_icon(), 'Deflector Array',
                                        origin='trainer_td')

    assert len(SETSIconMatcher._session_examples) == 1


# ── What it was doing to matching ──────────────────────────────────────────

def test_a_real_icon_is_no_longer_scored_as_empty(monkeypatch):
    """The consequence, at the stage that produced it: with the flat crop
    refused, the only `__empty__` on offer is the one that genuinely looks
    like an empty slot — and a real icon does not match it."""
    from warp.recognition.icon_matcher import (
        MATCH_SIZE, SETSIconMatcher,
    )
    import cv2

    SETSIconMatcher.add_session_example(
        np.zeros((90, 70, 3), dtype=np.uint8), '__empty__', origin='community')

    query = cv2.resize(_icon(), (MATCH_SIZE, MATCH_SIZE),
                       interpolation=cv2.INTER_AREA)
    matcher = SETSIconMatcher.__new__(SETSIconMatcher)
    name, score, _entry = matcher._best_session_match(
        query, SETSIconMatcher._hist_hsv(query), {'__empty__'})

    assert score < 0.8, f'{name!r} scored {score:.3f} against a real icon'
