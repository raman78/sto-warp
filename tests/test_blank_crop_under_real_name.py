"""A blank cell may not be learned under a real item's name.

Every other guard in the matcher looks one way: a colourful crop labelled
`__empty__`. Nothing looked for the opposite, and it is the more damaging of
the two. A blank cell filed under an item's name teaches the gallery that the
item is what nothing looks like; the recogniser then answers with that item on
every blank cell it meets, and confirming those answers feeds the loop.

It ran for months. Measured on the community mirror 2026-09-03: of 9227 crops
carrying a real item name, 25 are blank cells, and 20 of the 25 are the same
name — `Charged Particle Burst`, which is 20 of the 29 crops that class has.
An inactive BOFF cell sits at cosine 0.92 from those 20 and at 0.45 from the
9 genuine ones, so every one of the recogniser's residual confusions on
inactive cells traced back to them rather than to the model.

Run standalone:
    python -m pytest tests/test_blank_crop_under_real_name.py -v
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


def _inactive_boff_cell() -> np.ndarray:
    """A locked BOFF cell: uniform navy with a faint X, as the game draws it.

    Hue ~110 and saturation ~180 in OpenCV's HSV, brightness low and even —
    the signature `LayoutDetector._classify_cell` keys on.
    """
    import cv2
    hsv = np.zeros((43, 33, 3), dtype=np.uint8)
    hsv[:, :, 0] = 110
    hsv[:, :, 1] = 180
    hsv[:, :, 2] = 70
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _real_icon() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (43, 33, 3), dtype=np.uint8)


# ── The guard ──────────────────────────────────────────────────────────────

def test_a_blank_cell_is_refused_under_an_item_name():
    from warp.recognition.icon_matcher import SETSIconMatcher

    SETSIconMatcher.add_session_example(
        _inactive_boff_cell(), 'Charged Particle Burst', origin='community')

    assert SETSIconMatcher._session_examples == []


def test_the_same_cell_is_still_accepted_under_a_virtual_name():
    """The crop is not the problem — the label is. An inactive cell filed as
    `__inactive__` is exactly what the anti-virtual rules need to compare
    against, and refusing it would break them."""
    from warp.recognition.icon_matcher import SETSIconMatcher

    SETSIconMatcher.add_session_example(
        _inactive_boff_cell(), '__inactive__', origin='community')

    assert len(SETSIconMatcher._session_examples) == 1


def test_a_real_icon_is_still_seeded():
    from warp.recognition.icon_matcher import SETSIconMatcher

    SETSIconMatcher.add_session_example(_real_icon(), 'Deflector Array',
                                        origin='trainer_td')

    assert len(SETSIconMatcher._session_examples) == 1


# ── The judgement it delegates to ──────────────────────────────────────────

def test_the_guard_reads_a_blank_cell_as_blank():
    from warp.recognition.icon_matcher import _real_crop_looks_blank

    assert _real_crop_looks_blank(_inactive_boff_cell()) is True


def test_the_guard_leaves_a_real_icon_alone():
    from warp.recognition.icon_matcher import _real_crop_looks_blank

    assert _real_crop_looks_blank(_real_icon()) is False
