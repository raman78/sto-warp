"""A name with two pictures must show the one that was actually seen.

34 items ship with both base art and a 23rd-century variant. `_build_index`
folds the variant onto the base name so it is recognisable at all, which
leaves two index entries under one name.

The embedder names an item without saying which picture it saw — its gallery
is keyed on names and has no notion of a variant — so `match()` falls back to
resolving the thumbnail from the name. Taking the first of the two entries is
a coin toss. On the reported screenshot it landed wrong: the crop was the red
base weapon, the two entries scored 0.443 and 0.574 against it, and the
tooltip showed the 0.443 one, a visibly different weapon under the right name.

`match()` already computes a template score per entry, so choosing the picture
that matches costs nothing.

Run standalone:
    python -m pytest tests/test_variant_thumbnail.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip('cv2')
pytest.importorskip('PySide6')


def _matcher_with_two_variants():
    from warp.recognition.icon_matcher import SETSIconMatcher

    m = SETSIconMatcher.__new__(SETSIconMatcher)
    blue = np.zeros((64, 49, 3), np.uint8); blue[:, :] = (200, 90, 20)
    red = np.zeros((64, 49, 3), np.uint8); red[:, :] = (20, 40, 210)
    m._index = [
        {'name': 'Other Item', 'orig': np.zeros((64, 49, 3), np.uint8)},
        {'name': 'Two Faces', 'orig': blue},     # first — the 23c-style entry
        {'name': 'Two Faces', 'orig': red},      # second — the better match
    ]
    return m, blue, red


def _colour(qimg):
    return qimg.pixelColor(10, 10).name()


def test_without_scores_the_first_entry_is_used():
    """Unchanged fallback: nothing to choose with, so keep the old behaviour."""
    m, blue, _red = _matcher_with_two_variants()

    assert _colour(m._thumb_for_name('Two Faces')) == _colour(
        m._bgr_to_qimage(blue))


def test_the_better_scoring_picture_wins():
    """The reported case."""
    m, _blue, red = _matcher_with_two_variants()
    scores = np.array([0.10, 0.443, 0.574], dtype=np.float32)

    assert _colour(m._thumb_for_name('Two Faces', scores)) == _colour(
        m._bgr_to_qimage(red))


def test_a_single_picture_is_unaffected():
    m, _blue, _red = _matcher_with_two_variants()
    scores = np.array([0.9, 0.1, 0.1], dtype=np.float32)

    assert m._thumb_for_name('Other Item', scores) is not None


def test_virtual_labels_have_no_thumbnail():
    m, _b, _r = _matcher_with_two_variants()

    assert m._thumb_for_name('__empty__', np.zeros(3, np.float32)) is None


def test_an_unknown_name_has_no_thumbnail():
    m, _b, _r = _matcher_with_two_variants()

    assert m._thumb_for_name('Not In Index', np.zeros(3, np.float32)) is None
