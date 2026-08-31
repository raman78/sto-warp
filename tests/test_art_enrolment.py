"""Wiki-art gallery entries: ranking, confidence, and cache invalidation.

The shipped gallery is built from confirmed crops, so an item nobody has ever
confirmed has no entry and the embedder cannot name it. It does not fall
silent on those: measured over 1500 crops with their own item hidden, it
answers with the nearest thing it knows at mean similarity 0.484, and 29.5% of
those wrong answers clear `ML_PRIMARY_THRESHOLD` and take the slot from the
template stage, which had the right answer.

Enrolling the wiki PNG gives it something correct to point at. Four things
about that are easy to get wrong and silent when wrong — each has a test here,
and each was a real defect in the first version of this code.

Run standalone:
    python -m pytest tests/test_art_enrolment.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip('torch')
pytest.importorskip('cv2')


def _matcher_with_gallery(vectors, names, is_art):
    """A matcher whose k-NN gallery is set directly — no model download."""
    from warp.recognition.icon_matcher import SETSIconMatcher

    m = SETSIconMatcher()
    emb = np.asarray(vectors, dtype=np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    m._gallery_emb = emb
    m._gallery_lbl = np.arange(len(names), dtype=np.int32)
    m._gallery_is_art = np.asarray(is_art, dtype=bool)
    m._label_map = dict(enumerate(names))
    m._ml_kind = 'embedder'

    class _StubEmbedder:
        """Returns the query vector the test asked for, L2-normed."""

        def __init__(self):
            self.vector = None

        def __call__(self, _tensor):
            import torch
            v = np.asarray(self.vector, dtype=np.float32)
            v = v / np.linalg.norm(v)
            return torch.from_numpy(v[None, :])

    m._ml_session = _StubEmbedder()
    return m


def _query(m, vector, candidates=None):
    """`candidate_names` mirrors the slot-driven call: the matcher only fills
    the real-vs-virtual similarity signals when the caller restricts the
    search, which is what `match()` does for every real slot."""
    m._ml_session.vector = vector
    crop = np.zeros((64, 64, 3), dtype=np.uint8)
    return m._classify_ml_embed(crop, candidates)


def test_the_offset_lets_art_win_a_contest_it_would_otherwise_lose():
    """Art sits ~0.33 further from a crop than a confirmed entry of the same
    item does, so without compensation it loses to real crops of *other*
    items — 97% of the residual errors."""
    m = _matcher_with_gallery(
        vectors=[[1.0, 0.0, 0.0],        # art entry for the true item
                 [0.94, 0.34, 0.0]],     # confirmed crop of a different item
        names=['True Item', 'Other Item'],
        is_art=[True, False],
    )

    name, _ = _query(m, [0.97, 0.24, 0.0])

    assert name == 'True Item'


def test_the_offset_does_not_inflate_reported_confidence():
    """The offset settles the contest; it must not travel into the number the
    rest of the pipeline reads.

    Clamped to [0, 1], an art row at raw 0.92 would otherwise report 1.00 —
    the certainty of a pixel-perfect confirmed crop, on weaker evidence —
    and that number drives ML_PRIMARY_THRESHOLD, VIRTUAL_OVERRIDE_CONF and
    WARP CORE's auto-accept. It is reachable, not theoretical: 30.8% of art
    rows sit within 0.95 of another art row.
    """
    m = _matcher_with_gallery(vectors=[[1.0, 0.0, 0.0]], names=['Art Item'],
                              is_art=[True])

    name, conf = _query(m, [1.0, 0.0, 0.0])

    assert name == 'Art Item'
    assert conf == pytest.approx(1.0)          # a perfect match may say 1.0

    m2 = _matcher_with_gallery(vectors=[[1.0, 0.0, 0.0]], names=['Art Item'],
                               is_art=[True])
    _, conf2 = _query(m2, [0.95, 0.312, 0.0])  # cosine ≈ 0.95, not 1.0

    assert conf2 < 1.0, 'the +0.10 ranking offset leaked into the confidence'
    assert conf2 == pytest.approx(0.95, abs=0.01)


def test_empty_slot_detection_is_judged_on_raw_scores():
    """`_last_embed_sim_real` vs `_last_embed_sim_virtual` answers a different
    question — "is there an item here at all?" — and the virtual entries have
    no art counterpart to compensate. Boosting only the real side would tilt
    an empty slot towards being called occupied, which invents an item rather
    than merely misnaming one.
    """
    from warp.recognition.icon_matcher import VIRTUAL_LABELS

    virtual = sorted(VIRTUAL_LABELS)[0]
    m = _matcher_with_gallery(
        vectors=[[1.0, 0.0, 0.0],     # art entry for a real item
                 [0.0, 1.0, 0.0]],    # the empty/inactive entry
        names=['Art Item', virtual],
        is_art=[True, False],
    )

    _query(m, [0.6, 0.8, 0.0], candidates={'Art Item'})   # closer to virtual

    assert m._last_embed_sim_real == pytest.approx(0.6, abs=0.01)
    assert m._last_embed_sim_virtual == pytest.approx(0.8, abs=0.01)
    assert m._last_embed_sim_real < m._last_embed_sim_virtual


def test_an_art_win_is_reported_as_its_own_source():
    m = _matcher_with_gallery(vectors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                              names=['Art Item', 'Crop Item'],
                              is_art=[True, False])

    _query(m, [1.0, 0.0, 0.0])
    assert m._last_embed_was_art is True

    _query(m, [0.0, 1.0, 0.0])
    assert m._last_embed_was_art is False


def test_the_art_cache_is_rebuilt_when_the_embedder_changes(tmp_path, monkeypatch):
    """A retrained embedder of the same architecture is byte-for-byte the same
    length — 17.6 MB, 360 tensors — with entirely different weights. Keyed on
    size, the cache would survive that boundary and serve vectors from the
    previous embedding space: no error raised, just quietly worse matches.
    `model_updater` copies new models in without deleting anything, so a stale
    cache really does survive an update.
    """
    import cv2

    from warp.recognition.icon_matcher import SETSIconMatcher

    icons = tmp_path / 'icons'
    icons.mkdir()
    cv2.imwrite(str(icons / 'Some Item.png'), np.zeros((64, 64, 3), np.uint8))

    models = tmp_path / 'models'
    models.mkdir()
    embedder = models / 'icon_embedder.pt'
    embedder.write_bytes(b'\x01' * 4096)

    calls = []

    def _fresh():
        m = SETSIconMatcher(icons)
        m._gallery_emb = np.zeros((1, 8), dtype=np.float32)
        m._gallery_lbl = np.zeros(1, dtype=np.int32)
        m._gallery_is_art = np.zeros(1, dtype=bool)
        m._label_map = {0: 'Already Known'}
        m._embed_crop = lambda crop: (calls.append(1),
                                      np.ones(8, dtype=np.float32))[1]
        return m

    _fresh()._enroll_wiki_art(models)
    assert calls == [1], 'first run must embed the missing icon'

    _fresh()._enroll_wiki_art(models)
    assert calls == [1], 'same embedder — the cached vector must be reused'

    same_size = bytearray(b'\x01' * 4096)
    same_size[-1] = 0x02                       # new weights, identical length
    embedder.write_bytes(bytes(same_size))
    assert embedder.stat().st_size == 4096

    _fresh()._enroll_wiki_art(models)
    assert len(calls) == 2, 'a different embedder must invalidate the cache'


def test_enrolment_folds_era_variant_art_onto_its_item(tmp_path, monkeypatch):
    """34 PNGs are named 'X (23c)' for the same item as 'X'. Enrolled under the
    raw filename they become labels cargo has never heard of — unreachable as
    slot candidates, and able to win an unrestricted top-1 with a name nothing
    downstream can resolve. `_build_index` folds them; enrolment must too.
    """
    import cv2

    from warp.recognition import icon_matcher as IM

    icons = tmp_path / 'icons'
    icons.mkdir()
    cv2.imwrite(str(icons / 'Impulse Engines (23c).png'),
                np.zeros((64, 64, 3), np.uint8))
    models = tmp_path / 'models'
    models.mkdir()
    (models / 'icon_embedder.pt').write_bytes(b'\x01' * 512)

    monkeypatch.setattr(IM, 'canonical_names', lambda: {'Impulse Engines'},
                        raising=False)
    monkeypatch.setattr('warp.data.cargo.canonical_names',
                        lambda: {'Impulse Engines'})

    m = IM.SETSIconMatcher(icons)
    m._gallery_emb = np.zeros((1, 8), dtype=np.float32)
    m._gallery_lbl = np.zeros(1, dtype=np.int32)
    m._gallery_is_art = np.zeros(1, dtype=bool)
    m._label_map = {0: 'Already Known'}
    m._embed_crop = lambda crop: np.ones(8, dtype=np.float32)

    m._enroll_wiki_art(models)

    enrolled = [m._label_map[int(l)]
                for l, a in zip(m._gallery_lbl, m._gallery_is_art) if a]
    assert enrolled == ['Impulse Engines'], enrolled
