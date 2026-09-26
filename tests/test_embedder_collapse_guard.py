"""A collapsed embedder gallery is refused, and recognition falls back.

The embedder published 2026-09-26 05:59 UTC mapped every picture to almost
the same vector: random gallery pairs averaged 0.990 cosine similarity,
against 0.034 for every earlier version, while its val_recall@1 still read
0.80. Every absolute threshold the client applies to embedder similarity —
auto-accept, the knowledge picture check, the empty-slot guards — stops
meaning anything. The client now measures the spread on load and refuses
a collapsed gallery.

Run standalone:
    python -m pytest tests/test_embedder_collapse_guard.py -v
"""
from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip('cv2')
pytest.importorskip('torch')


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    monkeypatch.setenv('WARP_LOG_DIR', str(tmp_path / 'logs'))
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'config'))
    monkeypatch.setenv('XDG_DATA_HOME', str(tmp_path / 'data'))
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path / 'cache'))


def _healthy(n=400, d=64):
    return np.random.default_rng(3).normal(size=(n, d)).astype(np.float32)


def _collapsed(n=400, d=64):
    base = np.random.default_rng(4).normal(size=d)
    noise = np.random.default_rng(5).normal(scale=0.05, size=(n, d))
    return (base + noise).astype(np.float32)


def test_a_spread_out_gallery_measures_near_zero():
    from warp.recognition.icon_matcher import gallery_spread

    assert abs(gallery_spread(_healthy())) < 0.1


def test_a_collapsed_gallery_measures_near_one():
    from warp.recognition.icon_matcher import GALLERY_COLLAPSED_SIM, gallery_spread

    assert gallery_spread(_collapsed()) > GALLERY_COLLAPSED_SIM


def _models(embeddings):
    """A models dir holding an embedder whose gallery is `embeddings`."""
    from warp import userdata
    d = userdata.models_dir()
    (d / 'icon_embedder.pt').write_bytes(b'not read before the spread check')
    np.savez(d / 'embedding_index.npz', embeddings=embeddings,
             labels=np.zeros(len(embeddings), dtype=np.int32))
    (d / 'embedder_label_map.json').write_text(json.dumps({'0': 'Item'}))
    return d


def test_a_collapsed_gallery_is_not_used(tmp_path):
    from warp.recognition.icon_matcher import SETSIconMatcher
    _models(_collapsed())
    m = SETSIconMatcher()
    m._get_ml_session()

    assert m._ml_kind != 'embedder'
    assert m._gallery_emb is None


def test_the_refusal_is_said_in_the_log(tmp_path, capfd):
    from warp.recognition.icon_matcher import SETSIconMatcher
    _models(_collapsed())
    SETSIconMatcher()._get_ml_session()

    assert 'collapsed' in capfd.readouterr().err


def test_the_refusal_is_not_repeated_on_every_match(tmp_path, capfd):
    """With no classifier to fall back to, each match used to retry — and
    log — the whole load again."""
    from warp.recognition.icon_matcher import SETSIconMatcher
    _models(_collapsed())
    m = SETSIconMatcher()
    for _ in range(3):
        m._get_ml_session()

    assert capfd.readouterr().err.count('collapsed') == 1
