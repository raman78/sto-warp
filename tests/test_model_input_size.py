"""The matcher feeds each model the input size its meta file declares.

224 was hardcoded in three places. It was the only size ever published, so
it held — but a model trained at 128 and fed 224 still answers, only worse,
and nothing reports it. The trainers already write `input_size` into
icon_classifier_meta.json and icon_embedder_meta.json; the matcher now reads
it, falls back to 224 for a meta that predates the field, and refuses a
model whose meta declares a size it cannot have been trained on.

Each test builds a real (untrained) EfficientNet-B0, saves it with a meta
file into an isolated models dir, and records the tensor that reaches it.

Run standalone:
    python -m pytest tests/test_model_input_size.py -v
"""
from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip('cv2')
torch = pytest.importorskip('torch')


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    monkeypatch.setenv('WARP_LOG_DIR', str(tmp_path / 'logs'))
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'config'))
    monkeypatch.setenv('XDG_DATA_HOME', str(tmp_path / 'data'))
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path / 'cache'))


def _classifier_dir(meta: dict | None):
    from torchvision.models import efficientnet_b0
    from warp import userdata
    d = userdata.models_dir()
    model = efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
    torch.save(model.state_dict(), d / 'icon_classifier.pt')
    (d / 'label_map.json').write_text(json.dumps({'0': 'A', '1': 'B'}))
    if meta is not None:
        (d / 'icon_classifier_meta.json').write_text(json.dumps(meta))
    return d


def _embedder_dir(meta: dict):
    from torchvision.models import efficientnet_b0
    from warp import userdata
    d = userdata.models_dir()
    backbone = efficientnet_b0(weights=None)
    backbone.classifier = torch.nn.Identity()
    proj = torch.nn.Linear(1280, 8)
    state = {f'backbone.{k}': v for k, v in backbone.state_dict().items()}
    state.update({f'proj.{k}': v for k, v in proj.state_dict().items()})
    torch.save(state, d / 'icon_embedder.pt')
    emb = np.random.default_rng(3).normal(size=(50, 8)).astype(np.float32)
    np.savez(d / 'embedding_index.npz', embeddings=emb,
             labels=np.zeros(50, dtype=np.int32))
    (d / 'embedder_label_map.json').write_text(json.dumps({'0': 'A'}))
    (d / 'icon_embedder_meta.json').write_text(json.dumps(meta))
    return d


def _seen_side(module, call) -> int:
    """Side of the image `module` receives while `call()` runs."""
    seen = []
    handle = module.register_forward_pre_hook(lambda _m, args: seen.append(args[0].shape))
    try:
        call()
    finally:
        handle.remove()
    assert seen, 'the model was never called'
    return seen[0][-1]


_CROP = np.zeros((64, 64, 3), dtype=np.uint8)


def test_the_classifier_gets_the_declared_size():
    from warp.recognition.icon_matcher import SETSIconMatcher
    _classifier_dir({'n_classes': 2, 'input_size': 128})
    m = SETSIconMatcher()
    m._get_ml_session()

    assert m._ml_kind == 'classifier'
    assert _seen_side(m._ml_session, lambda: m._classify_ml(_CROP)) == 128


def test_a_meta_without_the_field_means_224():
    """Every model published before the field was read was trained at 224."""
    from warp.recognition.icon_matcher import SETSIconMatcher
    _classifier_dir({'n_classes': 2})
    m = SETSIconMatcher()
    m._get_ml_session()

    assert _seen_side(m._ml_session, lambda: m._classify_ml(_CROP)) == 224


def test_no_meta_file_means_224():
    from warp.recognition.icon_matcher import SETSIconMatcher
    _classifier_dir(None)
    m = SETSIconMatcher()
    m._get_ml_session()

    assert _seen_side(m._ml_session, lambda: m._classify_ml(_CROP)) == 224


@pytest.mark.parametrize('bad', ['128', 0, 5000, True, 12.5])
def test_an_impossible_size_refuses_the_model(bad):
    """Guessing a size for a model that declares nonsense would answer
    wrongly with nothing to show for it; refusing says so in the log."""
    from warp.recognition.icon_matcher import SETSIconMatcher
    _classifier_dir({'n_classes': 2, 'input_size': bad})
    m = SETSIconMatcher()
    m._get_ml_session()

    assert m._ml_kind != 'classifier'


def test_the_embedder_gets_the_declared_size():
    from warp.recognition.icon_matcher import SETSIconMatcher
    _embedder_dir({'embed_dim': 8, 'input_size': 128})
    m = SETSIconMatcher()
    m._get_ml_session()

    assert m._ml_kind == 'embedder'
    assert _seen_side(m._ml_session, lambda: m._embed_crop(_CROP)) == 128
    assert _seen_side(m._ml_session,
                      lambda: m._classify_ml_embed(_CROP, None)) == 128
