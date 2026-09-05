"""The screen classifier reads its class count from the weights, not a guess.

It used to come from `screen_classifier_meta.json`, defaulting to 7 when that
file was absent — and `ModelUpdater` never downloaded it, so the default was
always what applied. When the published model grew to 8 classes the client kept
building a 7-class head and `load_state_dict` refused the whole model:

    size mismatch for classifier.3.weight: copying a param with shape
    torch.Size([8, 1024]) ... current model is torch.Size([7, 1024])

Screen typing then fell to its fallback ladder and got it wrong — a SPACE_EQ
screenshot was classified GROUND_EQ at 0.70 confidence. Observed on the
maintainer's install, 2026-09-06.

A checkpoint cannot disagree with itself, so the head shape now comes from it.

Offline: torch is required to build a checkpoint, so the whole module skips
without it. No network, no real model.
"""
from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from warp.recognition.screen_classifier import ScreenTypeClassifier


LABELS_8 = ['BOFFS', 'DISCARD', 'GROUND_EQ', 'GROUND_MIXED', 'SPACE_EQ',
            'SPACE_MIXED', 'SPECIALIZATIONS', 'TRAITS']


def _write_model(models_dir, n_classes: int, labels: list[str] | None,
                 meta: dict | None = None):
    """A real MobileNetV3 checkpoint with an n-class head."""
    from torchvision.models import mobilenet_v3_small
    import torch.nn as nn

    models_dir.mkdir(parents=True, exist_ok=True)
    m = mobilenet_v3_small(weights=None)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, n_classes)
    torch.save(m.state_dict(), models_dir / 'screen_classifier.pt')
    if labels is not None:
        (models_dir / 'screen_classifier_labels.json').write_text(
            json.dumps({str(i): s for i, s in enumerate(labels)}),
            encoding='utf-8')
    if meta is not None:
        (models_dir / 'screen_classifier_meta.json').write_text(
            json.dumps(meta), encoding='utf-8')
    return models_dir


# ── The production case ───────────────────────────────────────────────────

def test_an_eight_class_model_loads_without_a_meta_file(tmp_path):
    """Exactly the shipped situation: 8-class weights, 8 labels, no meta."""
    d = _write_model(tmp_path / 'models', 8, LABELS_8)
    c = ScreenTypeClassifier(d)
    assert c._session is not None
    assert not c._ml_disabled
    assert list(c._label_map.values()) == LABELS_8


def test_a_seven_class_model_still_loads(tmp_path):
    """The previous release's model must not stop working."""
    d = _write_model(tmp_path / 'models', 7, LABELS_8[:7])
    c = ScreenTypeClassifier(d)
    assert c._session is not None
    assert len(c._label_map) == 7


def test_a_stale_meta_file_cannot_break_the_load(tmp_path):
    """The weights outrank a file that says otherwise."""
    d = _write_model(tmp_path / 'models', 8, LABELS_8, meta={'n_classes': 7})
    c = ScreenTypeClassifier(d)
    assert c._session is not None
    assert not c._ml_disabled


# ── Refusing what cannot be reported ──────────────────────────────────────

def test_more_classes_than_labels_is_refused_loudly(tmp_path, capfd):
    """A class index nobody can name resolves to nothing downstream, so the
    model is not used — and the reason is said out loud.

    Captured with `capfd`, not `caplog`: `warp.debug.log` writes to its own
    file and to stderr rather than propagating through the stdlib logging
    tree, so `caplog.text` stays empty however loud the message is.
    """
    d = _write_model(tmp_path / 'models', 8, LABELS_8[:6])
    c = ScreenTypeClassifier(d)
    assert c._ml_disabled
    assert c._session is None
    err = capfd.readouterr().err
    assert '8 classes' in err and '6 labels' in err


def test_fewer_classes_than_labels_is_refused_too(tmp_path):
    d = _write_model(tmp_path / 'models', 6, LABELS_8)
    c = ScreenTypeClassifier(d)
    assert c._ml_disabled


# ── Degenerate inputs ─────────────────────────────────────────────────────

def test_no_model_file_is_not_an_error(tmp_path):
    d = tmp_path / 'models'
    d.mkdir()
    c = ScreenTypeClassifier(d)
    assert c._session is None
    assert not c._ml_disabled          # OCR fallback, not a failure


def test_no_labels_file_falls_back_to_the_built_in_names(tmp_path):
    """SCREEN_TYPES has 9 entries, so a 9-class model matches it."""
    from warp.recognition.screen_classifier import SCREEN_TYPES
    d = _write_model(tmp_path / 'models', len(SCREEN_TYPES), None)
    c = ScreenTypeClassifier(d)
    assert c._session is not None
    assert list(c._label_map.values()) == SCREEN_TYPES
