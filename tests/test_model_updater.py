"""Tests for warp.trainer.model_updater — embedder version gating.

The ArcFace embedder is published by its own trainer, so it can be newer than
the softmax classifier. These tests pin the rule: a current classifier must not
mask a newer embedder, and the embedder-only refresh must not drag the 31 MB
classifier along with it.
"""

import json

import pytest

from warp.trainer.model_updater import (
    ModelUpdater,
    _EMBEDDER_FILES,
    _MODEL_FILES,
    _REQUIRED_EMBEDDER,
)

_LOCAL_TRAINED_AT    = '2026-07-16T22:57:34+00:00Z'
_LOCAL_EMBEDDER_AT   = '2026-07-17T04:06:13+00:00Z'


def _write_models(models_dir, embedder_trained_at=_LOCAL_EMBEDDER_AT):
    """Populate a models dir that looks like a healthy, fully-installed client."""
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / 'model_version.json').write_text(
        json.dumps({'trained_at': _LOCAL_TRAINED_AT, 'n_classes': 3148}),
        encoding='utf-8',
    )
    (models_dir / 'icon_embedder_meta.json').write_text(
        json.dumps({'trained_at': embedder_trained_at, 'n_classes': 1970}),
        encoding='utf-8',
    )
    (models_dir / 'embedder_label_map.json').write_text(
        json.dumps({'0': 'Phaser Beam Array', '1': 'Quantum Torpedo Launcher'}),
        encoding='utf-8',
    )
    (models_dir / 'icon_embedder.pt').write_bytes(b'pt')
    (models_dir / 'embedding_index.npz').write_bytes(b'npz')
    (models_dir / 'screen_classifier.pt').write_bytes(b'pt')
    return models_dir


@pytest.fixture
def models_dir(tmp_path, monkeypatch):
    """Redirect userdata.models_dir()/cache_dir() into tmp_path."""
    md = tmp_path / 'models'
    cd = tmp_path / 'cache'
    cd.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr('warp.userdata.models_dir', lambda: md)
    monkeypatch.setattr('warp.userdata.cache_dir', lambda: cd)
    monkeypatch.setattr('warp.userdata.ensure_migrated', lambda: None)
    return _write_models(md)


# ── _embedder_is_outdated ──────────────────────────────────────────────────


def test_embedder_outdated_when_remote_is_newer(models_dir):
    remote = {'embedder_trained_at': '2026-08-13T03:26:02+00:00Z'}
    assert ModelUpdater._embedder_is_outdated(models_dir, remote) is True


def test_embedder_current_when_timestamps_match(models_dir):
    remote = {'embedder_trained_at': _LOCAL_EMBEDDER_AT}
    assert ModelUpdater._embedder_is_outdated(models_dir, remote) is False


def test_embedder_current_when_local_is_newer(models_dir):
    remote = {'embedder_trained_at': '2026-06-01T00:00:00+00:00Z'}
    assert ModelUpdater._embedder_is_outdated(models_dir, remote) is False


def test_embedder_outdated_when_local_meta_missing(models_dir):
    (models_dir / 'icon_embedder_meta.json').unlink()
    remote = {'embedder_trained_at': '2026-08-13T03:26:02+00:00Z'}
    assert ModelUpdater._embedder_is_outdated(models_dir, remote) is True


def test_no_embedder_field_keeps_old_behaviour(models_dir):
    """A backend that predates embedder_trained_at must not trigger downloads."""
    assert ModelUpdater._embedder_is_outdated(models_dir, {'trained_at': _LOCAL_TRAINED_AT}) is False


# ── _bg_check decision path ────────────────────────────────────────────────


@pytest.fixture
def spy_updater(monkeypatch):
    """ModelUpdater with the network stubbed out; records _download_model calls."""
    calls = []
    updater = ModelUpdater()

    def fake_download(models_dir, remote_meta, on_progress=None, files=None, required=None):
        calls.append({'files': files, 'required': required})
        return True

    monkeypatch.setattr(updater, '_download_model', fake_download)
    monkeypatch.setattr(updater, '_ensure_screen_classifier', lambda md: None)
    return updater, calls


def test_newer_embedder_downloads_embedder_files_only(models_dir, spy_updater, monkeypatch):
    updater, calls = spy_updater
    monkeypatch.setattr(updater, '_fetch_remote_version', lambda: {
        'available': True,
        'trained_at': _LOCAL_TRAINED_AT,                    # classifier unchanged
        'embedder_trained_at': '2026-08-13T03:26:02+00:00Z',  # embedder moved
    })

    updater._bg_check()

    assert len(calls) == 1
    assert calls[0]['files'] == _EMBEDDER_FILES
    assert calls[0]['required'] == _REQUIRED_EMBEDDER


def test_newer_classifier_downloads_full_model(models_dir, spy_updater, monkeypatch):
    updater, calls = spy_updater
    monkeypatch.setattr(updater, '_fetch_remote_version', lambda: {
        'available': True,
        'trained_at': '2026-08-13T03:00:00+00:00Z',
        'embedder_trained_at': _LOCAL_EMBEDDER_AT,
    })

    updater._bg_check()

    assert len(calls) == 1
    assert calls[0]['files'] == _MODEL_FILES


def test_everything_current_downloads_nothing(models_dir, spy_updater, monkeypatch):
    updater, calls = spy_updater
    monkeypatch.setattr(updater, '_fetch_remote_version', lambda: {
        'available': True,
        'trained_at': _LOCAL_TRAINED_AT,
        'embedder_trained_at': _LOCAL_EMBEDDER_AT,
    })

    updater._bg_check()

    assert calls == []
