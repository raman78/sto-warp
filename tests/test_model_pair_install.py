"""Weights and their label map are installed together, or not at all.

The screen classifier's head is built over `sorted(set(labels))` across the
classes that met the backend's minimum sample count, so both its size and the
meaning of every index move from run to run. That makes
`screen_classifier_labels.json` part of the weights, not a companion file.

Two download paths treated them as independent, and each failure was
best-effort:

- `_download_model` listed both as optional, so a run where the `.pt` arrived
  and the label map did not installed new weights over the *previous* run's
  names. `ScreenTypeClassifier` refuses a pair that disagrees on count, but
  two runs can produce the same count over a different class set, and nothing
  catches that.
- `_ensure_screen_classifier` returned early on `screen_classifier.pt`
  existing, so the same partial download was permanent: the weights were
  there, the check never ran again, and the model had no names.

Offline: no network — `hf_hub_download` is monkeypatched throughout.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from warp.trainer.model_updater import ModelUpdater, _PAIRED_FILES

PT = 'screen_classifier.pt'
LABELS = 'screen_classifier_labels.json'


# ── _drop_unpaired ────────────────────────────────────────────────────────

def _pairs(tmp_path, *names):
    return [(tmp_path / f'src_{n}', tmp_path / n) for n in names]


def test_weights_without_their_labels_are_withheld(tmp_path):
    kept = ModelUpdater._drop_unpaired(_pairs(tmp_path, PT))
    assert kept == []


def test_the_pair_together_is_installed(tmp_path):
    got = _pairs(tmp_path, PT, LABELS)
    assert ModelUpdater._drop_unpaired(got) == got


def test_labels_without_weights_are_kept(tmp_path):
    """Only the weights are meaningless alone. A newer label map beside the
    same weights is the pair the next run will complete."""
    got = _pairs(tmp_path, LABELS)
    assert ModelUpdater._drop_unpaired(got) == got


def test_unrelated_files_are_untouched(tmp_path):
    got = _pairs(tmp_path, 'icon_classifier.pt', 'label_map.json')
    assert ModelUpdater._drop_unpaired(got) == got


def test_withholding_weights_does_not_drop_the_rest_of_the_download(tmp_path):
    """A missing screen-classifier label map must not cost the icon model."""
    got = _pairs(tmp_path, 'icon_classifier.pt', 'label_map.json', PT)
    kept = [dst.name for _src, dst in ModelUpdater._drop_unpaired(got)]
    assert kept == ['icon_classifier.pt', 'label_map.json']


def test_the_pairing_names_a_file_the_updater_downloads(tmp_path):
    """Guard against the table naming something no download produces."""
    from warp.trainer.model_updater import _MODEL_FILES
    locals_ = {local for _hf, local in _MODEL_FILES}
    for name, partner in _PAIRED_FILES.items():
        assert name in locals_
        assert partner in locals_


# ── _ensure_screen_classifier ─────────────────────────────────────────────

@pytest.fixture
def fake_hub(monkeypatch, tmp_path):
    """Stand in for `hf_hub_download`; `fail` names what cannot be fetched."""
    src = tmp_path / 'hub'
    src.mkdir()
    state = {'fail': set(), 'asked': []}

    def _dl(repo_id, filename, repo_type):
        name = Path(filename).name
        state['asked'].append(name)
        if name in state['fail']:
            raise RuntimeError(f'simulated failure for {name}')
        p = src / name
        p.write_bytes(b'payload-' + name.encode())
        return str(p)

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', _dl)
    return state


def test_a_partial_download_installs_nothing(tmp_path, fake_hub):
    fake_hub['fail'] = {LABELS}
    models = tmp_path / 'models'
    ModelUpdater()._ensure_screen_classifier(models)
    assert not (models / PT).exists()
    assert not (models / LABELS).exists()


def test_a_partial_download_is_retried_next_time(tmp_path, fake_hub):
    """The defect: the old guard was `screen_classifier.pt` alone, so once the
    weights landed the check never ran again and the map stayed missing."""
    models = tmp_path / 'models'
    fake_hub['fail'] = {LABELS}
    ModelUpdater()._ensure_screen_classifier(models)

    fake_hub['fail'] = set()
    fake_hub['asked'].clear()
    ModelUpdater()._ensure_screen_classifier(models)
    assert LABELS in fake_hub['asked']
    assert (models / PT).exists()
    assert (models / LABELS).exists()


def test_a_complete_pair_is_not_downloaded_again(tmp_path, fake_hub):
    models = tmp_path / 'models'
    ModelUpdater()._ensure_screen_classifier(models)
    fake_hub['asked'].clear()
    ModelUpdater()._ensure_screen_classifier(models)
    assert fake_hub['asked'] == []


def test_only_the_missing_half_is_fetched(tmp_path, fake_hub):
    """A label map lost on its own is refetched without the 6 MB of weights."""
    models = tmp_path / 'models'
    ModelUpdater()._ensure_screen_classifier(models)
    (models / LABELS).unlink()
    fake_hub['asked'].clear()
    ModelUpdater()._ensure_screen_classifier(models)
    assert fake_hub['asked'] == [LABELS]
    assert (models / LABELS).exists()
