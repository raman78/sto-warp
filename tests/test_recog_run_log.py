"""The recognition run log must accumulate, because the history does not.

`recog_history.json` stores one entry per image and replaces it on every run
(`hist[image_name] = curr`). That is what the summary table's Δ column needs,
and exactly wrong for measuring a change over weeks: re-running a screenshot
destroys the state it would be compared against.

`recog_runs.jsonl` exists to answer the before/after question instead. It is
append-only, so a baseline survives every later run.

Run standalone:
    python -m pytest tests/test_recog_run_log.py -v
"""
from __future__ import annotations

import json

import pytest

pytest.importorskip('PySide6')


@pytest.fixture
def ui(tmp_path, monkeypatch):
    """Redirect both files. The paths are module constants resolved at import
    time, so redirecting XDG afterwards would not move them."""
    from warp.trainer import _ui_utils

    monkeypatch.setattr(_ui_utils, '_RECOG_HISTORY_PATH',
                        tmp_path / 'recog_history.json')
    monkeypatch.setattr(_ui_utils, '_RECOG_RUNS_PATH',
                        tmp_path / 'recog_runs.jsonl')
    return _ui_utils


def _match_log(name: str, conf: float, src: str = 'embed') -> list[dict]:
    return [{
        'slot': 'Fore Weapons', 'name': name, 'conf': conf, 'src': src,
        'stages': {'embed': conf, 'session': 0.0, 'template': 0.4,
                   'knowledge': 0.0},
    }]


def _lines(path):
    return [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l]


def test_a_run_is_recorded(ui, tmp_path):
    ui._log_match_summary('shot.png', _match_log('Phaser Beam Array', 0.74))

    rows = _lines(tmp_path / 'recog_runs.jsonl')
    assert len(rows) == 1
    assert rows[0]['image'] == 'shot.png'
    assert rows[0]['slot'] == 'Fore Weapons'
    assert rows[0]['name'] == 'Phaser Beam Array'
    assert rows[0]['src'] == 'embed'
    assert rows[0]['conf'] == pytest.approx(0.74)


def test_a_second_run_does_not_erase_the_first(ui, tmp_path):
    """The whole point. The history overwrites here; the run log must not."""
    ui._log_match_summary('shot.png', _match_log('Phaser Beam Array', 0.74))
    ui._log_match_summary('shot.png', _match_log('Disruptor Beam Array', 0.91))

    rows = _lines(tmp_path / 'recog_runs.jsonl')
    assert [r['name'] for r in rows] == ['Phaser Beam Array',
                                         'Disruptor Beam Array']

    # …while the history kept only the newer answer, as it always has.
    hist = json.loads((tmp_path / 'recog_history.json').read_text(encoding='utf-8'))
    assert hist['shot.png']['Fore Weapons']['name'] == 'Disruptor Beam Array'


def test_per_stage_scores_are_kept(ui, tmp_path):
    """So "did template already have the answer ML got wrong?" can be answered
    from the file, without re-running the corpus."""
    ui._log_match_summary('shot.png', _match_log('Phaser Beam Array', 0.74))

    row = _lines(tmp_path / 'recog_runs.jsonl')[0]

    assert row['template'] == pytest.approx(0.4)
    assert row['embed'] == pytest.approx(0.74)
    assert 'session' in row and 'knowledge' in row


def test_every_slot_of_a_run_is_recorded(ui, tmp_path):
    log = _match_log('Phaser Beam Array', 0.74) + _match_log('Photon Torpedo', 0.8)
    ui._log_match_summary('shot.png', log)

    rows = _lines(tmp_path / 'recog_runs.jsonl')

    # Repeated slots are disambiguated the same way the history keys them.
    assert [r['slot'] for r in rows] == ['Fore Weapons', 'Fore Weapons#1']


def test_an_empty_run_writes_nothing(ui, tmp_path):
    ui._log_match_summary('shot.png', [])

    assert not (tmp_path / 'recog_runs.jsonl').exists()


def test_logging_failure_never_breaks_recognition(ui, tmp_path, monkeypatch):
    """Telemetry is not allowed to take the pipeline down with it."""
    monkeypatch.setattr(ui, '_RECOG_RUNS_PATH', tmp_path / 'nope' / 'x.jsonl')
    monkeypatch.setattr(ui, '_model_fingerprint',
                        lambda: (_ for _ in ()).throw(RuntimeError('boom')))

    ui._log_match_summary('shot.png', _match_log('Phaser Beam Array', 0.74))
