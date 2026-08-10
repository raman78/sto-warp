"""The baseline snapshot must not quietly absorb upstream data loss.

`warp/data/baseline/` is what a user gets when their very first run has no
network. A shrunken snapshot committed to the wheel would hand them a
regression that no later upstream fix can reach.
"""
from __future__ import annotations

import json

import pytest

from warp.tools import make_baseline


@pytest.fixture
def baseline(tmp_path, monkeypatch):
    monkeypatch.setattr(make_baseline, 'BASELINE_DIR', tmp_path)
    monkeypatch.setattr(make_baseline, 'FILES', ('equipment.json',))
    return tmp_path


def _serve(monkeypatch, rows: int):
    payload = json.dumps([{'name': f'item {i}'} for i in range(rows)]).encode()
    monkeypatch.setattr(make_baseline, '_fetch',
                        lambda name: (payload, None, 'https://host/o/r/main/cargo'))
    return payload


def test_writes_when_upstream_grew(baseline, monkeypatch):
    (baseline / 'equipment.json').write_bytes(_serve(monkeypatch, 10))
    before = (baseline / 'equipment.json').stat().st_size
    _serve(monkeypatch, 40)

    assert make_baseline.main([]) == 0
    assert (baseline / 'equipment.json').stat().st_size > before


def test_refuses_a_material_shrink(baseline, monkeypatch):
    original = _serve(monkeypatch, 100)
    (baseline / 'equipment.json').write_bytes(original)
    _serve(monkeypatch, 40)

    assert make_baseline.main([]) == 3
    assert (baseline / 'equipment.json').read_bytes() == original


def test_shrink_can_be_accepted_deliberately(baseline, monkeypatch):
    (baseline / 'equipment.json').write_bytes(_serve(monkeypatch, 100))
    _serve(monkeypatch, 40)

    assert make_baseline.main(['--allow-shrink']) == 0
    assert len(json.loads((baseline / 'equipment.json').read_text())) == 40


def test_check_mode_never_writes(baseline, monkeypatch):
    original = _serve(monkeypatch, 100)
    (baseline / 'equipment.json').write_bytes(original)
    _serve(monkeypatch, 40)

    assert make_baseline.main(['--check']) == 2      # reported stale
    assert (baseline / 'equipment.json').read_bytes() == original
