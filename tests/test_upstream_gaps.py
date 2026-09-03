"""The ledger of items SETS drops must keep the two reasons apart.

An item WARP recognises can fail to survive a SETS import for two unrelated
reasons, and they lead to two different projects: no cargo table stores it
(ask the wiki) or SETS's own loader passes over the row (ask SETS). Merged
into one list they produce a request nobody can act on.

Run standalone:
    python -m pytest tests/test_upstream_gaps.py -v
"""
from __future__ import annotations

import json

import pytest


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    """The ledger lives under the user's data dir — keep the tests out of it."""
    monkeypatch.setenv('XDG_DATA_HOME', str(tmp_path / 'data'))
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'config'))
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path / 'cache'))
    monkeypatch.setenv('WARP_LOG_DIR', str(tmp_path / 'logs'))


def _violation(path: str, got: str, rule: str = 'not_in_sets'):
    from warp.sets_schema import Violation

    return Violation(path, rule, 'an item SETS can resolve', got)


def _ledger() -> dict:
    from warp import upstream_gaps

    return json.loads(upstream_gaps.ledger_path().read_text(encoding='utf-8'))


def test_an_overlay_item_is_recorded_as_a_wiki_gap():
    from warp import upstream_gaps

    upstream_gaps.record([
        _violation('/ground/ground_weapons[0]',
                   "'Elite Fleet Phaser Pulsewave' (scraped)"),
    ])

    entry = next(iter(_ledger().values()))
    assert entry['reason'] == upstream_gaps.REASON_CARGO
    assert entry['name'] == 'Elite Fleet Phaser Pulsewave'


def test_a_skipped_hangar_is_recorded_as_a_sets_gap():
    from warp import upstream_gaps

    upstream_gaps.record([
        _violation('/space/hangars[0]',
                   "'Hangar - Elite Peregrine' (SETS loader skips it)"),
    ])

    entry = next(iter(_ledger().values()))
    assert entry['reason'] == upstream_gaps.REASON_LOADER


def test_the_two_reasons_never_share_an_entry():
    """The same name under both reasons is two separate upstream requests."""
    from warp import upstream_gaps

    upstream_gaps.record([
        _violation('/space/hangars[0]', "'Ambiguous Item' (scraped)"),
        _violation('/space/hangars[0]',
                   "'Ambiguous Item' (SETS loader skips it)"),
    ])

    reasons = {e['reason'] for e in _ledger().values()}
    assert reasons == {upstream_gaps.REASON_CARGO, upstream_gaps.REASON_LOADER}


def test_repeated_exports_accumulate_rather_than_overwrite():
    """The count is the whole argument — one build proves nothing."""
    from warp import upstream_gaps

    v = _violation('/ground/ground_weapons[0]', "'Colony Rifle' (scraped)")
    for _ in range(3):
        upstream_gaps.record([v])

    assert next(iter(_ledger().values()))['exports'] == 3


def test_every_slot_an_item_appeared_in_is_kept():
    from warp import upstream_gaps

    upstream_gaps.record([
        _violation('/ground/ground_weapons[0]', "'Colony Rifle' (scraped)"),
        _violation('/ground/kit_modules[2]', "'Colony Rifle' (scraped)"),
    ])

    slots = next(iter(_ledger().values()))['slots']
    assert slots == ['ground/ground_weapons', 'ground/kit_modules']


def test_other_violations_are_not_recorded():
    """Only `not_in_sets` describes an upstream gap. The rest are WARP's own
    bugs and belong in the log, not in a request to another project."""
    from warp import upstream_gaps

    n = upstream_gaps.record([
        _violation('/space/fore_weapons[0]', "'Typo Beam'", rule='unknown_item'),
    ])

    assert n == 0
    assert not upstream_gaps.ledger_path().exists()


def test_a_broken_ledger_does_not_break_the_export():
    """Bookkeeping must never cost the user their build."""
    from warp import upstream_gaps

    path = upstream_gaps.ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('not json at all', encoding='utf-8')

    upstream_gaps.record([
        _violation('/ground/ground_weapons[0]', "'Colony Rifle' (scraped)"),
    ])

    assert next(iter(_ledger().values()))['exports'] == 1


def test_the_summary_names_which_project_to_ask():
    from warp import upstream_gaps

    upstream_gaps.record([
        _violation('/ground/ground_weapons[0]', "'Colony Rifle' (scraped)"),
        _violation('/space/hangars[0]',
                   "'Hangar - Elite Peregrine' (SETS loader skips it)"),
    ])

    text = upstream_gaps.summary()

    assert 'ask the wiki' in text
    assert 'ask SETS' in text
    assert 'Colony Rifle' in text
    assert 'Hangar - Elite Peregrine' in text


def test_an_empty_ledger_says_so_rather_than_failing():
    from warp import upstream_gaps

    assert 'No items' in upstream_gaps.summary()


# ── The call site ──────────────────────────────────────────────────────────
#
# The mapping above is worth nothing if the exporter never calls it, so drive
# the real export path once.

def test_exporting_a_build_records_its_gaps(tmp_path, monkeypatch):
    from warp import sets_export, upstream_gaps

    recorded = []
    monkeypatch.setattr(upstream_gaps, 'record',
                        lambda violations: recorded.append(violations))
    monkeypatch.setattr(sets_export, 'build_sets_v3_dict',
                        lambda build, cache: {'stub': True})
    monkeypatch.setattr(sets_export, 'validate_sets_build',
                        lambda payload, cache: [
                            _violation('/space/hangars[0]',
                                       "'Hangar - Elite Peregrine' "
                                       '(SETS loader skips it)')])

    sets_export.write_sets_build({}, tmp_path / 'build.json', cache=None)

    assert recorded and recorded[0][0].rule == 'not_in_sets'
