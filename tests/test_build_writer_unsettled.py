"""The SETS build leaves out items whose equipment row is not settled.

An item in a row the detector could only guess, or not type at all
(RecognisedItem.row_guess), would land in SETS in a slot nobody confirmed.
It is left out and counted in WriteReport.unsettled_items, which the WARP
export message reports; the user settles the row in Fast Correction.

Offline: the bundled cargo baseline.
"""
from __future__ import annotations

import pytest

from warp import build_writer as bw
from warp.warp_importer import ImportResult, RecognisedItem


@pytest.fixture(autouse=True)
def _offline(monkeypatch, tmp_path):
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'cfg'))
    def _no_fetch(name):
        raise OSError('offline test')
    monkeypatch.setattr('warp.data.cargo._fetch', _no_fetch)


def _device(row_guess=''):
    return RecognisedItem(slot='Devices', slot_index=0, name='Beacon of Kahless',
                          confidence=0.95, row_guess=row_guess)


def test_a_device_in_a_settled_row_is_written():
    _, report = bw.build_from_result(ImportResult(build_type='SPACE', items=[_device()]))
    assert report.n_equipment == 1 and report.unsettled_items == 0


@pytest.mark.parametrize('row_guess', ['guess', 'unknown'])
def test_a_device_in_an_unsettled_row_is_left_out_and_counted(row_guess):
    _, report = bw.build_from_result(
        ImportResult(build_type='SPACE', items=[_device(row_guess)]))
    assert report.n_equipment == 0 and report.unsettled_items == 1
