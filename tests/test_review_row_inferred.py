"""The review row must show an inferred value as inferred.

`Ship Tier` can be worked out from the slots a ship shows when the `[T6-X2]`
badge is not in frame. It then flows downstream like any read value, so the
one place the difference has to be visible is the review row.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip('PySide6')

from warp.trainer.trainer_window import WarpCoreWindow  # noqa: E402

_visuals = WarpCoreWindow._review_row_visuals
_STUB = SimpleNamespace(_AUTO_COLOR='#c0ffee', _CONFIRMED_COLOR='#000000',
                        _CONFLICT_COLOR='#ff0000', _VIRTUAL_CONFIRMED='#111111',
                        _VIRTUAL_PENDING='#222222')


def _row(name='T6-X2', conf=1.0, **kw):
    kw.setdefault('confirmed', True)
    kw.setdefault('cross_check_failed', False)
    kw.setdefault('auto_confirmed', True)
    kw.setdefault('conflict_disk_name', '')
    return _visuals(_STUB, name, conf, **kw)


def test_inferred_row_is_labelled_inferred_not_auto():
    item_text, _conf, status, _color = _row(inferred=True)
    assert status == 'Inferred'
    assert item_text == 'T6-X2'


def test_a_normally_read_value_is_unaffected():
    _item, _conf, status, _color = _row(inferred=False)
    assert status == 'Auto'


def test_inferred_defaults_to_off_for_existing_callers():
    _item, _conf, status, _color = _row()
    assert status == 'Auto'


def test_a_conflict_still_wins_over_the_inferred_marker():
    _item, _conf, status, _color = _row(inferred=True,
                                        conflict_disk_name='T6')
    assert status == 'Conflict'
