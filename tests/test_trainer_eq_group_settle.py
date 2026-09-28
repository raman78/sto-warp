"""WARP CORE: equipment rows whose type is a guess or Unknown must be settled.

The detector names an equipment row by a guess, or calls it Unknown, when
nothing establishes its type (RecognisedItem.row_guess). The review list
shows such a group as "<slot> (guess)" or "Unknown", in its place on the
panel; its items cannot be accepted — by hand or by auto-accept — until the
user confirms or changes the group's type from its right-click menu; and
Mark Done says which groups are still open.

Offscreen Qt, a real window, no model: the matcher is replaced where a type
change rematches.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip('PySide6')
cv2 = pytest.importorskip('cv2')

from PySide6.QtWidgets import QApplication

from warp.trainer import trainer_window as tw


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setattr('warp.knowledge.sync_client.WARPSyncClient.refresh_knowledge',
                        lambda self: None)
    monkeypatch.setattr('warp.trainer.model_updater.ModelUpdater.check_and_update',
                        lambda self, *a, **kw: None)
    # With a fresh XDG dir there is no cargo cache, and anything that looks a
    # name up would fetch it from GitHub. Refuse, so the bundled baseline
    # answers — whichever path asks.
    def _no_fetch(name):
        raise OSError('offline test')
    monkeypatch.setattr('warp.data.cargo._fetch', _no_fetch)


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication([])


def _item(slot, y, x=160, name='', row_guess='', conf=0.95):
    return {'name': name, 'slot': slot, 'conf': conf, 'bbox': (x, y, 33, 42),
            'state': 'pending', 'thumb': None, 'crop_bgr': None, 'orig_name': name,
            'ship_name': '', 'cross_check_failed': False, 'seat_key': '',
            'slot_index': 0, 'src': 'embed', 'row_guess': row_guess}


@pytest.fixture
def win(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'cfg'))
    monkeypatch.setenv('XDG_DATA_HOME', str(tmp_path / 'data'))
    shot = tmp_path / 'panel.png'
    cv2.imwrite(str(shot), np.zeros((620, 240, 3), np.uint8))
    w = tw.WarpCoreWindow()
    w._screenshots = [shot]
    w._screen_types = {shot.name: 'SPACE_MIXED'}
    w._current_idx = 0
    items = [
        _item('Fore Weapons', 60, name='Phaser Dual Heavy Cannons'),
        _item('Aft Weapons', 300, name='Phaser Turret'),
        _item('Devices', 348, name='Beacon of Kahless', row_guess='guess'),
        _item('Unknown', 396, name='Console - Universal - M6 Computer', row_guess='unknown'),
        _item('Tactical Consoles', 444, name='Console - Tactical - Bellum Prefire Chamber'),
    ]
    w._populate_review_panel(items, 'SPACE_MIXED')
    yield w
    w.close()


def _heading(win, key):
    return win._review_list._slot_parents[key].text(0)


def _row_of(win, slot):
    return next(i for i, ri in enumerate(win._recognition_items) if ri['slot'] == slot)


def test_a_guessed_group_says_so_and_an_unknown_one_is_called_unknown(win):
    assert _heading(win, 'Devices') == 'Devices (guess)'
    assert _heading(win, 'Unknown') == 'Unknown'


def test_an_unknown_group_sits_where_its_row_is_not_at_the_end(win):
    order = [win._review_list.topLevelItem(i).data(0, 256)
             for i in range(win._review_list.topLevelItemCount())]
    assert order.index('Unknown') == order.index('Devices') + 1


def test_an_item_in_an_unsettled_row_cannot_be_accepted(win):
    for slot in ('Devices', 'Unknown'):
        row = _row_of(win, slot)
        win._review_list.setCurrentRow(row)
        win._on_accept()
        assert win._recognition_items[row]['state'] == 'pending'


def test_auto_accept_leaves_unsettled_rows_alone(win):
    win._apply_auto_accept()
    assert win._recognition_items[_row_of(win, 'Devices')]['state'] == 'pending'
    assert win._recognition_items[_row_of(win, 'Unknown')]['state'] == 'pending'


def test_mark_done_names_the_groups_still_to_settle(win):
    win._refresh_mark_done_btn()
    tip = win._btn_done.toolTip()
    assert 'Devices' in tip and 'Unknown' in tip


def test_confirming_a_guessed_type_settles_the_row(win):
    parent = win._review_list._slot_parents['Devices']
    win._settle_eq_group(parent, 'Devices', 'Devices')
    ri = win._recognition_items[_row_of(win, 'Devices')]
    assert ri['row_guess'] == '' and ri['name'] == 'Beacon of Kahless'
    assert _heading(win, 'Devices') == 'Devices'


def test_choosing_a_type_for_an_unknown_row_rematches_and_moves_it(win, monkeypatch):
    class _Matcher:
        def __init__(self, *a, **k):
            pass

        def match(self, crop, candidate_names=None):
            return 'Console - Universal - M6 Computer', 0.9, None, False
    monkeypatch.setattr('warp.recognition.icon_matcher.SETSIconMatcher', _Matcher)
    monkeypatch.setattr(win, '_build_search_candidates',
                        lambda slot: ['Console - Universal - M6 Computer'])
    parent = win._review_list._slot_parents['Unknown']
    win._settle_eq_group(parent, 'Unknown', 'Universal Consoles')
    ri = win._recognition_items[_row_of(win, 'Universal Consoles')]
    assert ri['row_guess'] == '' and ri['name'] == 'Console - Universal - M6 Computer'
    assert 'Unknown' not in win._review_list._slot_parents
