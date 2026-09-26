"""Pick the right item from the crop's look-alikes, or search the whole group.

A wrongly recognised slot used to mean opening vger in a browser and
comparing icons by eye. The picker shows the crop beside the items of that
slot most similar to it, and a name search lists the whole group for when
none of the close ones is right. Choosing one confirms it exactly as typing
the name would.

Run standalone:
    python -m pytest tests/test_pick_icon_dialog.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip('cv2')
pytest.importorskip('torch')
pytest.importorskip('PySide6')


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    monkeypatch.setenv('WARP_LOG_DIR', str(tmp_path / 'logs'))
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'config'))
    monkeypatch.setenv('XDG_DATA_HOME', str(tmp_path / 'data'))
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path / 'cache'))


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


CROP = np.random.default_rng(5).integers(0, 256, (44, 35, 3), dtype=np.uint8)


# ── The ranking ────────────────────────────────────────────────────────────

class _Embedder:
    def __init__(self, v):
        self.v = np.asarray(v, dtype=np.float32)

    def __call__(self, _t):
        import torch
        return torch.from_numpy((self.v / np.linalg.norm(self.v))[None, :])


def _matcher(query, gallery):
    from warp.recognition.icon_matcher import SETSIconMatcher
    m = SETSIconMatcher()
    m._index = []
    names = list(gallery)
    emb = np.asarray([gallery[n] for n in names], dtype=np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    m._gallery_emb, m._gallery_lbl = emb, np.arange(len(names), dtype=np.int32)
    m._gallery_is_art = np.zeros(len(names), dtype=bool)
    m._label_map = dict(enumerate(names))
    m._ml_kind, m._ml_session = 'embedder', _Embedder(query)
    return m


def test_every_candidate_comes_back_closest_first():
    m = _matcher([1, 0, 0], {'Near': [0.9, 0.1, 0], 'Far': [0, 1, 0], 'Mid': [0.5, 0.5, 0]})
    ranked, _tm = m.rank_candidates(CROP, {'Near', 'Far', 'Mid'})

    assert [n for n, _s in ranked] == ['Near', 'Mid', 'Far']


def test_only_the_slots_items_are_ranked():
    """The candidate set is the slot's item list; nothing else may appear."""
    m = _matcher([1, 0, 0], {'Near': [1, 0, 0], 'Other slot': [1, 0, 0]})
    ranked, _tm = m.rank_candidates(CROP, {'Near'})

    assert [n for n, _s in ranked] == ['Near']


def test_an_item_with_no_gallery_row_is_still_offered():
    """The point of the search is the item nobody has confirmed yet."""
    m = _matcher([1, 0, 0], {'Near': [1, 0, 0]})
    ranked, _tm = m.rank_candidates(CROP, {'Near', 'Never seen'})

    assert {n for n, _s in ranked} == {'Near', 'Never seen'}


# ── The dialog ─────────────────────────────────────────────────────────────

def _dialog(qapp, n=75):
    from warp.trainer.pick_icon_dialog import PickIconDialog
    ranked = [(f'Beam Array {i:02d}', 1 - i / 100) for i in range(n - 1)]
    ranked.append(('Quantum Torpedo Launcher', 0.05))
    return PickIconDialog(CROP, 'Fore Weapons', ranked, lambda _n: None,
                          current='Beam Array 07')


def test_it_opens_on_the_closest_page(qapp):
    from warp.trainer.pick_icon_dialog import PAGE
    d = _dialog(qapp)

    assert d.shown_names()[:2] == ['Beam Array 00', 'Beam Array 01']
    assert len(d.shown_names()) == PAGE


def test_show_more_extends_it(qapp):
    from warp.trainer.pick_icon_dialog import PAGE
    d = _dialog(qapp)
    d._show_more()

    assert len(d.shown_names()) == 2 * PAGE


def test_the_search_reaches_items_far_down_the_ranking(qapp):
    """When none of the close ones is right: the whole group, by name."""
    d = _dialog(qapp)
    d._search.setText('quantum torpedo')

    assert d.shown_names() == ['Quantum Torpedo Launcher']


def test_choosing_a_tile_returns_its_name(qapp):
    d = _dialog(qapp)
    d._grid.setCurrentRow(3)
    d._use_selected()

    assert d.chosen == 'Beam Array 03'


# ── The window ─────────────────────────────────────────────────────────────

def test_a_pick_confirms_the_row_like_typing_the_name(qapp, monkeypatch):
    from PySide6.QtWidgets import QDialog, QLineEdit, QWidget
    import warp.trainer.trainer_window as tw
    import warp.trainer.pick_icon_dialog as pd

    class _Matcher:
        def __init__(self, *_a):
            pass

        def rank_candidates(self, crop, names):
            return sorted((n, 0.5) for n in names), None

        def _thumb_for_name(self, *_a):
            return None

    monkeypatch.setattr('warp.recognition.icon_matcher.SETSIconMatcher', _Matcher)

    def _exec(dlg):
        dlg.chosen = 'Advanced Phaser Beam Array'
        return QDialog.DialogCode.Accepted
    monkeypatch.setattr(pd.PickIconDialog, 'exec', _exec)

    accepted = []

    class _Win(QWidget):
        _recognition_items = [{'slot': 'Aft Weapons', 'name': 'Transphasic Mine Launcher',
                               'bbox': (0, 0, 35, 44), 'crop_bgr': CROP}]
        _current_idx = -1
        _sets = None

        def __init__(self):
            super().__init__()
            self._name_edit = QLineEdit()
            self._review_list = type('L', (), {'setCurrentRow': lambda s, r: None})()

        def _build_search_candidates(self, slot=''):
            return ['Advanced Phaser Beam Array', 'Transphasic Mine Launcher']

        def statusBar(self):
            return type('B', (), {'showMessage': lambda *a: None})()

        def _on_accept(self, *a, auto=False):
            accepted.append((self._name_edit.text(), auto))

        _open_pick_dialog = tw.WarpCoreWindow._open_pick_dialog

    _Win()._open_pick_dialog(0)

    assert accepted == [('Advanced Phaser Beam Array', False)]


def test_the_canvas_asks_the_window_for_its_menu(qapp):
    """One menu for the list and the canvas; the canvas kept its own copy,
    which offered links only and nothing for an unrecognised box."""
    from PySide6.QtCore import QPoint
    from warp.trainer.annotation_widget import AnnotationWidget
    w = AnnotationWidget.__new__(AnnotationWidget)
    AnnotationWidget.__bases__[0].__init__(w)
    w._review_items = [{'slot': 'Devices', 'name': '', 'bbox': (0, 0, 10, 10)}]
    w._hit_test_review = lambda pos: 0
    seen = []
    w.context_menu_requested.connect(lambda row, gp: seen.append(row))

    class _Ev:
        def pos(self):
            return QPoint(1, 1)

        def globalPos(self):
            return QPoint(5, 5)
    w.contextMenuEvent(_Ev())

    assert seen == [0]
