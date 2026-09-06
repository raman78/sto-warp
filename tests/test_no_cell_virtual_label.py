"""A virtual label is only worth what the panel supports.

`_classify_cell` reports what is *in* a cell and takes for granted that the
crop is one. Where an equipment row does not reach, it answers `empty` — and
the importer reported that at confidence 1.00, which the trainer auto-accepts
at 0.75. So a patch of bare panel entered the training data as an example of
an empty slot. Measured over 1596 such positions: 95% came back as a slot.

The box is not dropped. A box nobody is shown is a mistake nobody can
correct, and the user deleting it is worth more than its silent absence — so
it is reported, loudly, at a confidence that keeps it away from auto-accept.

Offline: synthetic panels, no screenshot and no ML.
"""
from __future__ import annotations

import types

import numpy as np
import pytest

from warp.warp_importer import (_EQ_GRID_SLOTS, _NO_CELL_VIRTUAL_CONF,
                                WarpImporter)

CELL_W, CELL_H, DX = 35, 45, 37.0


def _panel(w=400, h=90, fill=40):
    return np.full((h, w, 3), fill, np.uint8)


def _with_cell(x, y=10):
    img = _panel()
    img[y:y + CELL_H, x:x + CELL_W] = 3
    img[y, x:x + CELL_W] = 200
    img[y + CELL_H - 1, x:x + CELL_W] = 200
    img[y:y + CELL_H, x] = 200
    img[y:y + CELL_H, x + CELL_W - 1] = 200
    return img


def _importer(dx=DX):
    imp = WarpImporter.__new__(WarpImporter)
    geom = types.SimpleNamespace(final_dx=dx)
    imp._get_layout = lambda: types.SimpleNamespace(
        _get_eq_geometry=lambda img: geom)
    return imp


# ── The check itself ──────────────────────────────────────────────────────

def test_a_drawn_cell_is_accepted():
    assert _importer()._cell_is_really_there(
        _with_cell(200), (200, 10, CELL_W, CELL_H), 'Science Consoles')


def test_bare_panel_is_refused():
    assert not _importer()._cell_is_really_there(
        _panel(), (200, 10, CELL_W, CELL_H), 'Science Consoles')


# ── Only where it was measured ────────────────────────────────────────────

def test_panels_outside_the_measured_corpus_are_left_alone():
    """`_cell_exists` was measured on the space equipment grid. The BOFF tray
    and the trait panels draw their empty cells differently and were not in
    that corpus."""
    for slot in ('Boff Tactical', 'Personal Space Traits', 'Space Reputation'):
        assert _importer()._cell_is_really_there(
            _panel(), (200, 10, CELL_W, CELL_H), slot)


def test_every_equipment_row_is_covered():
    for slot in ('Fore Weapons', 'Aft Weapons', 'Devices', 'Hangars',
                 'Universal Consoles', 'Science Consoles'):
        assert slot in _EQ_GRID_SLOTS


def test_no_geometry_means_no_opinion():
    """Answering yes when it cannot tell is the safe direction: the caller
    only uses this to lower a confidence."""
    imp = WarpImporter.__new__(WarpImporter)
    imp._get_layout = lambda: types.SimpleNamespace(
        _get_eq_geometry=lambda img: None)
    assert imp._cell_is_really_there(
        _panel(), (200, 10, CELL_W, CELL_H), 'Science Consoles')


def test_a_broken_geometry_lookup_does_not_stop_recognition():
    imp = WarpImporter.__new__(WarpImporter)

    def _boom():
        raise RuntimeError('no geometry today')

    imp._get_layout = _boom
    assert imp._cell_is_really_there(
        _panel(), (200, 10, CELL_W, CELL_H), 'Science Consoles')


# ── The confidence has to keep it away from auto-accept ───────────────────

def test_the_reported_confidence_is_below_the_auto_accept_bar():
    """The trainer accepts at 0.75 without asking. A label the panel does not
    support must not clear that."""
    assert _NO_CELL_VIRTUAL_CONF < 0.75


def test_it_is_below_the_low_confidence_warning_too():
    """0.35 is where the trainer starts calling a row out, which is exactly
    what this row needs."""
    assert _NO_CELL_VIRTUAL_CONF < 0.35


def test_it_is_not_zero():
    """Zero reads as "no opinion" elsewhere in the pipeline; this is an
    opinion, just a weak one."""
    assert _NO_CELL_VIRTUAL_CONF > 0
