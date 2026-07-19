"""WARP folds recognised skill trees into its SETS-build export + shows a
one-line skill summary after recognition (skill screens carry no items, so the
Results tree is empty for them).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")
Image = pytest.importorskip("PIL.Image")

from warp.gui.warp_window import WarpWindow


def _skill_result(tmp_path):
    s = tmp_path / 's.png'
    g = tmp_path / 'g.png'
    Image.new('RGB', (400, 400)).save(s)
    Image.new('RGB', (400, 300)).save(g)
    return SimpleNamespace(per_file_screen_type={
        str(s): 'SPACE_SKILLS', str(g): 'GROUND_SKILLS'})


def test_skill_summary_reports_space_and_ground(tmp_path):
    msg = WarpWindow._skill_summary(_skill_result(tmp_path))
    assert 'Space skills' in msg
    assert 'Ground skills' in msg


def test_skill_summary_empty_without_skill_screens():
    result = SimpleNamespace(per_file_screen_type={'/x.png': 'SPACE_EQ'})
    assert WarpWindow._skill_summary(result) == ''


def test_dup_warning_fires_on_two_same_env_skill_screens():
    result = SimpleNamespace(per_file_screen_type={
        '/a.png': 'SPACE_SKILLS', '/b.png': 'SPACE_SKILLS'})
    msg = WarpWindow._skill_dup_warning(result)
    assert '⚠' in msg and '2 space' in msg


def test_dup_warning_silent_on_one_each():
    result = SimpleNamespace(per_file_screen_type={
        '/s.png': 'SPACE_SKILLS', '/g.png': 'GROUND_SKILLS'})
    assert WarpWindow._skill_dup_warning(result) == ''
