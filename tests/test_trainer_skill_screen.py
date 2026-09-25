"""WARP CORE shows skill-tree recognition read-only, and a skill screen whose
type the user has confirmed is marked Done on its own.

Node states are read by `skill_grid` with a fixed threshold — no model learns
from them, so WARP CORE only displays them. The screen type is the one thing on
a skill screen that trains a model (the screen classifier), which is why
auto-Done follows a *user* confirmation and never an ML guess.

`skill_grid` is stubbed here: its own reading is covered by
`test_skill_grid.py`; these tests cover what the trainer does with the result.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('PySide6')
Image = pytest.importorskip('PIL.Image')

from PySide6.QtWidgets import QApplication, QLabel, QPushButton  # noqa: E402


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def shot(tmp_path):
    p = tmp_path / 'skills.png'
    Image.fromarray(np.zeros((40, 60, 3), dtype=np.uint8)).save(p)
    return p


class _Canvas:
    def __init__(self):
        self.boxes = ['stale']
        self.locked = False

    def set_skill_boxes(self, boxes):
        self.boxes = list(boxes)

    def set_locked(self, locked):
        self.locked = locked


class _Stub:
    def __init__(self, path, stype='SPACE_SKILLS', done=False):
        self._ann_widget = _Canvas()
        self._review_summary = QLabel('previous text')
        self._screenshots = [path]
        self._current_idx = 0
        self._screen_types = {path.name: stype}
        self._screenshots_done = {path.name} if done else set()
        self._btn_done = QPushButton('✓ Mark Done')
        self._btn_done.setCheckable(True)
        self.calls = []

    def _save_done(self): self.calls.append('save_done')
    def _update_screen_type_ui(self, stype): self.calls.append('type_ui')
    def _update_add_bbox_btn(self): self.calls.append('bbox_btn')
    def _refresh_mark_done_btn(self): self.calls.append('done_btn')
    def _update_file_list_color(self, row): self.calls.append('color')

    class _FileList:
        def blockSignals(self, b): return False
    _file_list = _FileList()


def _show(stub, stype):
    from warp.trainer.trainer_window import WarpCoreWindow
    WarpCoreWindow._show_skill_recognition(stub, stub._screenshots[0], stype)


def _boxes(on_flags):
    return [(0, 0, 10, 10, on) for on in on_flags]


def _stub_grid(monkeypatch, env, boxes):
    from warp.recognition import skill_grid
    monkeypatch.setattr(skill_grid, 'env_of', lambda rgb: env)
    monkeypatch.setattr(skill_grid, 'detect_boxes',
                        lambda rgb, e: boxes if e else [])


def test_space_screen_draws_boxes_and_counts_per_career(app, shot, monkeypatch):
    boxes = _boxes([True] * 10 + [False] * 20      # eng 10
                   + [True] * 9 + [False] * 21     # sci 9
                   + [True] * 27 + [False] * 3)    # tac 27
    _stub_grid(monkeypatch, 'space', boxes)
    stub = _Stub(shot)

    _show(stub, 'SPACE_SKILLS')

    assert stub._ann_widget.boxes == boxes
    assert stub._review_summary.text() == (
        'Space skills ON — Eng 10/30 · Sci 9/30 · Tac 27/30')


def test_ground_screen_counts_per_tree(app, shot, monkeypatch):
    boxes = _boxes([True] * 6 + [False] * 6 + [False] * 4 + [True] * 4)
    _stub_grid(monkeypatch, 'ground', boxes)
    stub = _Stub(shot, 'GROUND_SKILLS')

    _show(stub, 'GROUND_SKILLS')

    assert stub._review_summary.text() == (
        'Ground skills ON per tree — 6/6 · 0/6 · 0/4 · 4/4')


def test_generic_skills_says_which_env_it_read(app, shot, monkeypatch):
    _stub_grid(monkeypatch, 'ground', _boxes([False] * 20))
    stub = _Stub(shot, 'SKILLS')

    _show(stub, 'SKILLS')

    assert stub._review_summary.text().endswith(
        '(read as ground from the grid shape)')


def test_missing_grid_is_said_not_left_blank(app, shot, monkeypatch):
    """A generic SKILLS screen whose grid is not found must tell the user
    what to do, not show an empty canvas with a stale summary."""
    _stub_grid(monkeypatch, None, [])
    stub = _Stub(shot, 'SKILLS')

    _show(stub, 'SKILLS')

    assert stub._ann_widget.boxes == []
    assert 'no skill node grid found' in stub._review_summary.text()


def test_other_screens_clear_the_overlay_and_keep_the_summary(app, shot):
    stub = _Stub(shot, 'SPACE_EQ')

    _show(stub, 'SPACE_EQ')

    assert stub._ann_widget.boxes == []
    assert stub._review_summary.text() == 'previous text'


def test_unreadable_image_reports_the_failure(app, tmp_path):
    stub = _Stub(tmp_path / 'missing.png')

    _show(stub, 'SPACE_SKILLS')

    assert stub._ann_widget.boxes == []
    assert stub._review_summary.text().startswith('Skill recognition failed')


# ── auto Mark Done ────────────────────────────────────────────────────

def _auto_done(stub, row=0):
    from warp.trainer.trainer_window import WarpCoreWindow
    WarpCoreWindow._auto_mark_done_skill(stub, row, stub._screenshots[row])


def test_auto_done_marks_locks_and_saves(app, shot):
    stub = _Stub(shot)

    _auto_done(stub)

    assert shot.name in stub._screenshots_done
    assert stub._ann_widget.locked
    assert stub._btn_done.isChecked()
    assert stub._btn_done.text() == '↩ Back to Edit'
    assert 'save_done' in stub.calls


def test_auto_done_on_another_row_leaves_the_open_screen_alone(app, shot):
    """The file-list checkbox can confirm a row that is not the one open."""
    stub = _Stub(shot)
    stub._current_idx = 5

    _auto_done(stub)

    assert shot.name in stub._screenshots_done
    assert not stub._ann_widget.locked
    assert not stub._btn_done.isChecked()


def test_auto_done_is_a_no_op_when_already_done(app, shot):
    stub = _Stub(shot, done=True)

    _auto_done(stub)

    assert stub.calls == []


def test_both_user_confirmation_paths_auto_done_skill_screens():
    """Wiring: the type menu and the file-list checkbox are the two ways a
    user confirms a type. Checked on source — driving either needs the whole
    window (data manager, classifier, file list)."""
    import inspect
    from warp.trainer.trainer_window import WarpCoreWindow

    for fn in (WarpCoreWindow._on_type_override_changed,
               WarpCoreWindow._on_file_item_changed):
        assert '_auto_mark_done_skill(' in inspect.getsource(fn), fn.__name__


def test_ml_type_guess_does_not_auto_done_skill_screens():
    """An ML-guessed type still needs a human look — it is the training
    signal — so the bulk screen-type detection must not mark skills Done."""
    import inspect
    from warp.trainer.trainer_window import WarpCoreWindow

    body = inspect.getsource(WarpCoreWindow._on_detect_finished)

    assert '_auto_mark_done_skill(' not in body


def test_every_recognition_display_path_shows_skills():
    """Load, type change and Auto-Detect all repaint the summary, so all
    three must re-show the skill result or it is overwritten."""
    import inspect
    from warp.trainer.trainer_window import WarpCoreWindow

    for fn in (WarpCoreWindow._load_screenshot,
               WarpCoreWindow._on_type_override_changed,
               WarpCoreWindow._on_recognition_done):
        assert '_show_skill_recognition(' in inspect.getsource(fn), fn.__name__


# ── canvas ────────────────────────────────────────────────────────────

@pytest.fixture
def canvas(app):
    from warp.trainer.annotation_widget import AnnotationWidget

    class _DataMgr:
        def get_annotations(self, path):
            return []

    w = AnnotationWidget(_DataMgr())
    yield w
    w.close()


def test_canvas_draws_skill_boxes(canvas, shot):
    canvas.load_image(shot)
    canvas.resize(200, 200)
    canvas.set_skill_boxes([(5, 5, 10, 10, True), (20, 5, 10, 10, False)])

    img = canvas.grab().toImage()          # runs paintEvent

    assert not img.isNull()
    assert len(canvas._skill_boxes) == 2


def test_loading_another_image_clears_skill_boxes(canvas, shot):
    """Boxes belong to one screenshot; the next one sets its own, if any."""
    canvas.load_image(shot)
    canvas.set_skill_boxes([(5, 5, 10, 10, True)])

    canvas.load_image(shot)

    assert canvas._skill_boxes == []
