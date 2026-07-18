"""Tests for warp.recognition.skill_grid (space skill on/off recognition).

Self-contained: builds synthetic RGB arrays (vivid vs grey tiles at template
positions) so the suite needs only numpy — no cv2/torch, no sample files.
"""
from __future__ import annotations

import numpy as np
import pytest

from warp.recognition import skill_grid as sg

Image = pytest.importorskip("PIL.Image")


def test_template_has_90_space_positions():
    pos = sg._TEMPLATE['space']['positions']
    assert set(pos) == {'eng', 'sci', 'tac'}
    for career in ('eng', 'sci', 'tac'):
        assert len(pos[career]) == 30
        assert all(len(p) == 2 for p in pos[career])


def _blank(h=400, w=300):
    return np.zeros((h, w, 3), dtype=np.uint8)


def _paint(img, cx, cy, colour, r=13):
    img[cy - r:cy + r, cx - r:cx + r] = colour


def test_is_on_vivid_vs_grey():
    img = _blank()
    _paint(img, 100, 100, (240, 30, 30))   # vivid red -> ON
    _paint(img, 200, 100, (110, 110, 110))  # flat grey -> OFF
    s, v = sg._sat_val(img)
    assert sg._is_on(s, v, 100, 100) is True
    assert sg._is_on(s, v, 200, 100) is False


def test_detect_space_recovers_painted_pattern():
    # Extent chosen so every template position lands inside the image.
    x0, y0, x1, y1 = 20, 20, 280, 380
    w, h = x1 - x0, y1 - y0
    img = _blank(400, 300)
    pos = sg._TEMPLATE['space']['positions']

    # Turn ON every 3rd node per career; leave the rest grey (OFF).
    expected = {c: [False] * 30 for c in ('eng', 'sci', 'tac')}
    for career in ('eng', 'sci', 'tac'):
        for i, (nx, ny) in enumerate(pos[career]):
            cx = int(round(x0 + nx * w))
            cy = int(round(y0 + ny * h))
            if i % 3 == 0:
                _paint(img, cx, cy, (250, 20, 20))   # vivid -> ON
                expected[career][i] = True
            else:
                _paint(img, cx, cy, (100, 100, 100))  # grey -> OFF

    out = sg.detect_space(img, extent=(x0, y0, x1, y1))
    assert out == expected


def test_detect_space_no_anchor_returns_all_off():
    # Blank image: nothing to anchor -> safe all-OFF, no crash.
    out = sg.detect_space(_blank())
    assert out == {c: [False] * 30 for c in ('eng', 'sci', 'tac')}


def test_template_has_ground_trees():
    trees = sg._TEMPLATE['ground']['positions']
    assert [len(t) for t in trees] == [6, 6, 4, 4]


def test_detect_ground_recovers_painted_pattern():
    x0, y0, x1, y1 = 20, 20, 280, 380
    w, h = x1 - x0, y1 - y0
    img = _blank(400, 300)
    trees = sg._TEMPLATE['ground']['positions']

    expected = []
    for tree in trees:
        row = []
        for k, (nx, ny) in enumerate(tree):
            cx = int(round(x0 + nx * w))
            cy = int(round(y0 + ny * h))
            on = k % 2 == 0
            _paint(img, cx, cy, (250, 20, 20) if on else (100, 100, 100))
            row.append(on)
        expected.append(row)

    out = sg.detect_ground(img, extent=(x0, y0, x1, y1))
    assert out == expected


def test_detect_dispatch_ground_and_unknown():
    assert 'ground_skills' in sg.detect(_blank(), 'ground')
    with pytest.raises(ValueError):
        sg.detect(_blank(), 'nonsense')


def test_to_skill_tree_shape_matches_sets_schema():
    tree = sg.to_skill_tree()
    assert set(tree) == {'space_skills', 'ground_skills', 'skill_unlocks',
                         'skill_desc'}
    assert {k: len(v) for k, v in tree['space_skills'].items()} == \
        {'eng': 30, 'sci': 30, 'tac': 30}
    assert [len(t) for t in tree['ground_skills']] == [6, 6, 4, 4]
    assert {k: len(v) for k, v in tree['skill_unlocks'].items()} == \
        {'eng': 5, 'sci': 5, 'tac': 5, 'ground': 5}
    assert tree['skill_desc'] == {'space': '', 'ground': ''}


def _paint_template(img, positions, x0, y0, x1, y1):
    w, h = x1 - x0, y1 - y0
    for nx, ny in positions:
        cx, cy = int(x0 + nx * w), int(y0 + ny * h)
        img[cy - 9:cy + 9, cx - 9:cx + 9] = (250, 20, 20)


def test_detect_boxes_space_count_and_state():
    # Realistic extent so node sampling patches don't overlap neighbours.
    x0, y0, x1, y1 = 30, 30, 720, 1140
    w, h = x1 - x0, y1 - y0
    img = _blank(1160, 740)
    pos = sg._TEMPLATE['space']['positions']
    for career in ('eng', 'sci', 'tac'):
        for i, (nx, ny) in enumerate(pos[career]):
            cx, cy = int(x0 + nx * w), int(y0 + ny * h)
            _paint(img, cx, cy, (250, 20, 20) if i % 2 == 0 else (100, 100, 100))

    boxes = sg.detect_boxes(img, 'space', extent=(x0, y0, x1, y1))
    assert len(boxes) == 90
    assert all(len(b) == 5 for b in boxes)
    assert sum(b[4] for b in boxes) == 45   # 15 even-index ON per career


def test_detect_boxes_ground_count():
    boxes = sg.detect_boxes(_blank(400, 300), 'ground', extent=(20, 20, 280, 380))
    assert len(boxes) == 20
    assert all(not b[4] for b in boxes)     # blank -> all OFF


def test_env_of_distinguishes_space_and_ground():
    space_pos = [p for c in ('eng', 'sci', 'tac')
                 for p in sg._TEMPLATE['space']['positions'][c]]
    ground_pos = [p for tree in sg._TEMPLATE['ground']['positions'] for p in tree]
    simg = _blank(620, 420)
    _paint_template(simg, space_pos, 40, 40, 380, 580)      # tall extent
    gimg = _blank(420, 620)
    _paint_template(gimg, ground_pos, 40, 40, 580, 300)     # wide extent
    assert sg.env_of(simg) == 'space'
    assert sg.env_of(gimg) == 'ground'
    assert sg.env_of(_blank()) is None


def test_skills_from_files_routes_by_screen_type(tmp_path):
    space = tmp_path / 's.png'
    ground = tmp_path / 'g.png'
    Image.new('RGB', (400, 400)).save(space)
    Image.new('RGB', (400, 300)).save(ground)
    out = sg.skills_from_files({
        str(space): 'SPACE_SKILLS',
        str(ground): 'GROUND_SKILLS',
        '/nope.png': 'SPACE_EQ',      # non-skill type ignored
    })
    assert set(out) == {'space_skills', 'ground_skills'}
    assert [len(t) for t in out['ground_skills']] == [6, 6, 4, 4]
    assert set(out['space_skills']) == {'eng', 'sci', 'tac'}


def test_to_skill_tree_overlays_recognised_envs():
    space = {'eng': [True] * 30, 'sci': [False] * 30, 'tac': [False] * 30}
    ground = [[True] * 6, [False] * 6, [False] * 4, [True] * 4]
    tree = sg.to_skill_tree(space_skills=space, ground_skills=ground)
    assert tree['space_skills'] is space
    assert tree['ground_skills'] is ground
    # untouched env keeps empty defaults (None unlocks -> SETS fills them)
    assert tree['skill_unlocks']['eng'] == [None] * 5
