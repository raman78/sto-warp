"""A trait name shared by space and ground shows the icon of the slot's side.

'Sniper' is a ground personal trait (Sniper.png, yellow) and a starship
trait (Sniper (starship).png, blue). The icon lookup returned the bare file
first whatever the slot, so a Starship Traits row showed the ground icon;
and it knew only the '(space)' / '(ground)' tags, not '(starship)' or
'(Space)'.

Run standalone:
    python -m pytest tests/test_env_icon_choice.py -v
"""
from __future__ import annotations

import pytest


@pytest.fixture
def icons(tmp_path, monkeypatch):
    from warp.data import cargo
    d = tmp_path / 'icons'
    d.mkdir()
    for f in ('Sniper.png', 'Sniper+%28starship%29.png',
              'Engineered+Soldier.png', 'Engineered+Soldier+%28Space%29.png',
              'Plain.png'):
        (d / f).write_bytes(b'\x89PNG')
    monkeypatch.setattr(cargo, 'icons_dir', lambda: d)
    monkeypatch.setattr(cargo, '_trait_icon_aliases', lambda: {
        'Sniper': ['Sniper (starship)'],
        'Engineered Soldier': ['Engineered Soldier (Space)'],
    })
    return cargo


def test_a_starship_slot_gets_the_starship_icon(icons):
    assert icons.ref_icon_path('Sniper', 'space').name == 'Sniper+%28starship%29.png'


def test_a_ground_slot_keeps_the_ground_icon(icons):
    assert icons.ref_icon_path('Sniper', 'ground').name == 'Sniper.png'


def test_the_tag_is_matched_whatever_its_case(icons):
    assert icons.ref_icon_path('Engineered Soldier', 'space').name == \
        'Engineered+Soldier+%28Space%29.png'


def test_a_name_with_one_icon_is_unaffected(icons):
    assert icons.ref_icon_path('Plain', 'space').name == 'Plain.png'


def test_the_tag_rule_reads_both_sides():
    from warp.data.cargo import env_tag_matches
    assert env_tag_matches('Sniper (starship)', 'space') is True
    assert env_tag_matches('Sniper (starship)', 'ground') is False
    assert env_tag_matches('Slippery Target (Lukari Reputation)', 'space') is None
