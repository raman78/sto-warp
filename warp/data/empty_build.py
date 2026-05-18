"""SETS v3.0.0 `empty_build` skeleton — port of `src.datafunctions.empty_build`.

Used by `warp.build_writer` to seed a SETS-loadable build dict that we
populate from an `ImportResult`. Kept SETS-free so this module can sit
under `warp.data` alongside the cargo loaders.

Output schema matches upstream SETS v3.0.0 tag exactly so the resulting
JSON loads cleanly via SETS' `File → Load Build` after we pipe it
through `warp.sets_export.write_sets_build` (which adds `_version` and
normalises BOFF ability ranks).
"""
from __future__ import annotations


def empty_build(build_type: str = 'full') -> dict:
    new_build = {
        'space': {
            'active_rep_traits': [None] * 5,
            'aft_weapons':       [None] * 5,
            'boffs':             [[None] * 4 for _ in range(6)],
            'boff_specs':        [[None, None] for _ in range(6)],
            'core':              [''],
            'deflector':         [''],
            'devices':           [None] * 6,
            'doffs_spec':        [''] * 6,
            'doffs_variant':     [''] * 6,
            'eng_consoles':      [None] * 5,
            'engines':           [''],
            'experimental':      [None],
            'fore_weapons':      [None] * 5,
            'hangars':           [None] * 2,
            'rep_traits':        [None] * 5,
            'sci_consoles':      [None] * 5,
            'sec_def':           [None],
            'shield':            [''],
            'ship':              '',
            'ship_name':         '',
            'ship_desc':         '',
            'starship_traits':   [None] * 7,
            'tac_consoles':      [None] * 5,
            'tier':              '',
            'traits':            ['', '', '', '', '', '', '', '', '', None, None, ''],
            'uni_consoles':      [None] * 3,
        },
        'ground': {
            'active_rep_traits': [None] * 5,
            'armor':             [''],
            'boffs':             [[''] * 4 for _ in range(4)],
            'boff_profs':        ['Tactical'] * 4,
            'boff_specs':        ['Command'] * 4,
            'ground_desc':       '',
            'ground_devices':    ['', '', '', '', None],
            'doffs_spec':        [''] * 6,
            'doffs_variant':     [''] * 6,
            'ev_suit':           [''],
            'kit':               [''],
            'kit_modules':       ['', '', '', '', '', None],
            'rep_traits':        [''] * 5,
            'personal_shield':   [''],
            'traits':            ['', '', '', '', '', '', '', '', '', None, None, ''],
            'weapons':           [''] * 2,
        },
        'captain': {
            'career': '', 'elite': False, 'faction': '', 'name': '',
            'primary_spec': '', 'secondary_spec': '', 'species': '',
        },
    }
    new_skills = {
        'space_skills':  {'eng': [False] * 30, 'sci': [False] * 30, 'tac': [False] * 30},
        'skill_unlocks': {'eng': [None] * 5, 'sci': [None] * 5,
                          'tac': [None] * 5, 'ground': [None] * 5},
        'ground_skills': [[False] * 6, [False] * 6, [False] * 4, [False] * 4],
        'skill_desc':    {'space': '', 'ground': ''},
    }
    if build_type == 'build':
        return new_build
    if build_type == 'skills':
        return new_skills
    new_build.update(new_skills)
    return new_build
