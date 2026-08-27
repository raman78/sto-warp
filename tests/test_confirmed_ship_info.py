"""A ship class or tier the user confirmed outranks OCR — in WARP CORE only.

The tier is not a label, it is a slot count: `_apply_ship_and_tier_bonuses`
grants +1 Universal Console, Device and Starship Trait for `-X` and +2 for
`-X2`. Before this existed, correcting a misread tier in the review panel
changed nothing — the confirmed row survived the trainer's merge, but the
grid had already been sized from the OCR value, so the same surplus rows came
back on every Auto-Detect.

`annotations.json` is training data. Reading it here is the `_use_confirmed`
gate (trainer calls only), not a detection fallback: WARP proper never takes
this path, and what OCR read is still logged.
"""
from __future__ import annotations

import hashlib
import json

import pytest

from warp.warp_importer import WarpImporter


@pytest.fixture
def store(tmp_path, monkeypatch):
    """An annotations.json under a redirected XDG_DATA_HOME, plus a shot."""
    monkeypatch.setenv('XDG_DATA_HOME', str(tmp_path / 'data'))
    shot = tmp_path / 'screenshot.png'
    shot.write_bytes(b'\x89PNG\r\n\x1a\n' + b'pretend pixels')
    sha16 = hashlib.sha256(shot.read_bytes()).hexdigest()[:16]

    def _write(payload: dict):
        from warp import userdata
        path = userdata.training_data_dir() / 'annotations.json'
        path.write_text(json.dumps(payload), encoding='utf-8')

    return shot, sha16, _write


def _ann(slot, name, state='confirmed', auto=False):
    return {'slot': slot, 'name': name, 'state': state,
            'auto_confirmed': auto, 'bbox': [1, 2, 3, 4]}


def _importer() -> WarpImporter:
    return WarpImporter.__new__(WarpImporter)


# ── Finding the rows at all ────────────────────────────────────────────

def test_the_current_key_scheme_is_read(store):
    """Keyed by content hash, rows wrapped in `annotations` — 168 of 178
    entries in one maintainer's store are in this form, and the older
    loaders miss every one of them."""
    shot, sha16, write = store
    write({sha16: {'filename': 'whatever.png',
                   'annotations': [_ann('Ship Tier', 'T6-X')]}})

    assert _importer()._load_confirmed_ship_info(str(shot)) == {'Ship Tier': 'T6-X'}


def test_the_legacy_key_scheme_still_works(store):
    """Entries written before the hash keying are keyed by filename and hold
    the bare list."""
    shot, _sha16, write = store
    write({shot.name: [_ann('Ship Tier', 'T6-X')]})

    assert _importer()._load_confirmed_ship_info(str(shot)) == {'Ship Tier': 'T6-X'}


def test_the_content_hash_wins_over_a_shared_filename(store):
    """Two screenshots can share a name; handing one's ground truth to the
    other would be worse than having none."""
    shot, sha16, write = store
    write({
        sha16:     {'filename': shot.name, 'annotations': [_ann('Ship Tier', 'T6-X')]},
        shot.name: [_ann('Ship Tier', 'T5-U')],
    })

    assert _importer()._load_confirmed_ship_info(str(shot))['Ship Tier'] == 'T6-X'


def test_an_unknown_image_yields_nothing(store):
    shot, sha16, write = store
    write({'0' * 16: {'filename': 'other.png',
                      'annotations': [_ann('Ship Tier', 'T6-X2')]}})

    assert _importer()._load_confirmed_ship_info(str(shot)) == {}


def test_a_missing_store_is_not_an_error(store):
    shot, _sha16, _write = store

    assert _importer()._load_confirmed_ship_info(str(shot)) == {}


# ── Whose confirmation counts ──────────────────────────────────────────

def test_an_auto_confirmed_row_is_not_ground_truth(store):
    """Auto-confirm is the detector accepting its own answer on a threshold.
    Feeding it back would let a misread tier confirm itself and then defend
    the slot count it invented."""
    shot, sha16, write = store
    write({sha16: {'filename': shot.name,
                   'annotations': [_ann('Ship Tier', 'T6-X2', auto=True)]}})

    assert _importer()._load_confirmed_ship_info(str(shot)) == {}


def test_a_pending_row_is_not_ground_truth(store):
    shot, sha16, write = store
    write({sha16: {'filename': shot.name,
                   'annotations': [_ann('Ship Tier', 'T6-X', state='pending')]}})

    assert _importer()._load_confirmed_ship_info(str(shot)) == {}


def test_both_fields_are_returned_and_nothing_else(store):
    shot, sha16, write = store
    write({sha16: {'filename': shot.name, 'annotations': [
        _ann('Ship Type', 'Mirror Engle Strike Wing Escort'),
        _ann('Ship Tier', 'T6-X'),
        _ann('Ship Name', 'U.S.S. Whatever'),
        _ann('Fore Weapons', 'Phaser Beam Array'),
    ]}})

    assert _importer()._load_confirmed_ship_info(str(shot)) == {
        'Ship Type': 'Mirror Engle Strike Wing Escort',
        'Ship Tier': 'T6-X',
    }


def test_an_empty_name_is_not_a_correction(store):
    """A confirmed bbox with no text says where the tier is, not what it is."""
    shot, sha16, write = store
    write({sha16: {'filename': shot.name,
                   'annotations': [_ann('Ship Tier', '   ')]}})

    assert _importer()._load_confirmed_ship_info(str(shot)) == {}


# ── What it is worth ───────────────────────────────────────────────────

def test_the_layout_and_profile_loaders_see_the_same_rows(store):
    """Both used to key on the filename alone while the store had moved to
    content hashes — the confirmed layout was silently not applied."""
    shot, sha16, write = store
    write({sha16: {'filename': shot.name, 'annotations': [
        _ann('Fore Weapons', 'Phaser Beam Array'),
        _ann('Fore Weapons', 'Phaser Beam Array'),
        _ann('Deflector', 'Deflector Array'),
        _ann('Ship Tier', 'T6-X'),          # non-icon: not a layout row
    ]}})
    imp = _importer()

    layout = imp._load_confirmed_layout(str(shot))
    profile = imp._load_confirmed_profile(str(shot))

    assert set(layout) == {'Fore Weapons', 'Deflector'}
    assert len(layout['Fore Weapons']) == 2
    assert profile == {'Fore Weapons': 2, 'Deflector': 1}


def test_the_confirmed_tier_changes_the_slot_count(store):
    """The point of the whole exercise, stated as arithmetic."""
    from warp.warp_importer import _apply_ship_and_tier_bonuses

    counts = {}
    for tier in ('T6-X2', 'T6-X'):
        profile = {'Universal Consoles': 0, 'Devices': 3, 'Starship Traits': 5}
        _apply_ship_and_tier_bonuses(profile, None, tier)
        counts[tier] = (profile['Universal Consoles'], profile['Devices'])

    assert counts['T6-X2'] == (2, 5)
    assert counts['T6-X'] == (1, 4)
