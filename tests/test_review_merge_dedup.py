"""A confirmed annotation outranks a fresh detection of the same thing.

Ship Name / Type / Tier and the single equipment slots may hold one row per
screenshot, and equipment / trait / BOFF boxes may not sit on top of each
other. Both rules are decided by the two helpers exercised here, which
`_populate_review_panel` calls while merging disk annotations with detector
output.
"""
from __future__ import annotations

import pytest

pytest.importorskip('PySide6')

from warp.trainer.training_data import SINGLE_INSTANCE_SLOTS  # noqa: E402
from warp.trainer.trainer_window import (  # noqa: E402
    _confirmed_overlap,
    _single_instance_twin,
)


def _entry(slot: str, bbox: tuple) -> dict:
    return {'slot': slot, 'bbox': list(bbox), 'state': 'confirmed'}


# ── Rule 1: one row per single-instance slot ──────────────────────────

def test_ship_tier_is_capped_at_one_row():
    assert 'Ship Tier' in SINGLE_INSTANCE_SLOTS
    assert 'Ship Type' in SINGLE_INSTANCE_SLOTS
    assert 'Ship Name' in SINGLE_INSTANCE_SLOTS


def test_the_single_space_equipment_slots_are_capped_too():
    for slot in ('Deflector', 'Sec-Def', 'Engines', 'Warp Core', 'Shield',
                 'Experimental'):
        assert slot in SINGLE_INSTANCE_SLOTS


def test_the_single_ground_equipment_slots_are_capped_too():
    for slot in ('Kit', 'Body Armor', 'EV Suit', 'Personal Shield'):
        assert slot in SINGLE_INSTANCE_SLOTS


@pytest.mark.parametrize('build_type', ['SPACE', 'GROUND', 'SPEC',
                                        'SPACE_MIXED', 'GROUND_MIXED'])
def test_the_cap_matches_the_slots_the_importer_maxes_at_one(build_type):
    # Keeps the review panel's cap and the importer's slot profile from
    # drifting apart as ships/slots are added.
    from warp.warp_importer import SLOT_ORDER
    capped = [e['name'] for e in SLOT_ORDER[build_type] if e.get('max') == 1]
    assert capped, f'no max=1 slot in {build_type} — test would be vacuous'
    for name in capped:
        assert name in SINGLE_INSTANCE_SLOTS, name


def test_a_moved_tier_box_still_finds_its_confirmed_twin():
    # The OCR line width changes between runs, so the fresh bbox overlaps
    # the confirmed one far below the 0.5 IoU recovery threshold.
    confirmed = {'aid-tier': _entry('Ship Tier', (23, 66, 153, 20))}
    assert _single_instance_twin('Ship Tier', confirmed, set()) == 'aid-tier'


def test_a_consumed_twin_is_not_offered_twice():
    confirmed = {'aid-tier': _entry('Ship Tier', (23, 66, 153, 20))}
    assert _single_instance_twin('Ship Tier', confirmed, {'aid-tier'}) is None


def test_multi_instance_slots_have_no_twin():
    confirmed = {'aid-fore': _entry('Fore Weapons', (621, 41, 35, 44))}
    assert _single_instance_twin('Fore Weapons', confirmed, set()) is None


# ── Rule 2: icon boxes may not overlap ────────────────────────────────

def test_a_fresh_icon_box_on_a_confirmed_one_is_reported():
    confirmed = {'aid-dev': _entry('Devices', (548, 347, 35, 44))}
    hit = _confirmed_overlap('Devices', (547, 346, 35, 44), confirmed, set())
    assert hit is not None
    assert hit[0] == 'aid-dev'


def test_a_free_position_reports_no_clash():
    confirmed = {'aid-dev': _entry('Devices', (548, 347, 35, 44))}
    assert _confirmed_overlap('Devices', (731, 347, 35, 44),
                              confirmed, set()) is None


def test_a_clash_with_a_different_slot_still_counts():
    confirmed = {'aid-uni': _entry('Universal Consoles', (695, 398, 35, 44))}
    assert _confirmed_overlap('Engineering Consoles', (697, 399, 35, 44),
                              confirmed, set()) is not None


def test_ship_type_and_tier_may_share_the_text_line():
    # They overlap by design — rule 1 keeps each of them unique instead.
    confirmed = {'aid-tier': _entry('Ship Tier', (29, 65, 204, 20))}
    assert _confirmed_overlap('Ship Type', (26, 37, 233, 48),
                              confirmed, set()) is None


def test_an_icon_box_over_the_ship_info_line_is_left_alone():
    confirmed = {'aid-name': _entry('Ship Name', (27, 33, 126, 18))}
    assert _confirmed_overlap('Devices', (27, 33, 126, 18),
                              confirmed, set()) is None


def test_a_consumed_confirmed_box_no_longer_blocks():
    confirmed = {'aid-dev': _entry('Devices', (548, 347, 35, 44))}
    assert _confirmed_overlap('Devices', (547, 346, 35, 44),
                              confirmed, {'aid-dev'}) is None
