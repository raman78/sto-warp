# warp/trainer/constants.py
# Shared constants for WARP CORE — screen types, slot groups, settings keys,
# confidence thresholds. Extracted from trainer_window.py during the
# Phase-0 refactor so workers + UI modules can share them without
# importing the giant WarpCoreWindow module.

from __future__ import annotations


# ── QSettings keys ─────────────────────────────────────────────────────
_KEY_LAST_DIR    = 'warp_core/last_dir'
_KEY_AUTO_ACCEPT = 'warp_core/auto_accept_enabled'
_KEY_AUTO_CONF   = 'warp_core/auto_accept_conf'


# ── Confidence buckets ────────────────────────────────────────────────
CONF_HIGH   = 0.85
CONF_MEDIUM = 0.70


# ── Slot groups per screen type ────────────────────────────────────────
SLOT_GROUPS: dict[str, list[str]] = {
    # SPACE_EQ: space equipment + ship metadata (name/type/tier live on space screenshots)
    'SPACE_EQ': [
        'Fore Weapons', 'Deflector', 'Sec-Def', 'Engines', 'Warp Core', 'Shield',
        'Aft Weapons', 'Experimental', 'Devices', 'Universal Consoles',
        'Engineering Consoles', 'Science Consoles', 'Tactical Consoles', 'Hangars',
        'Ship Name', 'Ship Type', 'Ship Tier',
    ],
    # GROUND_EQ: ground equipment only — no ship metadata
    'GROUND_EQ': [
        'Body Armor', 'EV Suit', 'Personal Shield', 'Weapons', 'Kit', 'Kit Modules', 'Ground Devices',
    ],
    'TRAITS': [
        'Personal Space Traits', 'Starship Traits', 'Space Reputation', 'Active Space Rep',
        'Personal Ground Traits', 'Ground Reputation', 'Active Ground Rep',
    ],
    'SPACE_TRAITS': [
        'Personal Space Traits', 'Starship Traits', 'Space Reputation', 'Active Space Rep',
    ],
    'GROUND_TRAITS': [
        'Personal Ground Traits', 'Ground Reputation', 'Active Ground Rep',
    ],
    'BOFFS': [
        'Boff Tactical', 'Boff Engineering', 'Boff Science',
        'Boff Intelligence', 'Boff Command', 'Boff Pilot', 'Boff Miracle Worker', 'Boff Temporal',
    ],
    'SPACE_BOFFS': [
        'Boff Tactical', 'Boff Engineering', 'Boff Science',
        'Boff Intelligence', 'Boff Command', 'Boff Pilot', 'Boff Miracle Worker', 'Boff Temporal',
    ],
    'GROUND_BOFFS': [
        'Boff Tactical', 'Boff Engineering', 'Boff Science',
        'Boff Intelligence', 'Boff Command', 'Boff Pilot', 'Boff Miracle Worker', 'Boff Temporal',
    ],
    'SPECIALIZATIONS': [],
    # SPACE_MIXED: merged space screenshot — equipment + traits + boffs + specs, no ground gear
    'SPACE_MIXED': [
        'Fore Weapons', 'Deflector', 'Sec-Def', 'Engines', 'Warp Core', 'Shield',
        'Aft Weapons', 'Experimental', 'Devices', 'Universal Consoles',
        'Engineering Consoles', 'Science Consoles', 'Tactical Consoles', 'Hangars',
        'Ship Name', 'Ship Type', 'Ship Tier',
        'Personal Space Traits', 'Starship Traits', 'Space Reputation', 'Active Space Rep',
        'Boff Tactical', 'Boff Engineering', 'Boff Science',
        'Boff Intelligence', 'Boff Command', 'Boff Pilot', 'Boff Miracle Worker', 'Boff Temporal',
        'Primary Specialization', 'Secondary Specialization',
    ],
    # GROUND_MIXED: merged ground screenshot — ground gear + traits + boffs + specs, no space gear
    'GROUND_MIXED': [
        'Body Armor', 'EV Suit', 'Personal Shield', 'Weapons', 'Kit', 'Kit Modules', 'Ground Devices',
        'Personal Ground Traits', 'Ground Reputation', 'Active Ground Rep',
        'Boff Tactical', 'Boff Engineering', 'Boff Science',
        'Boff Intelligence', 'Boff Command', 'Boff Pilot', 'Boff Miracle Worker', 'Boff Temporal',
        'Primary Specialization', 'Secondary Specialization',
    ],
}


SCREEN_TYPE_LABELS: dict[str, str] = {
    'SPACE_EQ': 'Space Equipment', 'GROUND_EQ': 'Ground Equipment', 'TRAITS': 'Traits',
    'SPACE_TRAITS': 'Space Traits', 'GROUND_TRAITS': 'Ground Traits',
    'BOFFS': 'Bridge Officers', 'SPACE_BOFFS': 'Space Bridge Officers',
    'GROUND_BOFFS': 'Ground Bridge Officers', 'SPECIALIZATIONS': 'Specializations',
    'SPACE_MIXED': 'Space Mixed (merged)', 'GROUND_MIXED': 'Ground Mixed (merged)', 'UNKNOWN': 'Unknown',
}

SCREEN_TYPE_ICONS: dict[str, str] = {
    'SPACE_EQ': '🚀', 'GROUND_EQ': '🦶', 'TRAITS': '✨',
    'SPACE_TRAITS': '✨', 'GROUND_TRAITS': '✨',
    'BOFFS': '👥', 'SPACE_BOFFS': '👥', 'GROUND_BOFFS': '👥',
    'SPECIALIZATIONS': '🎯', 'SPACE_MIXED': '🌌', 'GROUND_MIXED': '🗺️', 'UNKNOWN': '❓',
}

SCREEN_TO_SLOT_GROUP: dict[str, str] = {
    'SPACE_EQ':       'SPACE_EQ',
    'GROUND_EQ':      'GROUND_EQ',
    'TRAITS':         'TRAITS',
    'SPACE_TRAITS':   'SPACE_TRAITS',
    'GROUND_TRAITS':  'GROUND_TRAITS',
    'BOFFS':          'BOFFS',
    'SPACE_BOFFS':    'SPACE_BOFFS',
    'GROUND_BOFFS':   'GROUND_BOFFS',
    'SPECIALIZATIONS':'SPECIALIZATIONS',
    'SPACE_MIXED':    'SPACE_MIXED',
    'GROUND_MIXED':   'GROUND_MIXED',
    'UNKNOWN':        'ALL',   # unknown type → show everything, let user decide
}

FIXED_VALUE_SLOTS: frozenset[str] = frozenset(['Ship Tier', 'Ship Type'])
_SHIP_INFO_SLOTS = ['Ship Name', 'Ship Type', 'Ship Tier']


# Build ALL_SLOTS as a flat deduplicated list of every slot across all groups
ALL_SLOTS: list[str] = []
for _slots in SLOT_GROUPS.values():
    for _s in _slots:
        if _s not in ALL_SLOTS:
            ALL_SLOTS.append(_s)
for _s in _SHIP_INFO_SLOTS:
    if _s not in ALL_SLOTS:
        ALL_SLOTS.append(_s)
SLOT_GROUPS['ALL'] = ALL_SLOTS


SPECIALIZATION_NAMES: list[str] = [
    'Command Officer', 'Intelligence Officer', 'Miracle Worker',
    'Pilot', 'Temporal Operative', 'Constable', 'Commando', 'Strategist',
]
