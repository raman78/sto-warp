"""The space equipment panel's rows — the one table every module reads.

The game draws one row per slot type, top to bottom, in a fixed order; a ship
that lacks an optional row does not draw it and the rows below move up. That
order used to be written out five times — in eq_geometry, the layout detector
(twice, as two loops that inserted the optional rows) and the importer — and
the copies drifted: one followed the grouping of docs/sto_slots_rules.md
instead of the screen, another put Hangars after Aft Weapons. Every other view
of the order is now derived from ROWS.

Order measured from the confirmed boxes of every annotated SPACE_EQ /
SPACE_MIXED screenshot; see the row table in docs/sto_slots_rules.md.
"""
from __future__ import annotations

from typing import NamedTuple


class Row(NamedTuple):
    slot: str       # the slot name the rest of the program uses
    label: str      # the English label the game prints beside the row
    optional: bool  # some ships do not draw this row
    assumed: bool   # drawn when the ship's slot counts are unknown
    key: str        # importer / build-writer key
    max: int        # most cells the row can hold
    weapon: bool
    exp: bool


ROWS: tuple[Row, ...] = (
    Row('Fore Weapons',         'Fore Weapons',         False, True,  'fore_weapons', 5, True,  False),
    Row('Deflector',            'Deflector',            False, True,  'deflector',    1, False, False),
    Row('Sec-Def',              'Sec-Def',              True,  False, 'sec_def',      1, False, False),
    Row('Engines',              'Engines',              False, True,  'engines',      1, False, False),
    Row('Warp Core',            'Warp Core',            False, True,  'core',         1, False, False),
    Row('Shield',               'Shields',              False, True,  'shield',       1, False, False),
    Row('Aft Weapons',          'Aft Weapons',          True,  True,  'aft_weapons',  5, True,  False),
    Row('Experimental',         'Experimental',         True,  False, 'experimental', 1, True,  True),
    Row('Devices',              'Devices',              False, True,  'devices',      6, False, False),
    Row('Universal Consoles',   'Universal Consoles',   True,  True,  'uni_consoles', 3, False, False),
    Row('Engineering Consoles', 'Engineering Consoles', False, True,  'eng_consoles', 5, False, False),
    Row('Science Consoles',     'Science Consoles',     False, True,  'sci_consoles', 5, False, False),
    Row('Tactical Consoles',    'Tactical Consoles',    False, True,  'tac_consoles', 5, False, False),
    Row('Hangars',              'Hangars',              True,  False, 'hangars',      4, False, False),
)

SLOTS: tuple[str, ...] = tuple(r.slot for r in ROWS)
INDEX: dict[str, int] = {r.slot: i for i, r in enumerate(ROWS)}
LABEL_INDEX: dict[str, int] = {r.label: i for i, r in enumerate(ROWS)}
SLOT_BY_INDEX: dict[int, str] = dict(enumerate(SLOTS))
OPTIONAL: frozenset[str] = frozenset(r.slot for r in ROWS if r.optional)

# The rows assumed when nothing says otherwise, and the carrier's, which adds
# Hangars — last, where the game draws it.
BASE_ORDER: tuple[str, ...] = tuple(r.slot for r in ROWS if r.assumed)
CARRIER_ORDER: tuple[str, ...] = BASE_ORDER + ('Hangars',)


def row_sequence(slot_order, profile: dict | None) -> list[str]:
    """The rows this ship draws, top to bottom.

    Starts from *slot_order*, drops any row the profile counts as 0, and puts
    each optional row the profile counts above 0 — and *slot_order* lacks —
    at its place in ROWS. A row the profile does not mention is kept: the
    profile says what a ship lacks, not what it has. Rows that are not space
    equipment (a ground order) pass through with the same zero rule.
    """
    profile = profile or {}
    rows = list(dict.fromkeys(slot_order))
    if any(s in INDEX for s in rows):
        for r in ROWS:
            if r.slot in rows or not r.optional or profile.get(r.slot, 0) <= 0:
                continue
            at = next((i for i, s in enumerate(rows)
                       if INDEX.get(s, -1) > INDEX[r.slot]), len(rows))
            rows.insert(at, r.slot)
    return [s for s in rows if profile.get(s, 1) > 0]
