"""
SETS v3.0.0 build JSON exporter — port from sets-warp.

Final step of the standalone WARP → SETS path: takes an in-memory SETS
build dict (produced by `warp.build_writer.build_from_result`) and emits
a JSON file loadable by SETS v3.0.0 `File → Load Build`.

Two contract adjustments vs. our internal dict shape:

  1. Top-level `_version` field — `BuildLoader` keys migration on it.
  2. BOFF ability dicts get a `rank` Roman numeral. WARP's icon-based
     detector doesn't know rank (icons are shared across I/II/III), so
     we resolve the highest rank ≤ slot's max that exists in
     `cargo.boff_abilities['all'][base]`. Slot 0→max I, 1→max II,
     2/3→max III. Without this, v3.0.0's `load_boff_stations` raises
     KeyError on `ability['rank']` and `remove_invalid_build_items`
     cascades into trait loss.

Every payload is checked against the frozen format contract on its way
out (`warp.sets_schema`); see `docs/SETS_FORMAT_CONTRACT.md`.

No SETS imports — exporter is the boundary between WARP and any
v3.0.0-compatible build planner.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

from warp import upstream_gaps
from warp.debug import log
from warp.sets_schema import (
    BOFF_RANKS, SEAT_RANK_SLOT, errors, summarise, validate_sets_build)


BUILD_VERSION = 1


_VALID_RANKS = BOFF_RANKS
# Fallback when cargo has no usable rank data: the highest tier the slot
# could hold. Slot 0 = Ensign, 1 = Lieutenant, 2 = LtCmdr, 3 = Commander
# (rank IV abilities don't exist, so 2 and 3 share the cap).
_SLOT_MAX_RANK_IDX = (0, 1, 2, 2)


def _split_rank(full_name: str) -> tuple[str, str]:
    base, sep, suffix = full_name.rpartition(' ')
    if sep and suffix in _VALID_RANKS:
        return base, suffix
    return full_name, ''


def _resolve_rank(base_name: str, slot_idx: int, cache) -> str:
    """Pick the rank `base_name` is actually offered at in slot `slot_idx`.

    Rank tiers are not evenly spread across seats. Cargo's
    `rank<N>rank` names the seat rank at which Roman tier N unlocks, and
    that is what SETS keys its per-slot ability lists on
    (`cargomanager.cache_boff_data`). `Cannons: Rapid Fire` unlocks
    I/II/III at Lieutenant/LtCmdr/Commander, so a Lieutenant slot must
    carry rank I — while `Aceton Beam` starts at LtCmdr and has no rank
    that an Ensign slot could hold.

    Order: the highest tier unlocking exactly at this slot; else the
    highest tier unlocking below it (legal in game, though SETS' picker
    doesn't list it there); else the lowest tier that exists at all.
    """
    max_idx = _SLOT_MAX_RANK_IDX[slot_idx] if 0 <= slot_idx < 4 else 2
    entry = None
    if cache is not None:
        try:
            entry = cache.boff_abilities.get('all', {}).get(base_name)
        except Exception:
            entry = None
    if not isinstance(entry, dict):
        return _VALID_RANKS[max_idx]

    unlock_slots = {}
    for idx, roman in enumerate(_VALID_RANKS):
        slot = SEAT_RANK_SLOT.get(entry.get(f'rank{idx + 1}rank'))
        if slot is not None:
            unlock_slots[roman] = slot

    if not unlock_slots:
        # Ground abilities carry kit-module wikitext instead of a rank.
        for idx in range(max_idx, -1, -1):
            if entry.get(f'rank{idx + 1}info'):
                return _VALID_RANKS[idx]
        return _VALID_RANKS[max_idx]

    for roman in reversed(_VALID_RANKS):
        if unlock_slots.get(roman) == slot_idx:
            return roman
    below = [r for r in _VALID_RANKS if unlock_slots.get(r, 99) < slot_idx]
    if below:
        return below[-1]

    lowest = min(unlock_slots.values())
    roman = next(r for r in _VALID_RANKS if unlock_slots.get(r) == lowest)
    log.warning(f'sets_export: {base_name} has no rank for slot {slot_idx} '
                f'(unlocks at slot {lowest}) — writing {roman}')
    return roman


def _normalise_boffs(seats: list, cache) -> int:
    n = 0
    if not isinstance(seats, list):
        return 0
    for seat in seats:
        if not isinstance(seat, list):
            continue
        for slot_idx, ability in enumerate(seat):
            if not isinstance(ability, dict) or 'item' not in ability:
                continue
            if ability.get('rank') in _VALID_RANKS:
                continue
            base, rank = _split_rank(ability['item'])
            if not rank:
                rank = _resolve_rank(base, slot_idx, cache)
            ability['item'] = base
            ability['rank'] = rank
            n += 1
    return n


def build_sets_v3_dict(sets_build: dict, cache=None) -> dict:
    """Convert in-memory SETS build dict → SETS v3.0.0 build dict."""
    out = copy.deepcopy(sets_build)
    out['_version'] = BUILD_VERSION

    n_space  = _normalise_boffs(out.get('space',  {}).get('boffs', []), cache)
    n_ground = _normalise_boffs(out.get('ground', {}).get('boffs', []), cache)

    log.info(f'sets_export: serialised build — boff_ranks_split=space:{n_space} ground:{n_ground}')
    return out


def write_sets_build(sets_build, path, cache=None, report_to: list | None = None) -> Path:
    """Serialise `sets_build` to `path`, checking it against the contract.

    Violations never block the write — a user who exported a build gets
    the file. They land in `warp_system.log` and, when `report_to` is
    given, in that list so a caller (the GUI) can offer a pre-filled
    issue. `WARP_STRICT_EXPORT=1` turns errors fatal instead; CI sets it
    so a schema regression fails the test run.
    """
    path = Path(path)
    payload = build_sets_v3_dict(sets_build, cache)

    violations = validate_sets_build(payload, cache)
    if report_to is not None:
        report_to.extend(violations)

    # `not_in_sets` is nobody's bug here — the item was recognised correctly
    # and SETS still drops it. Keep a tally so the case for fixing it upstream
    # can be made from real builds rather than from one user's anecdote.
    upstream_gaps.record(violations)

    if violations:
        log.warning(f'SETS_SCHEMA: {summarise(violations)}')
        if os.environ.get('WARP_STRICT_EXPORT') == '1' and errors(violations):
            raise ValueError(f'SETS schema violations: {summarise(violations)}')

    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    log.info(f'sets_export: wrote build → {path}')
    return path
