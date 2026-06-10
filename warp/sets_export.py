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

No SETS imports — exporter is the boundary between WARP and any
v3.0.0-compatible build planner.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from warp.debug import log


BUILD_VERSION = 1


_VALID_RANKS = ('I', 'II', 'III')
# Max rank permissible at each BOFF slot index. Slot 0 = Ensign (rank I
# only), slot 1 = Lieutenant (≤II), slot 2 = LtCmdr (≤III), slot 3 =
# Commander (≤III — rank IV abilities don't exist).
_SLOT_MAX_RANK_IDX = (0, 1, 2, 2)


def _split_rank(full_name: str) -> tuple[str, str]:
    base, sep, suffix = full_name.rpartition(' ')
    if sep and suffix in _VALID_RANKS:
        return base, suffix
    return full_name, ''


def _resolve_rank(base_name: str, slot_idx: int, cache) -> str:
    """Pick highest rank ≤ slot's max that exists for `base_name` in
    `cache.boff_abilities['all']`.

    The upstream ability record stores rank tiers as `rank1info` /
    `rank2info` / `rank3info` (the at-rank effect description) — a tier
    is "present" when that field is a non-empty string. Only 1/237
    abilities currently has a missing tier (`Dampening Field` has no
    III), but the check keeps the export honest if that ratio grows.
    """
    max_idx = _SLOT_MAX_RANK_IDX[slot_idx] if 0 <= slot_idx < 4 else 2
    if cache is None:
        return _VALID_RANKS[max_idx]
    try:
        entry = cache.boff_abilities.get('all', {}).get(base_name)
        if not isinstance(entry, dict):
            return _VALID_RANKS[max_idx]
        for idx in range(max_idx, -1, -1):
            if entry.get(f'rank{idx + 1}info'):
                return _VALID_RANKS[idx]
    except Exception:
        pass
    return _VALID_RANKS[max_idx]


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


def write_sets_build(sets_build, path, cache=None) -> Path:
    path = Path(path)
    payload = build_sets_v3_dict(sets_build, cache)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    log.info(f'sets_export: wrote build → {path}')
    return path
