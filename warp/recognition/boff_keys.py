"""BOFF slot key schema — single source of truth for parsing the
slot-name strings that the layout detectors emit.

Producer: `layout_detector._detect_boffs_via_markers` (and legacy
band-scan fallbacks).
Consumers: `warp_dialog` (Phase 2 cluster→seat matching) and
`trainer_window._build_search_candidates` (autocomplete).

Key formats supported
---------------------
- `Boff Tactical`              — legacy profession-keyed
- `Boff Engineering`
- `Boff Science`
- `Boff Temporal` (etc.)       — legacy spec-prof seat
- `Boff Seat L[T]_483`         — new marker-keyed (side, prof code, marker_y)
- `Boff Seat L[T+Plt]_483`     — new with spec stripe (multi-char spec code)
- `Boff Seat L[U]_483`         — Universal — caller does content-based fallback
- `Boff Seat L_483`            — legacy seat-keyed without prof code

`parse_seat_profession` returns the base profession (Tactical /
Engineering / Science) or None when the key is Universal / legacy
seat-keyed without code. `parse_seat_spec` returns the spec
profession (Command / Intelligence / Temporal / Pilot / Miracle
Worker) or None.
"""
from __future__ import annotations

import re

# Codes match boff_marker.SEAT_CODE_LABEL / SPEC_CODE_LABEL — duplicated
# here so this module stays free of CV dependencies (kept tiny so UI
# layers can import it without dragging in OpenCV / numpy).
_SEAT_CODE_TO_PROF = {
    'T': 'Tactical',
    'E': 'Engineering',
    'S': 'Science',
    # 'U' (Universal) intentionally maps to None — Universal seats have
    # no inherent profession; callers must derive it from ability content.
    # 'G' (Ground) also maps to None: the brown ground marker carries no
    # profession info, and a single ground seat may mix base + spec
    # profs across its slots (e.g. 1× Tac + 3× MW). Use is_ground_seat()
    # to distinguish 'G' from 'U' for environment-based candidate filtering.
    'G': None,
}
# Multi-char human-friendly spec codes — single letters (O/P/Y/C/L)
# were not first-letter mnemonics and confused human readers of logs
# and the Recognition Report.
_SPEC_CODE_TO_PROF = {
    'Cmd': 'Command',
    'Int': 'Intelligence',
    'Tem': 'Temporal',
    'Plt': 'Pilot',
    'MW':  'Miracle Worker',
}

# Backward-compat: old single-letter spec codes still appear in stale
# learned-layout caches (anchors.json, community_anchors.json) and other
# data files saved before the 2026-05-07 rename. Producers in the live
# code path emit new codes only; this map exists solely so parsers can
# read legacy data. Old keys naturally disappear as the cache regenerates.
_LEGACY_SPEC_CODE_MAP = {
    'O': 'Cmd',  # Command
    'P': 'Int',  # Intelligence
    'Y': 'Tem',  # Temporal
    'C': 'Plt',  # Pilot
    'L': 'MW',   # Miracle Worker
}

# Legacy profession-keyed names emitted by _detect_via_full_scan and
# pre-marker detectors. `Universal` is omitted because a profession-keyed
# Universal slot would itself be Unknown (caller goes content-based).
_LEGACY_PROFESSIONS = frozenset({
    'Tactical', 'Engineering', 'Science',
})
_LEGACY_SPEC_PROFESSIONS = frozenset(_SPEC_CODE_TO_PROF.values())

# Boff Seat L[T+Plt]_483 → groups (side, code, spec, my)
# Boff Seat L[T]_483     → groups (side, code, None, my)
# Single-letter codes (O/P/Y/C/L) accepted for backward compatibility
# with stale caches; normalized to canonical multi-char codes in the
# parsing helpers below.
_SEAT_KEY_RE = re.compile(
    r'^Boff Seat ([LR])(?:\[([TESUG])(?:\+(Cmd|Int|Tem|Plt|MW|O|P|Y|C|L))?\])?_(\d+)$'
)


def _canon_spec_code(spec_code: str | None) -> str | None:
    """Normalize a spec code to its canonical multi-char form. Returns
    None if input is None; passes new codes through unchanged; maps
    legacy single-letter codes via _LEGACY_SPEC_CODE_MAP.
    """
    if not spec_code:
        return None
    return _LEGACY_SPEC_CODE_MAP.get(spec_code, spec_code)


def parse_seat_profession(slot_name: str) -> str | None:
    """Return the base profession for a BOFF slot key, or None if the
    key carries no profession info (Universal or legacy seat-keyed).
    """
    if not isinstance(slot_name, str) or not slot_name.startswith('Boff '):
        return None
    # Legacy profession-keyed: 'Boff Tactical' etc.
    rest = slot_name[5:].strip()
    if rest in _LEGACY_PROFESSIONS:
        return rest
    # Legacy spec-prof seat: 'Boff Temporal' — fall back to spec lookup.
    # Spec-prof keys carry no base profession, so still return None.
    if rest in _LEGACY_SPEC_PROFESSIONS:
        return None
    # New marker-keyed: 'Boff Seat L[T]_483' / 'Boff Seat L[T+P]_483'
    m = _SEAT_KEY_RE.match(slot_name)
    if m:
        code = m.group(2)
        if code:
            return _SEAT_CODE_TO_PROF.get(code)  # 'U' → None
    return None


def parse_seat_spec(slot_name: str) -> str | None:
    """Return the specialization profession for a BOFF slot key, or None
    if no spec is encoded.
    """
    if not isinstance(slot_name, str) or not slot_name.startswith('Boff '):
        return None
    # Legacy spec-prof seat: 'Boff Temporal' etc.
    rest = slot_name[5:].strip()
    if rest in _LEGACY_SPEC_PROFESSIONS:
        return rest
    # New marker-keyed: spec is the optional second bracket group.
    m = _SEAT_KEY_RE.match(slot_name)
    if m:
        spec_code = _canon_spec_code(m.group(3))
        if spec_code:
            return _SPEC_CODE_TO_PROF.get(spec_code)
    return None


def is_seat_keyed(slot_name: str) -> bool:
    """True iff `slot_name` is a marker-keyed seat name (with or without
    profession code), i.e. matches `Boff Seat L_<y>` or `Boff Seat L[T]_<y>`.
    """
    return isinstance(slot_name, str) and bool(_SEAT_KEY_RE.match(slot_name))


def is_ground_seat(slot_name: str) -> bool:
    """True iff `slot_name` is a ground BOFF seat key (marker code 'G').
    Used by icon_matcher to restrict candidate abilities to the ground
    environment. Ground seats allow base+spec profession mixing within
    a single seat, so callers must NOT also constrain by profession.
    """
    if not isinstance(slot_name, str):
        return False
    m = _SEAT_KEY_RE.match(slot_name)
    return bool(m and m.group(2) == 'G')


_VIRTUAL_NAMES = frozenset({'', '__empty__', '__inactive__'})


def _seat_label_from_items(seat_key: str, items) -> str:
    """Display label for a physical seat — Universal seats are resolved
    to their content's voted profession so the label matches what
    `build_writer` writes into the SETS JSON `boff_specs[seat_id]` for
    the seat-type dropdown over a Universal seat.

    For non-Universal seats this reduces to `pretty_slot(seat_key)`.
    """
    base = parse_seat_profession(seat_key)
    spec = parse_seat_spec(seat_key)
    if base is None and is_seat_keyed(seat_key) and not is_ground_seat(seat_key):
        # Universal — derive from items' own profession (post-remap
        # `it.slot` carries 'Boff <prof>' for non-virtual abilities).
        from collections import Counter
        prof_slots = [
            getattr(it, 'slot', '')
            for it in items
            if getattr(it, 'name', '') not in _VIRTUAL_NAMES
        ]
        prof_slots = [s for s in prof_slots
                      if s.startswith('Boff ') and s != 'Boff Universal']
        if prof_slots:
            base = Counter(prof_slots).most_common(1)[0][0][len('Boff '):]
    if base:
        label = f'Boff {base}'
    elif is_seat_keyed(seat_key) and is_ground_seat(seat_key):
        label = 'Boff Ground'
    else:
        label = 'Boff Universal'
    return f'{label}+{spec}' if spec else label


def group_items_by_seat(items):
    """Group RecognisedItem-like objects into seat-aware display groups.

    BOFF items that carry a non-empty `seat_key` (set by
    `_remap_boff_seat_slots` when the original slot was marker-keyed)
    group by physical seat; the group's label is derived from the seat
    and its contents via `_seat_label_from_items()` — Universal seats
    are resolved to the voted profession of the abilities slotted into
    them so the UI label matches the seat-type the SETS JSON will
    promote them to. Items without `seat_key` (legacy detector paths)
    and all non-BOFF items group by `.slot` as before.

    When two groups share the same base label (e.g. two Tactical
    seats), they are numbered `#1` / `#2` by visual Y order of the
    groups' topmost bbox so the user can distinguish them.

    Returns `list[(label, [items])]` insertion-ordered by ascending Y
    of each group's topmost bbox. Callers that want a different order
    (e.g. canonical SLOT_ORDER for non-BOFF) should re-sort.
    """
    raw: dict[tuple, list] = {}
    for it in items:
        slot = getattr(it, 'slot', '') or ''
        seat_key = getattr(it, 'seat_key', '') or ''
        # Fallback: when remap left the item with a seat-keyed `slot` and
        # no `seat_key` (e.g. ground ability unknown to boff_abilities cache
        # so the profession remap didn't fire), use `slot` itself as the
        # seat key so the row gets a pretty 'Boff Ground'/'Boff Universal'
        # label via _seat_label_from_items instead of leaking the raw key.
        if not seat_key and is_seat_keyed(slot):
            seat_key = slot
        if slot.startswith('Boff') and seat_key and is_seat_keyed(seat_key):
            key = ('seat', seat_key)
        else:
            key = ('slot', slot)
        raw.setdefault(key, []).append(it)

    def _top_y(item_list):
        ys = [it.bbox[1] for it in item_list
              if getattr(it, 'bbox', None) and len(it.bbox) >= 2]
        return min(ys) if ys else 1_000_000_000

    ordered_keys = sorted(raw.keys(), key=lambda k: _top_y(raw[k]))

    base_labels: list[tuple[str, list]] = []
    for key in ordered_keys:
        kind, value = key
        item_list = raw[key]
        if kind == 'seat':
            base_label = _seat_label_from_items(value, item_list)
        else:
            base_label = value
        base_labels.append((base_label, item_list))

    label_total: dict[str, int] = {}
    for base_label, _ in base_labels:
        label_total[base_label] = label_total.get(base_label, 0) + 1

    label_seen: dict[str, int] = {}
    result: list[tuple[str, list]] = []
    for base_label, item_list in base_labels:
        if label_total[base_label] > 1:
            label_seen[base_label] = label_seen.get(base_label, 0) + 1
            label = f'{base_label} #{label_seen[base_label]}'
        else:
            label = base_label
        result.append((label, item_list))
    return result


def pretty_slot(slot_name: str) -> str:
    """Convert a dynamic BOFF seat key into a user-friendly label:
    - `Boff Seat L[E]_392`     → `Boff Engineering`
    - `Boff Seat R[T+Plt]_510` → `Boff Tactical+Pilot`
    - `Boff Seat L[U]_478`     → `Boff Universal`
    - `Boff Seat L[G]_478`     → `Boff Ground`
    - `Boff Seat L_478`        → `Boff Universal` (legacy seat-keyed without code)

    Non-seat-keyed slot names (e.g. `Boff Tactical`, `Fore Weapons`,
    `Ship Name`) are returned unchanged.
    """
    if not isinstance(slot_name, str) or not slot_name.startswith('Boff Seat'):
        return slot_name
    if not is_seat_keyed(slot_name):
        return slot_name
    prof = parse_seat_profession(slot_name)
    spec = parse_seat_spec(slot_name)
    if prof:
        base = f'Boff {prof}'
    elif is_ground_seat(slot_name):
        base = 'Boff Ground'
    else:
        base = 'Boff Universal'
    return f'{base}+{spec}' if spec else base
