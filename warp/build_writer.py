"""ImportResult → SETS v3.0.0 build dict converter.

Pure-dict port of `sets-warp/warp/warp_dialog._apply_to_sets()`. The
sets-warp original wrote into a live SETS Qt app (`self._sets.build`);
this version produces an in-memory dict that `warp.sets_export.
write_sets_build()` then serialises for SETS' `File → Load Build`.

Pipeline mirrors the sets-warp one:

  1. `_resolve_ship`  — match `ship_type` against `cargo.ships()` via
     exact / word-subset / fuzzy. Returns matched ship dict + canonical
     ship name; both go onto `build['space']['ship'/'tier'/'ship_name']`.
  2. Equipment + traits  — loop over `result.items`, route each via
     `SLOT_MAP` into the right list slot.
  3. BOFFs  — cluster ability items by (Y, X), match clusters to ship
     seats by profession (with spec fallback), write `{'item': name}`
     into the matching seat's list. Ranks are filled in later by
     `sets_export._normalise_boffs` using `cargo.boff_abilities['all']`.

No Qt, no SETS imports — only `warp.data.cargo` + the result dataclass.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from difflib import get_close_matches
from typing import Iterable

from warp import config
from warp.data.cargo import cache_view
from warp.data.empty_build import empty_build
from warp.debug import log
from warp.warp_importer import ImportResult, RecognisedItem


# Names emitted by the detector for empty / inactive BOFF cells; never
# written into the build dict.
VIRTUAL_ITEM_NAMES = frozenset({'__empty__', '__inactive__'})


# WARP slot → (SETS build_key, environment, is_equipment).
# Verbatim copy of `sets-warp/warp/warp_dialog.SLOT_MAP`.
SLOT_MAP: dict[str, tuple[str, str, bool]] = {
    # SPACE equipment
    'Fore Weapons':           ('fore_weapons',      'space', True),
    'Aft Weapons':            ('aft_weapons',       'space', True),
    'Experimental Weapon':    ('experimental',      'space', True),
    'Experimental':           ('experimental',      'space', True),
    'Devices':                ('devices',           'space', True),
    'Hangars':                ('hangars',           'space', True),
    'Deflector':              ('deflector',        'space', True),
    'Sec-Def':                ('sec_def',           'space', True),
    'Engines':                ('engines',           'space', True),
    'Warp Core':              ('core',              'space', True),
    'Shield':                 ('shield',            'space', True),
    'Universal Consoles':     ('uni_consoles',      'space', True),
    'Engineering Consoles':   ('eng_consoles',      'space', True),
    'Science Consoles':       ('sci_consoles',      'space', True),
    'Tactical Consoles':      ('tac_consoles',      'space', True),
    # SPACE traits
    'Personal Space Traits':  ('traits',            'space', False),
    'Starship Traits':        ('starship_traits',   'space', False),
    'Space Reputation':       ('rep_traits',        'space', False),
    'Reputation Traits':      ('rep_traits',        'space', False),
    'Active Space Rep':       ('active_rep_traits', 'space', False),
    'Active Rep Traits':      ('active_rep_traits', 'space', False),
    # GROUND equipment
    'Body Armor':             ('armor',             'ground', True),
    'EV Suit':                ('ev_suit',           'ground', True),
    'Personal Shield':        ('personal_shield',   'ground', True),
    'Weapons':                ('weapons',           'ground', True),
    'Kit':                    ('kit',               'ground', True),
    'Kit Modules':            ('kit_modules',       'ground', True),
    'Ground Devices':         ('ground_devices',    'ground', True),
    # GROUND traits
    'Personal Ground Traits': ('traits',            'ground', False),
    'Ground Reputation':      ('rep_traits',        'ground', False),
    'Ground Rep Traits':      ('rep_traits',        'ground', False),
    'Active Ground Rep':      ('active_rep_traits', 'ground', False),
}


_META_SLOTS = frozenset({
    'Ship Type', 'Ship Tier',
    'Primary Specialization', 'Secondary Specialization',
})

_FUZZY_MATCH_CUTOFF        = 0.68
_MIN_CONFIDENCE_PROFESSION = 0.40
_DEFAULT_RARITY            = 'Epic'
_DEFAULT_MARK              = 'XV'


# ── BOFF rank parsing ───────────────────────────────────────────────

_BOFF_RANKS = {
    'Commander': 4, 'Lieutenant Commander': 3, 'Lieutenant': 2, 'Ensign': 1,
}


def _get_boff_spec(seat_details: str) -> tuple[int, str, str]:
    """`'Lieutenant Tactical-Pilot'` → (2, 'Tactical', 'Pilot').
    Port of `src.buildupdater.get_boff_spec`."""
    if '-' in seat_details:
        rank_and_prof, spec = seat_details.split('-', 1)
    else:
        rank_and_prof, spec = seat_details, ''
    rank_name, _, profession = rank_and_prof.rpartition(' ')
    return (_BOFF_RANKS.get(rank_name, 4), profession, spec)


# ── Public entry point ──────────────────────────────────────────────


@dataclass
class WriteReport:
    """Diagnostic counters returned to the caller for status display."""
    ship:               str = ''
    ship_resolved:      bool = False
    n_equipment:        int = 0
    n_traits:           int = 0
    n_boff_abilities:   int = 0
    overflow_consoles:  int = 0
    unmatched_items:    int = 0


def build_from_result(result: ImportResult, cache=None) -> tuple[dict, WriteReport]:
    """Convert an `ImportResult` into a SETS-loadable build dict.

    `cache` is optional; if omitted, the module's cargo view is used.
    Returns `(build_dict, report)`.
    """
    if cache is None:
        cache = cache_view()

    build  = empty_build('full')
    report = WriteReport()

    ship_data = _resolve_ship(build, result.ship_type, result.ship_tier, cache, report)

    boff_items, overflow = _write_equipment_and_traits(build, result.items, cache, report)

    if overflow:
        _redistribute_console_overflow(build, overflow, report)

    if boff_items:
        is_ground = (result.build_type == 'GROUND_BOFFS')
        _write_boffs(build, boff_items, ship_data, cache, is_ground, report)

    _apply_elite_captain(build)

    return build, report


def _apply_elite_captain(build: dict) -> None:
    """Flip `captain.elite` on when any elite-gated slot was populated.

    SETS hides the 6th kit module, 5th ground device, and the 10th
    space/ground trait when the captain isn't Elite (buildmanager.py
    279-292). Without this flag those items end up in the JSON but
    invisible in the UI — the user opens their export and is missing a
    slot. Reading the slots after the write phase is a one-way inference:
    we never flip elite *off*, so a hand-edited build that explicitly
    set elite stays unchanged."""
    g, s = build['ground'], build['space']

    def _filled(x) -> bool:
        if isinstance(x, dict):
            return bool(x.get('item'))
        return isinstance(x, str) and bool(x)

    if (_filled(g['kit_modules'][5])
            or _filled(g['ground_devices'][4])
            or _filled(g['traits'][9])
            or _filled(s['traits'][9])):
        build['captain']['elite'] = True
        log.info('build_writer: detected Elite Captain — captain.elite = True')


# ── Ship resolution ─────────────────────────────────────────────────


def _resolve_ship(build: dict, ship_type: str | None, ship_tier: str | None,
                  cache, report: WriteReport) -> dict | None:
    if not ship_type:
        log.info('build_writer: no ship type — leaving ship blank')
        return None

    ships = cache.ships
    if ship_type in ships:
        match = ship_type
    else:
        ocr_words = set(ship_type.lower().split())
        candidates = list(ships.keys())
        subset = [c for c in candidates if ocr_words.issubset(set(c.lower().split()))]
        if len(subset) == 1:
            match = subset[0]
        elif len(subset) > 1:
            match = min(subset, key=lambda c: len(set(c.lower().split()) - ocr_words))
        else:
            hits = get_close_matches(ship_type, candidates, n=1, cutoff=_FUZZY_MATCH_CUTOFF)
            match = hits[0] if hits else None

    if not match:
        log.info(f'build_writer: ship {ship_type!r} not in cargo — leaving blank')
        return None

    ship_data = ships[match]
    tier_num  = ship_data.get('tier', 6)
    target_tier = ship_tier or f'T{tier_num}'

    build['space']['ship'] = match
    build['space']['tier'] = target_tier
    report.ship = match
    report.ship_resolved = True
    log.info(f'build_writer: ship {ship_type!r} → {match!r} tier={target_tier}')
    return ship_data


# ── Equipment + traits ──────────────────────────────────────────────


def _write_equipment_and_traits(
    build: dict, items: Iterable[RecognisedItem], cache, report: WriteReport,
) -> tuple[list[RecognisedItem], list[tuple[RecognisedItem, dict, str]]]:
    boffs: list[RecognisedItem] = []
    overflow: list[tuple[RecognisedItem, dict, str]] = []

    for ri in items:
        if not ri.name or ri.name in VIRTUAL_ITEM_NAMES:
            continue

        # Ship Name/Type/Tier and specialisations are already surfaced via
        # ImportResult.ship_* / spec_* metadata — the items list carries them
        # only so WARP CORE can show their OCR bboxes. Silently skip here.
        if ri.slot in _META_SLOTS:
            continue

        if ri.slot.startswith('Boff '):
            boffs.append(ri)
            continue

        slot_info = SLOT_MAP.get(ri.slot)
        if not slot_info:
            log.warning(f'build_writer: unknown slot {ri.slot!r} — skipping')
            report.unmatched_items += 1
            continue

        build_key, env, is_equipment = slot_info
        idx = ri.slot_index
        bucket = build[env].get(build_key)
        if bucket is None:
            log.warning(f'build_writer: missing bucket {env}/{build_key} — skipping')
            report.unmatched_items += 1
            continue

        if not is_equipment:
            if idx >= len(bucket):
                log.warning(f'build_writer: trait slot {ri.slot}[{idx}] OOR — skipping')
                report.unmatched_items += 1
                continue
            bucket[idx] = {'item': ri.name}
            report.n_traits += 1
            continue

        item_data = _make_equipment_item(ri, build_key, cache)
        if item_data is None:
            report.unmatched_items += 1
            continue

        if idx >= len(bucket):
            if build_key == 'uni_consoles':
                overflow.append((ri, item_data, env))
            else:
                log.warning(f'build_writer: {ri.slot}[{idx}] OOR — skipping {ri.name!r}')
                report.unmatched_items += 1
            continue

        bucket[idx] = item_data
        report.n_equipment += 1

    return boffs, overflow


def _make_equipment_item(ri: RecognisedItem, build_key: str, cache) -> dict | None:
    eq_cache = cache.equipment.get(build_key, {})
    entry    = eq_cache.get(ri.name)
    if not entry:
        log.warning(f'build_writer: {ri.name!r} not in cache[{build_key}] — skipping')
        return None
    rarity = entry.get('rarity') or _DEFAULT_RARITY
    return {
        'item':      ri.name,
        'rarity':    rarity,
        'mark':      _DEFAULT_MARK,
        'modifiers': [None] * 4,
    }


def _redistribute_console_overflow(
    build: dict, overflow: list[tuple[RecognisedItem, dict, str]], report: WriteReport,
):
    """When uni_consoles list is full, drop extras into the next free
    eng/sci/tac slot — port of `_handle_console_overflow`."""
    next_idx = {'eng_consoles': 0, 'sci_consoles': 0, 'tac_consoles': 0}
    for ck in next_idx:
        bucket = build['space'].get(ck, [])
        for i, cur in enumerate(bucket):
            if cur is None:
                next_idx[ck] = i
                break
        else:
            next_idx[ck] = len(bucket)

    overflow.sort(key=lambda x: x[0].bbox[1] if x[0].bbox else 0)
    for ri, item_data, env in overflow:
        placed = False
        for ck in ('eng_consoles', 'sci_consoles', 'tac_consoles'):
            bucket = build[env].get(ck, [])
            ni = next_idx[ck]
            if ni < len(bucket):
                bucket[ni] = item_data
                next_idx[ck] = ni + 1
                report.n_equipment += 1
                report.overflow_consoles += 1
                log.info(f'build_writer: {ri.slot}[{ri.slot_index}] overflow → {ck}[{ni}]')
                placed = True
                break
        if not placed:
            log.warning(f'build_writer: no console slot for overflow {ri.name!r}')
            report.unmatched_items += 1


# ── BOFFs ───────────────────────────────────────────────────────────


_SPEC_TO_PROF = {
    'Temporal Operative': 'Temporal',
    'Command':            'Command',
    'Miracle Worker':     'Miracle Worker',
    'Intelligence':       'Intelligence',
    'Pilot':              'Pilot',
}


def _write_boffs(build: dict, boff_items: list[RecognisedItem], ship_data: dict | None,
                 cache, is_ground: bool, report: WriteReport):
    seats_visual, boffs_build, visual_to_seat_id = _prepare_seats(build, ship_data, is_ground)
    if not seats_visual:
        return

    clusters = _cluster_boff_items(boff_items)
    if not clusters:
        return

    assigned, cluster_info = _match_clusters_to_seats(clusters, seats_visual, cache)
    if not assigned:
        log.warning('build_writer: no BOFF clusters matched to seats')
        return

    _write_abilities(build, assigned, cluster_info, seats_visual,
                     visual_to_seat_id, boffs_build, cache, is_ground, report)


def _prepare_seats(build: dict, ship_data: dict | None, is_ground: bool):
    if is_ground:
        boffs_build = build['ground']['boffs']
        n = len(boffs_build)
        # Ground seats are all rank 4 in SETS' empty build, profession fixed.
        seats_visual = [(4, build['ground']['boff_profs'][i], build['ground']['boff_specs'][i])
                        for i in range(n)]
        return seats_visual, boffs_build, {i: i for i in range(n)}

    if not ship_data or not ship_data.get('boffs'):
        log.warning('build_writer: space build but ship_data has no boffs list')
        return [], [], {}

    try:
        seats_visual = [_get_boff_spec(s) for s in ship_data['boffs']]
    except Exception as e:
        log.warning(f'build_writer: could not parse seat specs: {e}')
        return [], [], {}

    # SETS sorts seats descending by rank for display; map visual index →
    # storage index so the dict we emit matches what SETS expects.
    sorted_ix = sorted(enumerate(seats_visual), key=lambda p: p[1], reverse=True)
    visual_to_seat_id = {vis_i: seat_id for seat_id, (vis_i, _) in enumerate(sorted_ix)}
    boffs_build = build['space']['boffs']
    return seats_visual, boffs_build, visual_to_seat_id


def _cluster_boff_items(boff_items: list[RecognisedItem]) -> list[list[RecognisedItem]]:
    with_bbox = [ri for ri in boff_items if ri.bbox]
    if not with_bbox:
        return []

    by_y = sorted(with_bbox, key=lambda ri: ri.bbox[1])
    y_bands: list[list[RecognisedItem]] = []
    for ri in by_y:
        if not y_bands or ri.bbox[1] - y_bands[-1][-1].bbox[1] > config.BOFF_Y_THRESHOLD_PX:
            y_bands.append([ri])
        else:
            y_bands[-1].append(ri)

    clusters: list[list[RecognisedItem]] = []
    for band in y_bands:
        x_sorted = sorted(band, key=lambda ri: ri.bbox[0])
        cur = [x_sorted[0]]
        for ri in x_sorted[1:]:
            prev_right = cur[-1].bbox[0] + cur[-1].bbox[2]
            if ri.bbox[0] - prev_right > config.BOFF_X_THRESHOLD_PX:
                clusters.append(cur)
                cur = [ri]
            else:
                cur.append(ri)
        clusters.append(cur)

    log.info(f'build_writer: {len(with_bbox)} boff items → '
             f'{len(y_bands)} Y-bands → {len(clusters)} clusters')
    return clusters


def _match_clusters_to_seats(
    clusters: list[list[RecognisedItem]], seats_visual: list, cache,
) -> tuple[dict[int, int], list]:
    """Greedy seat-marker affinity matcher.

    For each (ship_seat, cluster) pair we compute an integer affinity
    tier (lower = better fit) using the cluster's `seat_key` marker
    FIRST, then falling back to content vote. A single global greedy
    assignment by `(tier, |rank - active|)` places marker-exact
    matches before fuzzy ones — so a screenshot's `[U]`-marker'd
    cluster lands on the ship's Universal seat, a `[T+Tem]`-marker'd
    cluster on the ship's Tac+Tem seat, etc., regardless of what the
    cluster's content vote happens to be. A final mop-up pass catches
    orphan clusters when ship/screenshot disagree on seat layout (e.g.
    a U-marker'd cluster was detected but the ship has no U seat) so
    no recognised abilities are silently dropped.

    Affinity tiers (lower = better):
      0  exact marker match (cluster's (prof, spec) align with seat's)
      1  cluster has extra spec the seat doesn't need (still slottable)
      2  Universal cluster ↔ Universal seat with spec mismatch
      3  explicit cluster's spec stripe matches a U+spec seat
      4  legacy/no-marker cluster, content vote matches seat prof
      5  explicit cluster as fallback for a U seat
      7  U-marker'd cluster going to explicit-prof seat by content vote
      8  spec-prof fallback (legacy 'Boff <spec>' cluster → spec seat)
      99 mop-up (no affinity but seats/clusters left over)
    """
    from warp.recognition.boff_keys import (
        parse_seat_profession, parse_seat_spec, is_seat_keyed, is_ground_seat,
    )

    cluster_info = []
    for c in clusters:
        # Vote on seat_key across non-virtual items so one mis-stamped
        # item can't hijack a cluster's identity. Empty seat_keys are
        # dropped from the vote so legacy items (which carry no seat_key)
        # don't dilute it.
        sks = [getattr(ri, 'seat_key', '') or '' for ri in c
               if ri.name and ri.name not in VIRTUAL_ITEM_NAMES]
        sks = [sk for sk in sks if sk]
        if sks:
            cluster_slot = Counter(sks).most_common(1)[0][0]
        else:
            cluster_slot = c[0].slot

        marker_prof = parse_seat_profession(cluster_slot)
        marker_spec = parse_seat_spec(cluster_slot)
        # A "Universal-marker'd" cluster: seat-keyed (Boff Seat …) with
        # no base profession code, and not a Ground seat.
        marker_universal = (
            is_seat_keyed(cluster_slot)
            and marker_prof is None
            and not is_ground_seat(cluster_slot)
        )

        content_profs: list[str] = []
        for ri in c:
            if ri.confidence < _MIN_CONFIDENCE_PROFESSION or not ri.name:
                continue
            for domain in ('space', 'ground'):
                found = False
                for career, ranks in cache.boff_abilities.get(domain, {}).items():
                    for rank_dict in ranks:
                        if isinstance(rank_dict, dict) and ri.name in rank_dict:
                            content_profs.append(career)
                            found = True
                            break
                    if found:
                        break
                if found:
                    break

        prof_set = set(content_profs)
        content_vote = (Counter(content_profs).most_common(1)[0][0]
                        if content_profs else 'Unknown')
        # `base_prof` preserves the legacy meaning consumed by
        # `_write_abilities` (Universal-seat promotion): marker base
        # when known, else the content vote.
        base_prof = marker_prof if marker_prof else content_vote
        active = sum(1 for ri in c if ri.name and ri.name not in VIRTUAL_ITEM_NAMES)
        cluster_info.append({
            'items':            c,
            'base_prof':        base_prof,
            'prof_set':         prof_set,
            'spec_prof':        marker_spec,
            'active':           active,
            'marker_universal': marker_universal,
            'marker_prof':      marker_prof,
            'content_vote':     content_vote,
        })

    def _affinity(seat_prof: str, seat_spec_prof: str | None, ci: int) -> int | None:
        info    = cluster_info[ci]
        m_prof  = info['marker_prof']
        m_uni   = info['marker_universal']
        m_spec  = info['spec_prof']
        c_base  = info['base_prof']
        c_profs = info['prof_set']

        if seat_prof == 'Universal':
            if m_uni:
                # U-marker'd cluster ↔ U seat — strongest preference.
                if seat_spec_prof:
                    if m_spec == seat_spec_prof:
                        return 0
                    return 2
                if not m_spec:
                    return 0
                return 1  # cluster has spec, U-no-spec seat — abilities fit
            # Explicit cluster going to a U seat (used as fallback when
            # there is no U cluster to receive the U seat).
            if seat_spec_prof:
                if m_spec == seat_spec_prof:
                    return 3
                if seat_spec_prof in c_profs:
                    return 4
                return 5
            return 5

        # Explicit-profession ship seat (Tactical / Engineering / Science)
        if m_prof == seat_prof:
            if seat_spec_prof:
                if m_spec == seat_spec_prof:
                    return 0
                if not m_spec:
                    return 0 if seat_spec_prof in c_profs else 2
                return None  # cluster spec mismatches the seat's spec
            if not m_spec:
                return 0
            return 1  # cluster carries extra spec, seat is plain
        if m_uni:
            # U-marker'd cluster going to an explicit seat — only via
            # content vote, and only as a last resort below mop-up tier.
            if c_base == seat_prof:
                return 7
            return None
        if m_prof is None:
            # Legacy / 'Boff Temporal'-style cluster, content vote
            if c_base == seat_prof:
                return 4
            if seat_spec_prof and c_base == seat_spec_prof:
                return 8
            return None
        return None

    assigned: dict[int, int] = {}
    used_ci: set[int] = set()

    def _greedy_assign(cands: list[tuple]) -> None:
        cands.sort()
        for tup in cands:
            vis_i, ci = tup[-2], tup[-1]
            if vis_i in assigned or ci in used_ci:
                continue
            assigned[vis_i] = ci
            used_ci.add(ci)

    # Main pass: global affinity-based greedy assignment.
    candidates: list[tuple[int, int, int, int, int]] = []
    for vis_i, (rank, prof, spec) in enumerate(seats_visual):
        seat_spec_prof = _SPEC_TO_PROF.get(spec) if spec else None
        for ci in range(len(cluster_info)):
            aff = _affinity(prof, seat_spec_prof, ci)
            if aff is None:
                continue
            cost = abs(rank - cluster_info[ci]['active'])
            candidates.append((aff, cost, vis_i, vis_i, ci))
    _greedy_assign(candidates)

    # Mop-up: any orphaned cluster goes into the best-size leftover seat.
    # Reached only when affinity logic under-populates (e.g. ship has no
    # U seat but a U-marker'd cluster was detected, or content vote
    # produced 'Unknown'). Better to surface the abilities somewhere than
    # to silently drop them.
    candidates = []
    for vis_i, (rank, prof, spec) in enumerate(seats_visual):
        if vis_i in assigned:
            continue
        for ci in range(len(cluster_info)):
            if ci in used_ci:
                continue
            cost = abs(rank - cluster_info[ci]['active'])
            candidates.append((99, cost, vis_i, vis_i, ci))
    _greedy_assign(candidates)

    # Build the legacy 6-tuple cluster_info shape expected by
    # `_write_abilities` so its `cluster_info[ci][N]` reads continue to
    # work without churn in unrelated code.
    legacy_ci = [
        [info['items'], info['base_prof'], info['prof_set'],
         info['spec_prof'], info['active'], info['marker_universal']]
        for info in cluster_info
    ]
    return assigned, legacy_ci


_BASE_PROFS = {'Tactical', 'Engineering', 'Science'}


def _write_abilities(build: dict, assigned: dict[int, int], cluster_info: list,
                     seats_visual: list, visual_to_seat_id: dict,
                     boffs_build: list, cache, is_ground: bool,
                     report: WriteReport):
    all_boff_cache = cache.boff_abilities.get('all', {})
    written = 0

    for vis_i, ci in assigned.items():
        seat_id = visual_to_seat_id.get(vis_i)
        if seat_id is None or seat_id >= len(boffs_build):
            continue

        rank, profession, spec = seats_visual[vis_i]
        primary_prof = cluster_info[ci][1]
        cluster_items = sorted(cluster_info[ci][0], key=lambda ri: ri.bbox[0])

        # Universal seat takes on its cluster's profession (the same
        # promotion sets-warp does for the dropdown). For ground that
        # means rewriting boff_profs; for space we rewrite boff_specs[i][0].
        if profession == 'Universal' and primary_prof in _BASE_PROFS:
            if is_ground:
                build['ground']['boff_profs'][seat_id] = primary_prof
            else:
                specs = build['space']['boff_specs']
                if seat_id < len(specs) and isinstance(specs[seat_id], list):
                    specs[seat_id] = [primary_prof, spec or '']

        direct = [ri.slot_index for ri in cluster_items]
        if all(0 <= si < rank for si in direct) and len(set(direct)) == len(direct):
            slot_indices = direct
        else:
            slot_indices = _slot_indices_from_x(cluster_items, rank)

        for ri, slot_idx in zip(cluster_items, slot_indices):
            if slot_idx >= rank or ri.name in VIRTUAL_ITEM_NAMES:
                continue
            if ri.name not in all_boff_cache:
                continue
            boffs_build[seat_id][slot_idx] = {'item': ri.name}
            written += 1

    report.n_boff_abilities = written
    log.info(f'build_writer: wrote {written} BOFF abilities')


def _slot_indices_from_x(items: list[RecognisedItem], rank: int) -> list[int]:
    xs = [ri.bbox[0] for ri in items]
    if len(xs) <= 1:
        return [0]
    gaps = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
    step = min(gaps)
    if step <= 0:
        return list(range(len(xs)))
    indices = [0]
    for gap in gaps:
        jump = max(1, round(gap / step))
        indices.append(min(indices[-1] + jump, rank - 1))
    return indices
