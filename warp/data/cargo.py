"""Cargo data loader for sto-warp (strategy B per CARGO_DATA_PLAN.md).

Provides the SETS-shaped `cache.*` views that `warp.warp_importer` and
`warp.recognition.*` expect, without depending on the SETS application:

    cargo.equipment()         # {build_key: {name: item_dict}}
    cargo.ships()             # {ship_name: ship_dict}
    cargo.traits()            # {env: {trait_type: {name: trait_dict}}}
    cargo.starship_traits()   # {name: trait_dict}
    cargo.boff_abilities()    # {'space': {...}, 'ground': {...}, 'all': {...}}

Source precedence per file:
  1. `$XDG_CONFIG_HOME/warp/cache/<file>.json` (user cache)
  2. Live fetch from STOCD/SETS-Data → write to cache, use
  3. `warp/data/baseline/<file>.json` (wheel-bundled fallback)

Refresh: background ETag-aware refresh kicked off by `refresh_async()`.
Force refresh: `refresh_all(force=True)`.

Equipment bucketing mirrors `src.datafunctions.load_cargo_data` so the
existing importer code paths work unchanged.
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable

from warp.debug import syslog as log

# --- constants ----------------------------------------------------------

# Data sources, tried in order. The first is our own mirror of the wiki,
# rebuilt by hand and on no fixed schedule; the second is the community
# mirror, kept as a fallback for the day ours is broken, stale or
# unreachable. Neither is guaranteed fresh. Both are byte-compatible
# supersets of what the builders below need — verified by building every cache
# from each source and diffing all 47 buckets.
UPSTREAM_BASES: tuple[str, ...] = (
    'https://raw.githubusercontent.com/raman78/warp-cargo-data/main/cargo',
    'https://raw.githubusercontent.com/STOCD/SETS-Data/main/cargo',
)

# Kept for callers that only care about the primary source.
UPSTREAM_BASE = UPSTREAM_BASES[0]

# Raw filename → bucketed cache key handled by this module.
RAW_FILES: tuple[str, ...] = (
    'equipment.json',
    'ship_list.json',
    'boff_abilities.json',
    'traits.json',
    'starship_traits.json',
)

# Items the wiki lists on a page but never stored in a cargo table, harvested
# from that page and published beside the mirror rather than inside it. Elite
# Fleet, Colony Security and K-13 ground weapons live here: cargo has no rows
# for them, so a name read off a ground build used to validate against nothing
# and be dropped.
#
# Only our own mirror serves these — SETS-Data has no such path — and the
# whole file is optional: a missing overlay costs those names and nothing
# else, so `_resolve_raw` treats its absence as empty rather than fatal.
# Each row carries `source`, and the publisher removes any row the real cargo
# table has started carrying, so the overlay shrinks to nothing on its own.
OVERLAY_BASE = 'https://raw.githubusercontent.com/raman78/warp-cargo-data/main/scraped'

OVERLAY_FILES: tuple[str, ...] = (
    'scraped_ground_weapons.json',
)

# equipment 'type' → bucket key used by `warp_importer` (`build_key`).
# Mirrors `src.constants.EQUIPMENT_TYPES`.
EQUIPMENT_TYPES: dict[str, str] = {
    'Body Armor': 'armor',
    'EV Suit': 'ev_suit',
    'Experimental Weapon': 'experimental',
    'Ground Device': 'ground_devices',
    'Ground Weapon': 'weapons',
    'Hangar Bay': 'hangars',
    'Impulse Engine': 'engines',
    'Kit': 'kit',
    'Kit Module': 'kit_modules',
    'Personal Shield': 'personal_shield',
    'Ship Aft Weapon': 'aft_weapons',
    'Ship Deflector Dish': 'deflector',
    'Ship Device': 'devices',
    'Ship Engineering Console': 'eng_consoles',
    'Ship Fore Weapon': 'fore_weapons',
    'Ship Science Console': 'sci_consoles',
    'Ship Secondary Deflector': 'sec_def',
    'Ship Shields': 'shield',
    'Ship Tactical Console': 'tac_consoles',
    'Ship Weapon': 'ship_weapon',
    'Singularity Engine': 'core',
    'Universal Console': 'uni_consoles',
    'Warp Engine': 'core',
}

# Hangars whose advanced/elite variants the SETS loader drops.
_ELITE_HANGAR_WHITELIST = {
    'Hangar - Elite Federation Mission Scout Ships',
    'Hangar - Elite Valor Fighters',
}

# Cache freshness window. Older than this triggers a background refresh on
# the next `refresh_async()` call.
_REFRESH_TTL_SECONDS = 24 * 3600

_BASELINE_DIR = Path(__file__).resolve().parent / 'baseline'


def _cache_dir() -> Path:
    env = os.environ.get('WARP_CACHE_DIR')
    if env:
        return Path(env)
    xdg = os.environ.get('XDG_CONFIG_HOME')
    base = Path(xdg) / 'warp' if xdg else Path.home() / '.config' / 'warp'
    return base / 'cache'


def icons_dir() -> Path:
    """Local icon library used by `SETSIconMatcher` (template + histogram index).

    Order: `$WARP_ICONS_DIR` → `$XDG_CONFIG_HOME/warp/icons` → `~/.config/warp/icons`.
    """
    env = os.environ.get('WARP_ICONS_DIR')
    if env:
        return Path(env)
    xdg = os.environ.get('XDG_CONFIG_HOME')
    base = Path(xdg) / 'warp' if xdg else Path.home() / '.config' / 'warp'
    return base / 'icons'


def ship_images_dir() -> Path:
    """Local ship-image library mirrored from STOCD/SETS-Data ship_images/."""
    env = os.environ.get('WARP_SHIP_IMAGES_DIR')
    if env:
        return Path(env)
    xdg = os.environ.get('XDG_CONFIG_HOME')
    base = Path(xdg) / 'warp' if xdg else Path.home() / '.config' / 'warp'
    return base / 'ship_images'


# --- raw fetch / cache primitives ---------------------------------------

_lock = threading.RLock()
_MEMO: dict[str, Any] = {}            # raw file name -> parsed JSON
_BUCKET_MEMO: dict[str, Any] = {}     # bucket key (e.g. 'equipment') -> shaped data


def _meta_path(name: str) -> Path:
    return _cache_dir() / f'{name}.meta'


def _read_meta(name: str) -> dict:
    p = _meta_path(name)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding='utf-8'))
        except Exception:
            return {}
    return {}


def _write_meta(name: str, meta: dict) -> None:
    p = _meta_path(name)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    except Exception as exc:
        log.warning(f'cargo: cannot write meta for {name}: {exc}')


def _assert_usable(name: str, payload: bytes) -> None:
    """Raise unless `payload` is a non-empty JSON document.

    A mirror can answer 200 with a truncated body or an error page. Without
    this the bytes would be cached, and the failure would only surface later
    as a parse error — with the cache already poisoned and the fallback source
    never consulted.
    """
    try:
        parsed = json.loads(payload.decode('utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f'{name} is not valid JSON ({exc})') from exc
    if not isinstance(parsed, (list, dict)) or not parsed:
        raise ValueError(f'{name} decoded to an empty or unexpected structure')


def _fetch(name: str, *, etag: str | None = None,
           source: str | None = None) -> tuple[bytes | None, str | None, str]:
    """Download `name`, falling back through `UPSTREAM_BASES`.

    Returns (bytes_or_None, etag_or_None, base_it_came_from); the payload is
    None on HTTP 304. Raises only if every source failed.

    `source` is the base the caller's `etag` was issued by. An ETag is
    meaningless to a different server, so it is only replayed against its own
    source — otherwise a fallback could answer 304 and leave us with the
    other mirror's stale bytes.
    """
    errors: list[str] = []
    bases = (OVERLAY_BASE,) if name in OVERLAY_FILES else UPSTREAM_BASES
    for base in bases:
        req = urllib.request.Request(f'{base}/{name}')
        if etag and source == base:
            req.add_header('If-None-Match', etag)
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                payload = resp.read()
            _assert_usable(name, payload)
            return payload, resp.headers.get('ETag'), base
        except urllib.error.HTTPError as e:
            if e.code == 304:
                return None, etag, base
            errors.append(f'{base}: HTTP {e.code}')
        except Exception as exc:                      # transport or content
            errors.append(f'{base}: {exc}')
        log.warning(f'cargo: {name} unavailable from {base} — trying next source')
    raise RuntimeError(f'cargo: no source served {name} ({"; ".join(errors)})')


def _resolve_raw(name: str) -> bytes:
    """Return raw bytes for `name`, using cache → live fetch → baseline.

    Updates the cache when a live fetch succeeds. Never writes to baseline.
    """
    cache_path = _cache_dir() / name
    if cache_path.exists():
        try:
            return cache_path.read_bytes()
        except Exception as exc:
            log.warning(f'cargo: cache read failed for {name}: {exc}')

    # No cache yet — try live fetch.
    try:
        payload, etag, base = _fetch(name)
        if payload is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(payload)
            _write_meta(name, {'etag': etag, 'fetched_at': int(time.time()),
                               'source': base})
            log.info(f'cargo: fetched {name} ({len(payload)} B) from {base}')
            return payload
    except Exception as exc:
        log.warning(f'cargo: live fetch of {name} failed ({exc}); falling back to baseline')

    baseline_path = _BASELINE_DIR / name
    if baseline_path.exists():
        log.info(f'cargo: serving {name} from bundled baseline (offline mode)')
        return baseline_path.read_bytes()

    if name in OVERLAY_FILES:
        # The overlay is additive: without it those names simply stay
        # unknown, which is the state every release before it shipped.
        log.warning(f'cargo: overlay {name} unavailable — continuing without it')
        return b'[]'

    raise RuntimeError(
        f'cargo: cannot resolve {name} — no cache, no network, no baseline.'
    )


def _load_raw(name: str) -> Any:
    with _lock:
        if name in _MEMO:
            return _MEMO[name]
        raw = _resolve_raw(name)
        try:
            parsed = json.loads(raw.decode('utf-8'))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            # A cache file written before this was validated — or damaged on
            # disk — would otherwise fail here on every run forever. Drop it
            # and resolve once more, which re-fetches and revalidates.
            log.warning(f'cargo: cached {name} is unreadable ({exc}); '
                        f'discarding and refetching')
            try:
                (_cache_dir() / name).unlink(missing_ok=True)
            except OSError:
                pass
            parsed = json.loads(_resolve_raw(name).decode('utf-8'))
        _MEMO[name] = parsed
        return parsed


# --- background refresh -------------------------------------------------

def refresh_async(names: Iterable[str] | None = None) -> None:
    """Kick off an ETag-aware refresh in a daemon thread.

    Stale (older than `_REFRESH_TTL_SECONDS`) or unknown files only.
    """
    targets = tuple(names) if names else RAW_FILES
    threading.Thread(target=_refresh_loop, args=(targets, False), daemon=True).start()


def refresh_all(*, force: bool = False) -> None:
    """Blocking refresh of every known file.

    `force=True` ignores ETag and freshness window — used by
    `sto-warp data refresh`.
    """
    _refresh_loop(RAW_FILES, force)


def _refresh_loop(names: Iterable[str], force: bool) -> None:
    now = int(time.time())
    for name in names:
        meta = _read_meta(name)
        cache_path = _cache_dir() / name
        if not force and cache_path.exists():
            age = now - int(meta.get('fetched_at', 0))
            if age < _REFRESH_TTL_SECONDS:
                log.info(f'cargo.refresh: {name} fresh ({age // 3600}h old, '
                         f'TTL {_REFRESH_TTL_SECONDS // 3600}h) — skipped')
                continue
        etag = None if force else meta.get('etag')
        try:
            payload, new_etag, base = _fetch(name, etag=etag,
                                             source=meta.get('source'))
        except Exception as exc:
            log.warning(f'cargo.refresh: {name} failed: {exc}')
            continue
        if payload is None:
            _write_meta(name, {**meta, 'fetched_at': now})
            log.info(f'cargo.refresh: {name} unchanged (HTTP 304)')
            continue
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(payload)
            _write_meta(name, {'etag': new_etag, 'fetched_at': now,
                               'source': base})
            log.info(f'cargo.refresh: {name} updated ({len(payload)} B) from {base}')
        except Exception as exc:
            log.warning(f'cargo.refresh: cannot write {name}: {exc}')
            continue
        # Invalidate memoized views that derive from this file.
        with _lock:
            _MEMO.pop(name, None)
            _BUCKET_MEMO.clear()

    # Surface any shape drift between upstream and consumer contracts
    # as a WARNING. Without this, the cache-read sites in warp_importer /
    # build_writer / sets_export silently swallow AttributeError and the
    # drift only shows up as degraded recognition (see v1.0.16 BOFF
    # regression).
    try:
        validate(on_problem='warn')
    except Exception as exc:
        log.warning(f'cargo.refresh: validate skipped — {exc}')


# --- bucketed accessors (SETS cache shape) ------------------------------

def _bucketed(key: str, build_fn) -> Any:
    with _lock:
        if key in _BUCKET_MEMO:
            return _BUCKET_MEMO[key]
        value = build_fn()
        _BUCKET_MEMO[key] = value
        return value


def equipment() -> dict[str, dict[str, dict]]:
    """`{build_key: {item_name: item_dict}}` — mirrors `cache.equipment`."""
    return _bucketed('equipment', _build_equipment)


def ships() -> dict[str, dict]:
    """`{ship_name: ship_dict}` — keyed by `Page` (canonical name)."""
    return _bucketed('ships', _build_ships)


def traits() -> dict[str, dict[str, dict[str, dict]]]:
    """`{env: {trait_type: {name: trait_dict}}}` — trait_type in
    `personal | rep | active_rep`. env in `space | ground`."""
    return _bucketed('traits', _build_traits)


def starship_traits() -> dict[str, dict]:
    """`{name: trait_dict}` flat — same shape SETS uses."""
    return _bucketed('starship_traits', _build_starship_traits)


def boff_abilities() -> dict:
    """`{'space': {prof: [rank_dict]}, 'ground': {...}, 'all': {name: dict}}`.

    See `_build_boff_abilities` for the bucketing rationale."""
    return _bucketed('boff_abilities', _build_boff_abilities)


def all_caches() -> dict[str, Any]:
    """Bundle all five accessors. Useful for wiring into legacy code that
    expects a single `cache`-like object."""
    return {
        'equipment': equipment(),
        'ships': ships(),
        'traits': traits(),
        'starship_traits': starship_traits(),
        'boff_abilities': boff_abilities(),
    }


# Captain specializations — not in cargo (hardcoded in WARP CORE), but valid
# recognisable labels, so they belong in the canonical name set.
SPECIALIZATION_NAMES: frozenset[str] = frozenset({
    'Command Officer', 'Intelligence Officer', 'Miracle Worker', 'Pilot',
    'Temporal Operative', 'Constable', 'Commando', 'Strategist',
})


def canonical_names() -> set[str]:
    """Flat set of every valid item / ability / trait / specialization name.

    Single source of truth for anything that needs to validate a label
    against "the things WARP can recognise" — e.g. a maintainer tool that
    relabels a mislabeled crop and must reject typos. Built from the same
    cargo accessors the client uses, so it tracks upstream data over time
    with no duplicated parsing. Each source is guarded independently: a
    shape drift in one file degrades that source's contribution rather than
    raising (mirrors the try/except reads elsewhere in this module)."""
    names: set[str] = set()

    try:  # equipment(): {build_key: {name: item_dict}}
        for bucket in equipment().values():
            if isinstance(bucket, dict):
                names.update(k for k in bucket if isinstance(k, str) and k)
    except Exception as exc:
        log.warning(f'cargo.canonical_names: equipment failed: {exc!r}')

    try:  # boff_abilities(): {..., 'all': {name: dict}}
        all_ab = boff_abilities().get('all', {})
        if isinstance(all_ab, dict):
            names.update(k for k in all_ab if isinstance(k, str) and k)
    except Exception as exc:
        log.warning(f'cargo.canonical_names: boff_abilities failed: {exc!r}')

    try:  # traits(): {env: {trait_type: {name: trait_dict}}}
        for env in traits().values():
            if not isinstance(env, dict):
                continue
            for tt in env.values():
                if isinstance(tt, dict):
                    names.update(k for k in tt if isinstance(k, str) and k)
    except Exception as exc:
        log.warning(f'cargo.canonical_names: traits failed: {exc!r}')

    try:  # starship_traits(): {name: trait_dict}
        names.update(k for k in starship_traits() if isinstance(k, str) and k)
    except Exception as exc:
        log.warning(f'cargo.canonical_names: starship_traits failed: {exc!r}')

    names.update(SPECIALIZATION_NAMES)
    names.discard('')
    return names


# --- shape validation ---------------------------------------------------

def _shape_problems() -> list[str]:
    """Return a list of human-readable shape violations for every cache.

    Empty list = every consumer-visible invariant holds. Each call site in
    `warp_importer`, `build_writer`, `sets_export`, `trainer_window` wraps
    cache reads in `try/except: pass` — so a drift between upstream JSON
    and the builder doesn't crash, it silently degrades recognition. This
    function makes the silent drift observable.
    """
    problems: list[str] = []

    def _check(name: str, fn):
        try:
            fn()
        except Exception as exc:
            problems.append(f'{name}: {exc!r}')

    def _check_equipment():
        eq = equipment()
        assert isinstance(eq, dict) and eq, 'equipment: empty or not dict'
        # build_writer / warp_importer iterate eq.values(); each value must
        # be {item_name: item_dict} so `.get(name)` and `entry.get('type')`
        # work without try/except.
        for bucket, items in eq.items():
            assert isinstance(items, dict), f'equipment[{bucket}!r]: not dict'

    def _check_ships():
        ss = ships()
        assert isinstance(ss, dict) and ss, 'ships: empty or not dict'
        sample = next(iter(ss.values()))
        assert isinstance(sample, dict), 'ships: values not dicts'

    def _check_traits():
        tr = traits()
        assert set(tr.keys()) >= {'space', 'ground'}, 'traits: missing env'
        for env in ('space', 'ground'):
            assert set(tr[env].keys()) >= {'personal', 'rep', 'active_rep'}, \
                f'traits[{env}]: missing trait kind'

    def _check_starship_traits():
        st = starship_traits()
        assert isinstance(st, dict) and st, 'starship_traits: empty'

    def _check_boff_abilities():
        bo = boff_abilities()
        assert set(bo.keys()) >= {'space', 'ground', 'all'}, \
            f'boff_abilities: top keys {sorted(bo.keys())} != space/ground/all'
        assert bo['all'], 'boff_abilities[all]: empty'
        sample_name, sample_info = next(iter(bo['all'].items()))
        assert isinstance(sample_info, dict) and 'profession' in sample_info, \
            f'boff_abilities[all][{sample_name!r}]: no profession field'
        # Each env bucket must be {profession: [rank_dict, ...]} — the
        # shape `_lookup_boff_profession` and `_item_valid_for_slot` walk.
        for env in ('space', 'ground'):
            assert isinstance(bo[env], dict), f'boff_abilities[{env}]: not dict'
            for prof, ranks in bo[env].items():
                assert isinstance(ranks, list) and ranks, \
                    f'boff_abilities[{env}][{prof}]: not non-empty list'
                assert isinstance(ranks[0], dict), \
                    f'boff_abilities[{env}][{prof}][0]: not dict'

    _check('equipment',       _check_equipment)
    _check('ships',           _check_ships)
    _check('traits',          _check_traits)
    _check('starship_traits', _check_starship_traits)
    _check('boff_abilities',  _check_boff_abilities)
    return problems


def validate(*, on_problem: str = 'warn') -> list[str]:
    """Load every cache once and check shape invariants.

    `on_problem='warn'` logs each violation at WARNING level and returns
    the list. `on_problem='raise'` raises `ValueError` on the first
    violation — used by the CI shape test. `on_problem='silent'` returns
    the list with no logging — for callers that want to format their own
    report (e.g. a future `sto-warp data verify` CLI).
    """
    problems = _shape_problems()
    if not problems:
        log.info(f'cargo.validate: all 5 caches OK')
        return problems
    if on_problem == 'raise':
        raise ValueError(f'cargo shape violation: {problems[0]}')
    if on_problem == 'warn':
        for p in problems:
            log.warning(f'cargo.validate: shape drift — {p}')
    return problems


class _CacheView:
    """Drop-in stand-in for the SETS `app.cache` object.

    Attribute reads are lazy — each accessor only builds its bucket when
    first touched, so callers that only need one or two slices don't pay
    for the rest. Returned dicts are the cached singletons from cargo's
    bucket memo (`_BUCKET_MEMO`), so mutating them mutates the global
    view; treat them as read-only.
    """

    __slots__ = ()

    @property
    def equipment(self) -> dict[str, dict[str, dict]]:
        return equipment()

    @property
    def ships(self) -> dict[str, dict]:
        return ships()

    @property
    def traits(self) -> dict[str, dict[str, dict[str, dict]]]:
        return traits()

    @property
    def starship_traits(self) -> dict[str, dict]:
        return starship_traits()

    @property
    def boff_abilities(self) -> dict:
        return boff_abilities()


def cache_view() -> _CacheView:
    """Return a `cache`-shaped object (SETS-compatible attribute access).

    Allows existing call sites (`app.cache.equipment`, etc.) to keep
    working with cargo as the data source.
    """
    return _CacheView()


def app_view():
    """Minimal SETS-app stand-in for trainer call sites.

    Trainer code reads `self._sets.cache.X` in dozens of places and
    sometimes attaches extras (e.g. `self._sets._warp_core_window = self`
    so the sync worker can locate the live `TrainingDataManager`).
    Returning a mutable `SimpleNamespace` lets standalone callers swap
    the SETS app for this shim with zero call-site changes.
    """
    import types
    return types.SimpleNamespace(cache=cache_view())


# --- builders -----------------------------------------------------------

def _sanitize_equipment_name(name: str) -> str:
    """Strip cargo-data modifier suffixes so cache keys match icon labels.

    Mirrors `src.textedit.sanitize_equipment_name` from sets-warp — the
    embedder and pHash knowledge ship clean names, so equipment cache keys
    must match (otherwise candidate-name filtering rejects every ML hit).
    """
    name = name.replace('&quot;', '"').replace('&#34;', '"')
    for sep in ('∞', 'Mk X', 'MK X', '['):
        if sep in name:
            name = name.split(sep, 1)[0]
    if name.endswith('-S'):
        name = name[:-2]
    return name.strip()


def _build_equipment() -> dict[str, dict[str, dict]]:
    raw = _load_raw('equipment.json')
    out: dict[str, dict[str, dict]] = {bk: {} for bk in EQUIPMENT_TYPES.values()}
    for item in raw:
        kind = item.get('type')
        bucket = EQUIPMENT_TYPES.get(kind)
        if bucket is None:
            continue
        raw_name = item.get('name')
        if not raw_name:
            continue
        if kind == 'Hangar Bay' and raw_name not in _ELITE_HANGAR_WHITELIST and (
                raw_name.startswith('Hangar - Advanced')
                or raw_name.startswith('Hangar - Elite')):
            continue
        name = _sanitize_equipment_name(raw_name)
        if not name:
            continue
        out[bucket][name] = item

    # Replicate SETS post-processing: ship_weapon fans out into fore/aft,
    # universal consoles fan out across tac/sci/eng (and vice versa).
    ship_weapon = out.pop('ship_weapon', {})
    out['fore_weapons'].update(ship_weapon)
    out['aft_weapons'].update(ship_weapon)
    tac, sci, eng, uni = (out['tac_consoles'], out['sci_consoles'],
                          out['eng_consoles'], out['uni_consoles'])
    tac.update(uni)
    sci.update(uni)
    eng.update(uni)
    uni.update(tac)
    uni.update(sci)
    uni.update(eng)

    _merge_overlay(out)
    return out


def _merge_overlay(buckets: dict[str, dict[str, dict]]) -> None:
    """Add harvested items to their bucket, never over a real cargo row.

    The overlay exists because the wiki lists these items without storing
    them (see `OVERLAY_FILES`). A name the real table already carries wins:
    the publisher drops such rows on its next run anyway, and until then a
    stale overlay entry must not shadow the authoritative one.

    Note what this does and does not buy. Fleet and colony weapons reuse the
    base weapon's picture, so icon matching cannot tell them apart and this
    changes nothing there. What it changes is validation: a name read off a
    ground build now resolves instead of being discarded as unknown.
    """
    try:
        rows = _load_raw('scraped_ground_weapons.json')
    except Exception as exc:
        log.warning(f'cargo: overlay unavailable ({exc}) — continuing without it')
        return
    if not isinstance(rows, list):
        return

    added = 0
    for item in rows:
        if not isinstance(item, dict):
            continue
        bucket = EQUIPMENT_TYPES.get(item.get('type'))
        name = _sanitize_equipment_name(item.get('name') or '')
        if bucket is None or not name or name in buckets.get(bucket, {}):
            continue
        buckets[bucket][name] = item
        added += 1
    if added:
        log.info(f'cargo: overlay added {added} items the cargo tables lack')


# `ship_list.json` is the one file whose field *types* differ per source.
# `STOCD/SETS-Data` (and the bundled baseline) serve typed JSON; the
# `raman78/warp-cargo-data` mirror serves the raw Special:CargoExport
# output, where every field is a string and list-valued columns arrive
# comma-joined. Consumers must not have to ask which source answered:
# `for seat in ship['boffs']` yields characters instead of seats, and a
# string `tier` breaks any arithmetic on the slot profile.
#
# The field lists below were measured by diffing all 797 ships from both
# sources; the other four cargo files carry no type conflicts (the mirror
# only adds fields SETS-Data lacks).
_SHIP_LIST_FIELDS = ('boffs', 'type', 'abilities')
_SHIP_NUMERIC_FIELDS = (
    'aft', 'consoleseng', 'consolessci', 'consolestac', 'devices',
    'experimental', 'fc', 'fore', 'hangars', 'hull', 'hullmod', 'impulse',
    'inertia', 'powerall', 'powerauxiliary', 'powerboost', 'powerengines',
    'powershields', 'powerweapons', 'secdeflector', 'shieldmod', 'tier',
    'turnrate',
)


def _as_number(value: str) -> int | float | str:
    """`'5'` → 5, `'1.25'` → 1.25, anything else unchanged."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    return int(number) if number.is_integer() and '.' not in value else number


def _normalise_ship(ship: dict) -> dict:
    """Coerce a raw-export ship record into the typed shape SETS uses.

    Idempotent: records that already arrived typed pass through
    untouched, so this runs safely against either source.
    """
    out = dict(ship)
    for field in _SHIP_LIST_FIELDS:
        value = out.get(field)
        if isinstance(value, str):
            value = value.split(',')
        elif value is None:
            value = []
        if isinstance(value, list):
            # Blank and whitespace-padded entries occur in both sources
            # (4 empty seats in SETS-Data, 6 in the baseline). SETS drops
            # them in `parse_boff_stations`; kept, they become phantom
            # Commander seats with no profession.
            out[field] = [str(part).strip() for part in value if str(part).strip()]
    for field in _SHIP_NUMERIC_FIELDS:
        value = out.get(field)
        if isinstance(value, str):
            out[field] = None if value == '' else _as_number(value)
    return out


def _build_ships() -> dict[str, dict]:
    raw = _load_raw('ship_list.json')
    return {ship['Page']: _normalise_ship(ship) for ship in raw if ship.get('Page')}


def _build_traits() -> dict[str, dict[str, dict[str, dict]]]:
    raw = _load_raw('traits.json')
    out: dict[str, dict[str, dict[str, dict]]] = {
        'space':  {'personal': {}, 'rep': {}, 'active_rep': {}},
        'ground': {'personal': {}, 'rep': {}, 'active_rep': {}},
    }
    for trait in raw:
        name = trait.get('name')
        kind = trait.get('type')
        env = trait.get('environment')
        if not name or env not in out or kind in (None, 'doff', 'boff'):
            continue
        if kind == 'reputation':
            tt = 'rep'
        elif kind == 'activereputation':
            tt = 'active_rep'
        else:
            tt = 'personal'
        out[env][tt][name] = trait
    return out


def _build_starship_traits() -> dict[str, dict]:
    raw = _load_raw('starship_traits.json')
    return {trait['name']: trait for trait in raw if trait.get('name')}


# The wiki spells one specialisation two ways in a BOFF ability's `type`:
# `Temporal` and `Temporal Operative`. Buckets are keyed by that value and the
# trainer looks them up from its slot name (`Boff Temporal`), so the longer
# spelling was simply unreachable — seven ground abilities, among them Causal
# Entanglement, could not be picked at all. Folding the aliases keeps every
# ability reachable from the slot that owns it, whichever spelling the wiki
# happens to use for it that week.
_BOFF_CAREER_ALIASES: dict[str, str] = {
    'Temporal Operative': 'Temporal',
}


def _build_boff_abilities() -> dict:
    """Bucketize raw `boff_abilities.json` (flat list) into the shape
    every consumer expects:

        {
          'space':  {profession: [{ability_name: ability_dict, ...}]},
          'ground': {profession: [{ability_name: ability_dict, ...}]},
          'all':    {ability_name: ability_dict},   # carries 'profession'
        }

    Per-env buckets keep one rank-dict per profession — consumers only
    test name-membership across the rank list, so the rank-index split
    is irrelevant for correctness.
    """
    raw = _load_raw('boff_abilities.json')
    out: dict = {'space': {}, 'ground': {}, 'all': {}}
    for ab in raw:
        name = ab.get('name')
        if not name:
            continue
        prof = ab.get('type') or 'Unknown'
        prof = _BOFF_CAREER_ALIASES.get(prof, prof)
        env  = 'ground' if (ab.get('region') or '').lower() == 'ground' else 'space'

        info = dict(ab)
        info.setdefault('profession', prof)

        out['all'][name] = info
        out[env].setdefault(prof, [{}])[0][name] = info
    return out


# --- external-link helpers ------------------------------------------------

_SLOT_TO_VGER_PAGE: dict[str, str] = {
    # Space equipment
    'Fore Weapons': 'space-equipment', 'Aft Weapons': 'space-equipment',
    'Deflector': 'space-equipment', 'Engines': 'space-equipment',
    'Warp Core': 'space-equipment', 'Shield': 'space-equipment',
    'Devices': 'space-equipment', 'Engineering Consoles': 'space-equipment',
    'Science Consoles': 'space-equipment', 'Tactical Consoles': 'space-equipment',
    'Universal Consoles': 'space-equipment', 'Hangars': 'space-equipment',
    'Experimental': 'space-equipment', 'Sec-Def': 'space-equipment',
    # Ground equipment
    'Kit Modules': 'ground-equipment', 'Kit': 'ground-equipment',
    'Body Armor': 'ground-equipment', 'EV Suit': 'ground-equipment',
    'Personal Shield': 'ground-equipment', 'Weapons': 'ground-equipment',
    'Ground Devices': 'ground-equipment',
    # Traits
    'Starship Traits': 'starship-traits',
    'Personal Space Traits': 'personal-traits', 'Space Reputation': 'personal-traits',
    'Active Space Rep': 'personal-traits', 'Personal Ground Traits': 'personal-traits',
    'Ground Reputation': 'personal-traits', 'Active Ground Rep': 'personal-traits',
}


# Trait slots and the (environment, category) bucket `_build_traits` files
# their contents under. Traits are where a display name alone is not enough:
# 135 of them carry two or three wiki pages — `Adaptive Offense` exists as a
# space trait and a ground trait, `Aggressive` as ground, bridge officer and
# duty officer — and the slot is what says which one is on screen. Keyed the
# same way `warp_importer._build_slot_candidates` keys `trait_slot_pools`, so
# the page a slot links to is the page of the pool it was matched against.
_TRAIT_SLOT_BUCKET: dict[str, tuple[str, str]] = {
    'Personal Space Traits':  ('space',  'personal'),
    'Personal Ground Traits': ('ground', 'personal'),
    'Space Reputation':       ('space',  'rep'),
    'Ground Reputation':      ('ground', 'rep'),
    'Active Space Rep':       ('space',  'active_rep'),
    'Active Ground Rep':      ('ground', 'active_rep'),
}


def _cargo_row(name: str, slot: str) -> dict | None:
    """The cargo row *name* came from, using *slot* to pick the table.

    Falls through to every other table when the slot says nothing useful or
    the name is not in the table it names — a misrecognised item is shown
    under the wrong slot, and looking it up on the wiki is exactly what the
    user does about that. Each source is guarded on its own, matching the
    try/except reads in `canonical_names`.
    """
    def _traits(env: str, cat: str):
        return (traits().get(env) or {}).get(cat) or {}

    def _equipment():
        return {n: row
                for key in sorted(equipment())
                for n, row in (equipment()[key] or {}).items()}

    if slot in _TRAIT_SLOT_BUCKET:
        first = [lambda: _traits(*_TRAIT_SLOT_BUCKET[slot])]
    elif slot == 'Starship Traits':
        first = [starship_traits]
    elif slot.startswith('Boff'):
        first = [lambda: boff_abilities().get('all') or {}]
    elif slot in ('Ship Type', 'Ship Name'):
        first = [ships]
    else:
        # Every remaining slot is an equipment slot. Naming it explicitly
        # rather than letting it reach equipment at the end of the fallback
        # matters: the day cargo files a trait under a name an item already
        # has, the fallback order alone would send the console to the trait's
        # page. Measured 2026-08-31 no such overlap exists — which is exactly
        # when a dependence on it is cheapest to remove.
        first = [_equipment]

    # Fallback, in a fixed order so the same name always resolves the same
    # way. Reached when the slot is unknown to cargo, or when the item is
    # shown under a slot it does not belong to — a misrecognition, which is
    # the case the user is most likely to be looking the item up about.
    rest = [lambda e=env, c=cat: _traits(e, c)
            for env, cat in (('space', 'personal'), ('ground', 'personal'),
                             ('space', 'rep'), ('ground', 'rep'),
                             ('space', 'active_rep'), ('ground', 'active_rep'))]
    rest += [starship_traits,
             lambda: boff_abilities().get('all') or {},
             _equipment,
             ships]
    lookups = first + rest

    for get in lookups:
        try:
            row = get().get(name)
        except Exception as exc:
            log.warning(f'cargo._cargo_row: lookup failed for {name!r}: {exc!r}')
            continue
        if isinstance(row, dict):
            return row
    return None


def wiki_url(name: str, slot: str = '') -> str:
    """STO Wiki URL for an item / trait / ability by name.

    The title comes from the row's `Page` field, which is the cargo built-in
    `_pageName` — the page the row was extracted from, so it always resolves.
    A name is not a title: traits and abilities carry a disambiguator the
    name does not (`Harmonic Shield Linkage (space trait)`,
    `Heisenberg Amplifier (ability)`), and an item without a page of its own
    is documented on its set's (`Console - Universal - Causal Anchor` lives
    on `31st Century Temporal Technologies Set`).

    Measured 2026-08-31 against the live wiki, sampling each table: the
    name-based title resolved for 0/40 starship traits, 4/40 BOFF abilities,
    10/40 personal traits and 85/100 equipment items. Every `Page` title
    resolved. Names cargo does not carry — skill nodes, specialisations —
    keep the name-based URL, which is all there is to go on.
    """
    from urllib.parse import quote
    row = _cargo_row(name, slot) if name else None
    page = (row or {}).get('Page') or name
    # Parentheses and apostrophes are legal in a title and the wiki leaves
    # them alone in its own links — `Harmonic_Shield_Linkage_(space_trait)`,
    # not `..._%28space_trait%29`. Both resolve; the readable one is what the
    # user sees in the address bar and what they can compare against a link
    # they were sent. Everything else stays encoded.
    return f'https://stowiki.net/wiki/{quote(page.replace(" ", "_"), safe="()\'/,")}'


def vger_url(slot: str) -> str | None:
    """Vger category-page URL for *slot*, or ``None`` for BOFF / unknown."""
    page = _SLOT_TO_VGER_PAGE.get(slot)
    if page:
        return f'https://vger.stobuilds.com/{page}'
    return None


def _build_trait_icon_aliases() -> dict[str, list[str]]:
    """Map a trait's display `name` → its `icon_name` variant(s).

    Trait icons on the SETS-Data mirror are filed under `icon_name`
    (e.g. ``Hive Defenses (space)``), which differs from the display
    `name` (``Hive Defenses``). Without this, `ref_icon_path` can't
    resolve a confirmed trait's icon because no ``<name>.png`` exists.
    """
    out: dict[str, list[str]] = {}
    for src in ('traits.json', 'starship_traits.json'):
        try:
            raw = _load_raw(src)
        except Exception:
            continue
        for trait in raw:
            name = trait.get('name')
            icn = trait.get('icon_name')
            if name and icn and icn != name:
                out.setdefault(name, [])
                if icn not in out[name]:
                    out[name].append(icn)
    return out


def _trait_icon_aliases() -> dict[str, list[str]]:
    return _bucketed('trait_icon_aliases', _build_trait_icon_aliases)


def ref_icon_path(name: str, env: str | None = None) -> Path | None:
    """Path to the local reference-icon PNG, or ``None`` if not cached.

    A handful of traits share one display `name` across both environments
    with *different* icons (e.g. 'Adaptive Offense' has 'Adaptive Offense
    (space)' and 'Adaptive Offense (ground)'). When *env* ('space'/'ground')
    is known, prefer the alias whose ``(env)`` suffix matches so a space
    trait doesn't show the ground icon (and vice-versa).
    """
    from urllib.parse import quote_plus
    d = icons_dir()
    p = d / f'{quote_plus(name)}.png'
    if p.is_file():
        return p
    # Traits are filed under `icon_name` (e.g. 'Hive Defenses (space)'),
    # not the display name — fall back to those variants so a confirmed
    # trait's icon still resolves in tooltips.
    aliases = list(_trait_icon_aliases().get(name, ()))
    if env in ('space', 'ground'):
        tag = f'({env})'
        # env-matching aliases first, preserving relative order otherwise.
        aliases.sort(key=lambda a: 0 if tag in a.lower() else 1)
    for alias in aliases:
        ap = d / f'{quote_plus(alias)}.png'
        if ap.is_file():
            return ap
    return None


# --- introspection helpers (used by `sto-warp check` / diagnostics) -----

def status() -> dict[str, Any]:
    """Per-file summary: source (cache/baseline), age, size, etag."""
    out = {}
    cache = _cache_dir()
    for name in RAW_FILES:
        cache_p = cache / name
        baseline_p = _BASELINE_DIR / name
        meta = _read_meta(name)
        out[name] = {
            'cache': str(cache_p) if cache_p.exists() else None,
            'cache_size': cache_p.stat().st_size if cache_p.exists() else 0,
            'baseline': str(baseline_p) if baseline_p.exists() else None,
            'baseline_size': baseline_p.stat().st_size if baseline_p.exists() else 0,
            'etag': meta.get('etag'),
            'fetched_at': meta.get('fetched_at'),
        }
    return out
