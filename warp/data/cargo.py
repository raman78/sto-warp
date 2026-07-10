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

UPSTREAM_BASE = 'https://raw.githubusercontent.com/STOCD/SETS-Data/main/cargo'

# Raw filename → bucketed cache key handled by this module.
RAW_FILES: tuple[str, ...] = (
    'equipment.json',
    'ship_list.json',
    'boff_abilities.json',
    'traits.json',
    'starship_traits.json',
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


def _fetch(name: str, *, etag: str | None = None) -> tuple[bytes | None, str | None]:
    """Download `name`. Returns (bytes_or_None, etag_or_None).

    Returns (None, etag) on HTTP 304. Raises on transport errors.
    """
    url = f'{UPSTREAM_BASE}/{name}'
    req = urllib.request.Request(url)
    if etag:
        req.add_header('If-None-Match', etag)
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return resp.read(), resp.headers.get('ETag')
    except urllib.error.HTTPError as e:
        if e.code == 304:
            return None, etag
        raise


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
        payload, etag = _fetch(name)
        if payload is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(payload)
            _write_meta(name, {'etag': etag, 'fetched_at': int(time.time())})
            log.info(f'cargo: fetched {name} ({len(payload)} B) from STOCD/SETS-Data')
            return payload
    except Exception as exc:
        log.warning(f'cargo: live fetch of {name} failed ({exc}); falling back to baseline')

    baseline_path = _BASELINE_DIR / name
    if baseline_path.exists():
        log.info(f'cargo: serving {name} from bundled baseline (offline mode)')
        return baseline_path.read_bytes()

    raise RuntimeError(
        f'cargo: cannot resolve {name} — no cache, no network, no baseline.'
    )


def _load_raw(name: str) -> Any:
    with _lock:
        if name in _MEMO:
            return _MEMO[name]
        raw = _resolve_raw(name)
        parsed = json.loads(raw.decode('utf-8'))
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
            payload, new_etag = _fetch(name, etag=etag)
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
            _write_meta(name, {'etag': new_etag, 'fetched_at': now})
            log.info(f'cargo.refresh: {name} updated ({len(payload)} B)')
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
    return out


def _build_ships() -> dict[str, dict]:
    raw = _load_raw('ship_list.json')
    return {ship['Page']: ship for ship in raw if ship.get('Page')}


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


def wiki_url(name: str) -> str:
    """STO Wiki URL for an item / trait / ability by name."""
    from urllib.parse import quote_plus
    return f'https://stowiki.net/wiki/{quote_plus(name.replace(" ", "_"))}'


def vger_url(slot: str) -> str | None:
    """Vger category-page URL for *slot*, or ``None`` for BOFF / unknown."""
    page = _SLOT_TO_VGER_PAGE.get(slot)
    if page:
        return f'https://vger.stobuilds.com/{page}'
    return None


def ref_icon_path(name: str) -> Path | None:
    """Path to the local reference-icon PNG, or ``None`` if not cached."""
    from urllib.parse import quote_plus
    p = icons_dir() / f'{quote_plus(name)}.png'
    return p if p.is_file() else None


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
