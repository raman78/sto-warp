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

from warp.debug import log

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
                continue
        etag = None if force else meta.get('etag')
        try:
            payload, new_etag = _fetch(name, etag=etag)
        except Exception as exc:
            log.warning(f'cargo.refresh: {name} failed: {exc}')
            continue
        if payload is None:
            _write_meta(name, {**meta, 'fetched_at': now})
            log.debug(f'cargo.refresh: {name} unchanged (304)')
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
    """`{'space': ..., 'ground': ..., 'all': ...}` — upstream shape is
    already what consumers expect, returned as-is."""
    return _bucketed('boff_abilities', lambda: _load_raw('boff_abilities.json'))


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


# --- builders -----------------------------------------------------------

def _build_equipment() -> dict[str, dict[str, dict]]:
    raw = _load_raw('equipment.json')
    out: dict[str, dict[str, dict]] = {bk: {} for bk in EQUIPMENT_TYPES.values()}
    for item in raw:
        kind = item.get('type')
        bucket = EQUIPMENT_TYPES.get(kind)
        if bucket is None:
            continue
        name = item.get('name')
        if not name:
            continue
        if kind == 'Hangar Bay' and name not in _ELITE_HANGAR_WHITELIST and (
                name.startswith('Hangar - Advanced') or name.startswith('Hangar - Elite')):
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
