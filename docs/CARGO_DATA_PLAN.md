# Cargo data loader — design note

**Status:** plan only (not implemented yet).
**Strategy chosen:** (iii) fetch from `STOCD/SETS-Data` GitHub raw URLs at
first run, cache locally.

## Motivation

sto-warp must work without the SETS build planner. Previously, cargo /
ship / trait / BOFF metadata was loaded from `~/.config/SETS/cache/`
populated by the upstream `src.datafunctions` loader. Standalone sto-warp
needs its own loader that:

- has zero dependency on SETS source,
- caches data per-user (no privileged writes inside the wheel),
- can be refreshed on demand,
- works behind a typical home network (no auth, just public HTTPS).

## Source

Public mirror maintained by the SETS community:

```
https://raw.githubusercontent.com/STOCD/SETS-Data/main/<file>.json
```

Files we depend on (initial set):

| File | Used by |
|---|---|
| `equipment.json` | `icon_matcher`, `layout_detector` (slot-type constraints) |
| `traits.json` | trait grid recognition |
| `boff_abilities.json` | BOFF marker / ability classification (`Page`, `name`, `I`, `II`, `III` keyed) |
| `ships.json` | ship roster (type-first disambiguation in importer) |

Additional files are added on demand; the loader does **not** hard-code
the full upstream file list — `fetch(name)` works for any path.

## Cache layout

```
$XDG_CONFIG_HOME/warp/cache/           # or ~/.config/warp/cache/
├── equipment.json
├── equipment.json.meta                # {etag, sha256, fetched_at}
├── boff_abilities.json
├── boff_abilities.json.meta
└── ...
```

`.meta` keeps the ETag (when available) and a fetched-at timestamp.

## Refresh policy

1. **First run:** no cache → fetch all required files synchronously,
   block the UI with a "downloading reference data…" splash.
2. **Subsequent runs:** load from cache immediately. In a background
   thread, issue `GET` with `If-None-Match: <etag>`:
   - `304 Not Modified` → keep cache, refresh `fetched_at`.
   - `200 OK` → write new file + meta, log change, post a signal so
     long-running consumers can reload.
3. **Manual refresh:** `sto-warp data refresh` CLI subcommand forces
   redownload, ignoring ETag.
4. **Forced TTL:** if `fetched_at` is older than 30 days, force refresh
   regardless of ETag (defensive — covers proxies that strip headers).

## Module shape (proposed)

`warp/data/cargo.py` (new module):

```python
def cache_dir() -> Path: ...
def fetch(name: str, *, force: bool = False) -> dict | list: ...
def load(name: str) -> dict | list: ...        # cache-first, fetches if absent
def refresh_all(names: Iterable[str]) -> None: ...
def loaded() -> dict[str, Any]:                # for inspectors
    return _MEMO
```

Single in-process memo (`_MEMO: dict[str, Any]`) so re-asking for the
same file is free.

Errors:

- Network failure on first fetch → raise `WarpDataUnavailable`, the GUI
  surfaces it with a "check your connection / try `sto-warp data
  refresh`" message.
- Network failure on background refresh → log warning, keep cache.

## What this replaces (SETS side)

The previous SETS-coupled call sites that pulled from
`~/.config/SETS/cache/` will, in the bridge package, adapt the SETS
loader output to the sto-warp `load(name)` shape — so sto-warp itself
never reaches into SETS' cache dir.

## Open questions

- Do we also mirror the upstream files on the sets-sto Hugging Face org
  so the loader can fall back when GitHub raw is rate-limited? (Decide
  before 1.0.)
- Do we ship a frozen snapshot inside the wheel as a last-resort
  offline fallback? (Cost: ~3 MB of JSON, but enables `pipx run` use
  with no network.)
