# Cargo data loader — design note

**Status:** plan only (not implemented yet).
**Strategy chosen:** (iii) fetch from `STOCD/SETS-Data` GitHub raw URLs at
first run, cache locally — confirmed 2026-05-18 after considering the
alternative of resurrecting `warp/data/item_db.json` via `warp/tools/
scraper.py`.

## Why not the existing `item_db.json` scraper

`warp/tools/scraper.py` builds a single consolidated `item_db.json` from
SETS cargo + vger.stobuilds.com + optional GitHub mirror. It works and
the file is already on disk in sets-warp — but recognition has never
read it (audit 2026-05-17: zero callers in `warp/recognition/`,
`warp/warp_importer.py`, `warp/warp_dialog.py`, `warp/trainer/`).

Rejected because:

- **Two sources of truth.** The scraper produces a derivative schema we
  must maintain forever as STOCD/SETS-Data evolves. With B we read
  upstream files as-is and move with the community.
- **Coarse refresh.** `item_db.json` is monolithic — refresh = rebuild
  everything via scraper run. B does per-file ETag refresh in the
  background, transparent to the user.
- **Unused enrichment.** Scraper's value-add (icon_url, wiki_url, vger
  metadata) is not consumed by recognition. Maintenance cost for data
  nobody reads.

The scraper stays in `warp/tools/` as a power-user tool for offline
enriched DB builds — secondary path, not the default.

## What we get from cargo vs HF

HF (`sets-sto/sto-icon-dataset`, `sets-sto/warp-knowledge`) is the
**perception layer**: trained model + label_map + pHash overrides.
Answers "what item is this crop?" → returns a name string.

Cargo is the **semantics layer**: metadata keyed by item name.
Recognition / importer needs four files for this:

| File | What recognition uses |
|---|---|
| `equipment.json` | `type` field → `SLOT_VALID_TYPES` constraint checking |
| `ship_list.json` | per-ship slot profile (BOFF seating, console counts) |
| `boff_abilities.json` | rank Roman numerals + profession mapping |
| `traits.json` | `environment` (space/ground) + `type` (personal/rep/…) |

Without cargo, recognition is semantically blind — it knows the name
but not what slot the item belongs to, what rank a BOFF ability is, or
whether a trait is space or ground. So HF and cargo are complementary,
not interchangeable.

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

## Offline fallback (decided 2026-05-18)

Ship a small frozen snapshot of the four required files inside the
wheel under `warp/data/baseline/`. Loader precedence:

1. `~/.config/warp/cache/<file>.json` if present.
2. Else: fetch from STOCD/SETS-Data raw, write to cache, use.
3. Else (no network on first run): copy `warp/data/baseline/<file>.json`
   to cache, log a warning that data is stale.

Snapshot is updated by maintainer via a make-snapshot script that
pulls current STOCD/SETS-Data and copies the four files into the
package source. Refresh cadence: per minor release.

## Open questions

- Mirror upstream files on the sets-sto HF org as a secondary endpoint
  when GitHub raw is rate-limited? (Defer until we see real 429s.)
