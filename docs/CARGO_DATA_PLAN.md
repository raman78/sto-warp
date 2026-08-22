# Cargo data loader — design note

**Status:** **implemented** since sto-warp 1.0.0. Live as `warp/data/cargo.py`
fetching into `~/.config/warp/cache/` — originally from `STOCD/SETS-Data`,
since August 2026 from `raman78/warp-cargo-data` with SETS-Data as fallback,
driven by the splash + 60 min refresh cycle described in
[`SYNC_ARCHITECTURE.md`](SYNC_ARCHITECTURE.md). This document records the
*why* of that choice; the runtime details live in the sync doc.
**Strategy chosen:** (iii) fetch from `STOCD/SETS-Data` GitHub raw URLs at
first run, cache locally — confirmed 2026-05-18 after considering the
alternative of resurrecting `warp/data/item_db.json` via `warp/tools/
scraper.py`.

## Why not the existing `item_db.json` scraper

> The scraper described below was **deleted in August 2026**, once it was
> clear nothing had ever read its output and its wiki endpoints had gone
> behind Cloudflare. It is preserved in git history; this section stays as
> the record of why the decision went the other way.

`warp/tools/scraper.py` built a single consolidated `item_db.json` from
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

The scraper was kept in `warp/tools/` for a while as a power-user tool for
offline enriched DB builds, then removed: it shipped to every user in the
wheel, nothing called it, and its wiki endpoints stopped answering.

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

Public mirror maintained by the SETS community (as of August 2026 the
fallback rather than the primary — see the update under *Offline fallback*):

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
2. Else: fetch from the first upstream that answers, write to cache, use.
3. Else (no network on first run): copy `warp/data/baseline/<file>.json`
   to cache, log a warning that data is stale.

> **Update (August 2026).** Step 2 now walks `UPSTREAM_BASES`:
> `raman78/warp-cargo-data` first, `STOCD/SETS-Data` as fallback. The move
> followed a regression where valid items silently disappeared from the
> community mirror and there was no way to fix it on our own schedule. The
> two were verified interchangeable by building every cache from each source
> and diffing all 47 buckets. Each source's ETag is tracked separately
> (`meta['source']`), since replaying one server's ETag at another risks a
> 304 that strands the cache on the wrong mirror's bytes.
>
> **What this does not buy (recorded 2026-08-11).** The mirror is refreshed
> by a browser-backed fetcher the maintainer runs manually — it needs a
> desktop session, steals focus while it works, and does not run when the
> machine is off. So it is a stop-gap for stowiki being unreachable behind
> Cloudflare, *not* the always-available cargo endpoint the availability
> problem actually calls for. It was announced to users in the 1.0.26
> release notes as a mirror "that refreshes itself every 8 hours"; that
> claim was unsupported and has been withdrawn from CHANGELOG, README and
> the published release. Do not re-announce the mirror as a user-facing
> feature until its refresh runs without the maintainer's workstation.

> **Field types differ per source (measured 2026-08-21).** "Interchangeable"
> holds for *which* records each mirror carries, not for how they are typed.
> `STOCD/SETS-Data` and the bundled baseline serve `ship_list.json` typed —
> `boffs`, `type` and `abilities` as lists, slot counts as numbers.
> `raman78/warp-cargo-data` serves the raw `Special:CargoExport` output,
> where every field is a string and list columns arrive comma-joined
> (794/797 ships). The other four files carry no type conflicts; the mirror
> only adds fields SETS-Data lacks.
>
> `cargo._normalise_ship` coerces records into the typed shape, and both
> readers of the file apply it: `cargo._build_ships` and
> `warp_importer.ShipDB._load` (which parses the cache file directly). It
> is idempotent, so it runs safely against either source. Without it,
> iterating `ship['boffs']` yields characters —
> `_boff_profile_from_shipdb` returned an empty BOFF slot profile for
> every ship on any install whose cargo came from the mirror, which is the
> default. Blank entries are dropped as well — six baseline ships carry an
> empty seat string, which parses into a phantom Commander seat with no
> profession; SETS drops them the same way.
>
> Verified by diffing all 797 normalised mirror records against SETS-Data.
> What remains is upstream data, not typing: five `image` values and one
> `type` string where the two mirrors hold different wiki text, HTML
> entities that SETS-Data leaves encoded (`&quot;`), and one ship
> (`Tamarian Deep Space Cruiser`) whose ability names contain literal
> commas. SETS-Data keeps those commas escaped as `&#44;` so its own split
> is unambiguous; our mirror decodes them before joining, so the split
> shatters five ability names into fragments. Only `'Innovation Effects'
> in abilities` reads this field (`warp_importer:694`), so the effect is
> cosmetic today — but the fix belongs in the mirror's exporter, which
> should keep `&#44;` encoded or emit arrays.

Snapshot is updated by the maintainer via `warp/tools/make_baseline.py`,
which walks the same `UPSTREAM_BASES` chain as the runtime loader. Refresh
cadence: per minor release — but **not** while upstream is known to be
missing data. A snapshot is the one copy a user without network ever sees,
so it refuses to write a file that came back materially smaller than the
committed one (`--allow-shrink` overrides).

## Items no cargo table holds (added 2026-08-22)

Elite Fleet, Colony Security and Fleet Station K-13 ground weapons appear on
the `Fleet Ground Weapons` wiki page and in **no** cargo table — the page
renders them from wikitext and stores nothing, and the individual weapons
have no item pages. Both mirrors are equally blind to them, so a name read
off a ground build validated against nothing and was dropped.

`warp-cargo-bay` harvests them from that page's own `{{item|…}}` calls and
publishes 136 rows to `scraped/scraped_ground_weapons.json`, beside the
mirror rather than inside it. On this side:

| Aspect | Behaviour |
|---|---|
| Source | `cargo.OVERLAY_BASE` — our mirror only; SETS-Data has no such path |
| Optional | a missing overlay logs a warning and yields `[]`; those names stay unknown, as before it existed |
| Precedence | `cargo._merge_overlay` skips any name a real cargo row already holds |
| Provenance | every row carries `source: listing-scrape` |
| Offline | shipped in `warp/data/baseline/` so a first run without network has them |

What it does for icons depends on the family, measured 2026-08-22 against the
wiki's own files:

| Family | Dedicated icon on the wiki | Consequence |
|---|---|---|
| Advanced Fleet, Elite Fleet | none | reuses the base weapon's picture, so icon matching cannot tell the variant from the plain weapon; the overlay buys name validation only |
| Elite Fleet Colony Security | yes, 49x64 | the picture is genuinely different — 58.5% of pixels differ from the base weapon by more than 8/255 — so matching *could* identify them, but only once those icons are in the database |

Neither `STOCD/SETS-Data` nor our icon database carries the Colony Security
pictures today, so those 48 weapons currently resolve by name and mismatch by
icon.

The overlay is meant to disappear. Each publisher run drops rows whose name
has turned up in `equipment.json`; one went that way on the first run. At
zero rows the whole mechanism can be retired.

## Open questions

- Mirror upstream files on the sets-sto HF org as a secondary endpoint
  when GitHub raw is rate-limited? (Defer until we see real 429s.)
