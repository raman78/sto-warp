# Startup-sync and refresh architecture

Technical reference for the data-refresh subsystem: who fetches what,
when, with what freshness guarantees, and where the marker / TTL state
lives. Companion to the user-facing "First-run setup splash" section
in `WARP_GUIDE.md`.

---

## 1. The seven data sources

sto-warp pulls reference data from four upstream origins. Every source
goes through the same dispatch path (`SyncCoordinator`), but each one
has its own freshness mechanism — picked to match the cost of a stale
local copy vs. the cost of a wasted HTTP fetch.

| # | Phase id   | Source                                                    | Freshness mechanism                                                     | TTL                | Implementation                                                  |
|---|------------|-----------------------------------------------------------|-------------------------------------------------------------------------|--------------------|-----------------------------------------------------------------|
| 1 | `cargo`    | `raw.githubusercontent.com/raman78/warp-cargo-data/main/cargo/`, falling back to `STOCD/SETS-Data/main/cargo/` | HTTP `ETag` + `If-None-Match` per file *per source*; 24 h skip-window on top | 24 h               | `warp.data.cargo._refresh_loop`                                 |
| 2 | `assets`   | `raw.githubusercontent.com/STOCD/SETS-Data/main/images/` **and** `raman78/warp-cargo-data/main/scraped/icons/` | GitHub Tree API SHA1 diff against local `_git_sha1`; one manifest cache per source | 1 h (tree cache)   | `warp.data.asset_sync.AssetSyncManager.run`                     |
| 3 | `knowledge`| sto-warp Space backend `/knowledge`                       | local-mtime TTL; full re-download on expiry; stale fallback on 5xx      | 24 h               | `warp.knowledge.sync_client.WARPSyncClient._download_knowledge_bg` |
| 4 | `model`    | sto-warp Space backend `/model/version`                   | remote `trained_at` + `embedder_trained_at` ISO comparison (independent); embedder self-heal | 15 min (rate-limit)| `warp.trainer.model_updater.ModelUpdater._bg_check`             |
| 5 | `crops`    | HF dataset `sets-sto/sto-icon-dataset` (tarball)          | dataset commit SHA recorded in `crops_manifest.json`                    | per-launch         | `warp.knowledge.community_crops.CommunityCropsClient.fetch`     |
| 6 | `equiv`    | HF dataset resolve URL → `icon_equivalence.json`          | local-mtime TTL; full re-download on expiry; stale fallback on 5xx      | 24 h               | `WARPSyncClient._download_icon_equivalence_bg`                  |
| 7 | `seed`     | derived from (5)                                          | mtime guard on `data/annotations.jsonl` from `community_crops`          | per-launch         | `warp.recognition.icon_matcher.SETSIconMatcher.seed_from_community_crops` |

#### Why `assets` has two sources

SETS-Data's `images/` is the only place item pictures ever came from, and
it is incomplete: measured 2026-08-22, 292 slottable item names had no
picture there and 71 of those exist on the wiki — `Jackal Mastiff` since
2019. An item with no picture cannot be matched from a screenshot and
shows an empty tooltip, however well its name resolves in cargo.

The naming is not the problem: `File:<item> icon.png` is what both sides
expect. The fetch is. stowiki answers 403 to plain HTTP clients behind its
Cloudflare challenge, so nothing without a browser reaches those files.
`warp-cargo-bay` harvests them through its browser session and republishes
them at `scraped/icons/`, named `quote_plus(<item name>).png` — exactly
what this sync writes into `icons/`.

`OVERLAY_GROUPS` (`warp/data/asset_sync.py`) adds that path as a second
source. Three properties matter:

- **Separate manifest cache** (`overlay_tree_cache.json`) — one source
  going quiet cannot invalidate the other's tree.
- **Additive and optional** — an unreachable overlay skips the group and
  leaves the run green. Those pictures stay missing, which is the state
  that predates the second source; nothing else changes.
- **Same target directory** — `_local_path` keys on the entry's own
  filename, so `scraped/icons/X.png` and `images/X.png` both land in
  `icons/X.png`. A picture SETS-Data later publishes simply overwrites the
  harvested one on the next SHA1 diff.

The names those pictures belong to are a separate concern — see
`docs/CARGO_DATA_PLAN.md` § *Items no cargo table holds*.

#### One item, two pictures: era-variant art

STO draws some gear differently in 23rd-century content, and the wiki files
that second picture as its own page — `File:Impulse Engines (23c) icon.png`
beside `File:Impulse Engines icon.png`. The **item** is unchanged: one name,
one cargo row, and the article renders both pictures side by side.

That breaks an assumption the icon index rests on. `SETSIconMatcher._build_index`
keys every entry on the filename (`name = unquote_plus(png.stem)`), so variant
art would enter the index under `Impulse Engines (23c)` — a name no cargo row
carries, which every candidate filter downstream then drops. Measured on
`Screenshot_2026-01-19_145418.png` (a Kelvin-timeline ship, so 23c art) before
the fold existed: the impulse slot came back as `Advanced Fleet Impulse
Engines` @0.48 restricted to engine candidates, and `Shield Array` @0.50
unrestricted. With the variant present and folded: `Impulse Engines` @0.72 in
both.

`_base_item_name` (`warp/recognition/icon_matcher.py`) folds a variant onto the
item it depicts, and asks cargo rather than reading the tag:

| Icon name | Base in cargo? | Name in cargo? | Indexed as |
|---|---|---|---|
| `Impulse Engines (23c)` | yes | no | `Impulse Engines` |
| `Modified Phaser Pistol (23c.)` | no | **yes** | unchanged |
| `Matter Anti-Matter Warp Core (23c)` | no | no | unchanged |
| *(cargo unavailable)* | — | — | unchanged |

The tag cannot be the deciding factor because it is not always a variant
marker: `Modified Phaser Pistol (23c.)` is a whole item name, tag included.
Measured 2026-08-22 against the live wiki and the cargo cache — 35 `(23c)`
files, none of them an item name, 34 with an item under the base name; one
`(23c.)` file that *is* an item name. Deciding from cargo also means the rule
needs no maintenance: an item the tables only start carrying later begins
folding on its own, and a tagged name that becomes a real item keeps working.

Two index entries then answer to the same item name, one per era. The index is
a list scanned for the best score, so both compete and the picture the
screenshot actually shows wins.

The publishing half mirrors this. `fetcher/icons.py::era_variants` in
`warp-cargo-bay` asks the wiki which variant files exist (`intitle:"23c"
intitle:"icon"`) on every run rather than working from a list, applies the
same cargo test, and publishes what is left to `scraped/icons/` under
`quote_plus('<item> (23c)').png`. 34 files as of 2026-08-22.

There's an eighth pseudo-phase, `upload`, run only by `SyncCoordinator`
(not by the splash) — it pushes pending confirmed crops back up to HF
when the user has been correcting in WARP CORE.

### TTL semantics

**Important:** the TTLs are *skip-windows inside the refresh
implementation*, not standalone schedulers. They only apply when
something else calls the refresh function — they do not cause
spontaneous network traffic. The full freshness story is therefore
"`<dispatcher> calls refresh; refresh checks TTL; if expired, refresh
hits the network`". The dispatcher is what makes the whole chain
work, and there are exactly two dispatchers:

1. **The startup-sync splash** (cold start only, see §3).
2. **`SyncCoordinator`** (every launch + every 60 minutes thereafter).

If a phase isn't wired into one of those two paths it doesn't run.
The cargo-staleness bug fixed in commit `6ec7e7e` was exactly that
class of bug: `cargo.refresh_all` was implemented correctly but had
zero callers outside a dev-only CLI, so the 24 h TTL was a dead
guarantee — files stayed at their install-time revision for weeks.

---

## 2. Dispatcher topology

```
                                ┌──────────────────────┐
                                │ QApplication.exec()  │
                                └──────────┬───────────┘
                                           │
            ┌──────────────────────────────┼──────────────────────────────┐
            │                              │                              │
            ▼                              ▼                              ▼
   ┌────────────────┐            ┌──────────────────┐           ┌──────────────────┐
   │ marker absent? │  yes →     │  ColdStartDialog │   then →  │  LauncherWindow  │
   │  (first run /  │ ────────►  │  blocking modal  │           │  (launcher tab)  │
   │   interrupted) │            │  worker = QThread│           │                  │
   └────────┬───────┘            │  7 phases serial │           │  init_sync() →   │
            │ no                 └────────┬─────────┘           │  QTimer 500 ms ─┐│
            │                             │                     └──────────────┬─┘│
            │                             ▼                                    │  │
            │                  all_done → write marker                         │  │
            │                                                                  │  │
            └─────────────────────────────────────────────────────────────────►│  │
                                                                               │  │
                                                                               ▼  ▼
                                                          ┌────────────────────────────────┐
                                                          │      SyncCoordinator           │
                                                          │  ─ start()  OR                 │
                                                          │  ─ arm_periodic_only() if      │
                                                          │     the splash already ran     │
                                                          │                                │
                                                          │  cycle: cargo → assets →       │
                                                          │  knowledge → model →           │
                                                          │  community → equiv →           │
                                                          │  seed → upload → done          │
                                                          │                                │
                                                          │  QTimer 60 min → repeat        │
                                                          └────────────────────────────────┘
```

Notes on the diagram:

- The splash and `SyncCoordinator` share the **same underlying refresh
  functions**, just on different threads. The splash drives them in the
  foreground with progress signals; `SyncCoordinator` drives them in a
  background `QThread` with status-bar text only.
- `arm_periodic_only()` exists so that after a clean splash run we
  don't *immediately* re-walk the cycle we just finished. The 60 min
  timer is still armed; the next walk happens on schedule.
- The marker file `~/.config/warp/startup_sync_done` exists purely to
  gate the splash. It carries no version info — only its existence
  matters. Delete the file to force the splash on the next launch.

---

## 3. Splash lifecycle (cold start only)

### Detection

```python
# warp/gui/cold_start_dialog.py
def is_cold_start() -> bool:
    return not (config_dir() / 'startup_sync_done').exists()
```

Three observations about this detector:

1. **Mirror-population heuristics intentionally not used.** An earlier
   version checked "crops dir empty AND icons dir empty". That broke
   for the partial-download case: 200 of 8 000 crops on disk was
   enough to look "populated" and skip the splash, while 7 800
   downloads then continued silently in the background.
2. **Marker is written only by `_on_all_done`**, after every phase
   has run to completion. Cancel / Close / kill -9 all leave it
   absent.
3. **No version field.** Subsequent feature additions that need a
   re-prompt should use a *different* marker file with its own
   migration story rather than overloading this one.

### Phase ordering

```
cargo → assets → knowledge → model → crops → seed → equiv
```

Order rationale:

- `cargo` first because it's tiny, fast, and CARGO drives label
  resolution downstream — if it fails everything else still works on
  baseline JSONs.
- `assets` early because it's the long one; running it second means a
  user who cancels after a few minutes already has the most
  expensive download out of the way.
- `seed` must come *after* `crops` (it walks the freshly downloaded
  community crop library).
- `equiv` last among the small ones because it's the most optional
  (the file may not even exist yet — admin-curated, opt-in).

### Failure handling

Phase failures are isolated by `try/except` around each `phase_fn`
call inside `_ColdStartWorker.run`. A `knowledge` HTTP 503 must not
block `crops`. The dialog row turns into a `✗ short error` indicator
and the loop continues. The marker is still written if every phase
*ran* — even with some failures — because the user has paid the
attention cost and we don't want to re-prompt on every launch for a
single phase that the server is throwing 503s on. The phase will be
retried on the next `SyncCoordinator` tick.

### Two exit paths

| Button | `closed_via_quit` | `completed_cleanly` | What happens                              |
|--------|-------------------|---------------------|-------------------------------------------|
| `_on_all_done` (auto)  | False     | True                | `accept()`, marker written, launcher opens|
| Close (exit)            | True      | False               | `reject()`, `QApplication.quit()`         |
| Cancel (degraded)       | False     | False               | `reject()`, launcher opens without marker |

`maybe_run_cold_start()` returns `(should_launch, skip_initial_sync)`:

- `(True, True)`   — splash ran every phase, launcher should arm only the periodic timer
- `(True, False)`  — warm start OR cancelled splash, launcher should run a full sync cycle
- `(False, False)` — user quit; main returns 0

---

## 4. `SyncCoordinator` cycle

`warp/gui/sync_coordinator.py`. One `QObject` per launcher window,
owns a single `_RefreshWorker` instance at a time (mutex on
`request_refresh`).

### Timeline of a launch

```
   t = 0          QApplication starts
   t ≈ 200 ms     LauncherWindow.show() returns
   t ≈ 700 ms     QTimer.singleShot(500ms) fires:
                    ├─ first refresh cycle starts on a QThread
                    └─ periodic QTimer (60 min) armed in parallel
   t ≈ 2-5  s     cycle finishes if everything was cached fresh
   t ≈ 60 min     periodic timer fires → cycle runs again
   …              repeated until the launcher window closes
```

If the splash ran to completion immediately before this, the 500 ms
tick calls `arm_periodic_only()` instead of `start()` so the initial
cycle is skipped — the periodic 60 min timer is still armed.

### `_RefreshWorker.run` step-by-step

```python
# warp/gui/sync_coordinator.py — abbreviated
def run(self):
    self.step.emit('upload');    sync_manager.check_and_upload()
    self.step.emit('cargo');     cargo.refresh_all(force=self._force)
    self.step.emit('assets');    AssetSyncManager().run()
    self.step.emit('knowledge'); sync_client._download_knowledge_bg(force=…)
    self.step.emit('model');     ModelUpdater()._bg_check(on_updated=None)
    self.step.emit('community'); CommunityCropsClient().fetch()
    self.step.emit('equiv');     sync_client._download_icon_equivalence_bg(force=…)
    self.step.emit('seed');      SETSIconMatcher.seed_from_community_crops()
    #                            then wait, bounded, on the upload worker
    self.step.emit('done')
```

Each step is wrapped in `try/except` so an upstream 5xx never aborts
the next step. Failures log at WARNING and the cycle proceeds; the
next 60 min tick retries naturally.

### The daily request budget

Everything that POSTs shares one budget, because the server counts them
together. `sets-warp-backend` admits `MAX_REQ_PER_INSTALL` **requests** per
UTC day — 500 by default — and keeps a second bucket of the same size keyed
on the client's IP. It does not count items, and it counts a request whether
or not it accepted what was in it.

Every client counter predating this measured something else. The trainer
counted crops it had queued (`MAX_DAILY_UPLOADS`, 1000); the knowledge client
counted contributions the server had accepted. Neither moves on a day of
refusals, so neither could stop one. `warp.backend_budget.DailyBudget` counts
requests, is shared by `warp.trainer.sync` and
`warp.knowledge.sync_client`, and lives in `~/.config/warp/backend_budget.json`
so a restart does not hand the client a budget the server disagrees with.

It stops on two independent signals, and both are needed:

- **Our own count**, against `MAX_DAILY_REQUESTS` (480, deliberately under
  the server's 500). This is a *prediction* of the per-install bucket, and it
  is what keeps an ordinary day from ever reaching a refusal.
- **A 429 we were actually given.** The prediction cannot stand alone: the
  per-IP bucket is shared with everyone behind the same address and is
  invisible from here, the buckets live in the server process so a restart
  clears them, and the cap is an environment variable that can change under
  us. A refusal is ground truth and is honoured until the UTC day turns.

The server gets the last word on its own state. Those buckets are a dict in
the backend process, so a Space restart — a deploy, or waking from idle —
clears them, and a client holding a refusal would otherwise sit out the rest
of the UTC day against a server that has already forgotten. Once per run, and
only while a block is in force, `DailyBudget.reconsider` reads `GET /quota`:
if the backend reports room in both buckets the block is lifted, and the
request count is taken from the server, which is the number the cap is
actually applied to. `/quota` is a read and is not rate limited, so asking is
free. Any failure leaves the block alone — an unreachable backend is not
evidence that it would accept anything. Verified 2026-09-06: immediately after
a deploy, `/quota` reported 0 of 500 in both buckets on an install that had
been refused all afternoon.

That same reading settled a question the backend's code had left open. It
resolves the caller from the **rightmost** `X-Forwarded-For` entry, which
identifies a client only behind exactly one trusted proxy — true of the Render
deployment it was written for, unverified since production moved to an HF
Space. Had there been a second hop, every client would have resolved to one
infrastructure address and the per-IP cap would have been a global 500/day for
the whole community. It is not: the Space forwards a single entry and it is
the caller's own public address.

A 429 therefore ends the whole upload run — `BackendBudgetExhausted`, which
every channel re-raises — rather than one channel. Before that, a refusal was
a per-channel warning and the loop carried on: about fifteen POSTs per cycle
learning the same answer, one per screen-type directory plus crops and
anchors, each counted against the very budget that was missing. The knowledge
client was worse: a 429 was treated as a transient outage, so each queued
contribution was retried three times, then again after a five-minute backoff
— on its own roughly 864 refused requests a day against a cap of 500. Between
them, clearing the backlog was what kept the door shut: measured 2026-09-06,
127 corrected screen types had been stuck at "not yet shared" for days while
the install sat at 638 of its own 1000-item allowance.

**`upload` goes first, and that ordering is deliberate.** It was last until
2026-09-06, which made the only step that *sends* the user's work the first
thing lost whenever a cycle was cut short — and it waited behind the two
slowest phases, `community` and `seed` (the seed walks some 12 000 approved
crops). A backlog of 129 corrected screen types sat untouched on the
maintainer's install while every download around it kept succeeding. Upload
depends on no other phase: it reads the local training store and POSTs to the
backend. The bounded wait on its worker stays at the end of the cycle so the
status bar does not report `done` while bytes are still going out.

### Interruption

`SyncCoordinator.stop()` (called from the launcher's `closeEvent`)
asks the worker thread to bail at the next step boundary, then waits
at most 200 ms. Anything still inside an HTTP call at that moment is
left to be reaped on interpreter exit — the alternative is a
noticeable UI freeze on close while an upload finishes.

---

## 5. Cache layout

```
~/.config/warp/
├── startup_sync_done                 ← splash completion marker
├── install_id.txt                    ← anonymous client id for HF
├── cache/                            ← cargo JSONs
│   ├── equipment.json + .meta        (ETag + fetched_at)
│   ├── traits.json + .meta
│   ├── starship_traits.json + .meta
│   ├── boff_abilities.json + .meta
│   ├── ship_list.json + .meta
│   ├── scraped_ground_weapons.json   ← items no cargo table holds
│   ├── github_tree_cache.json        ← asset-sync 1 h tree manifest
│   ├── overlay_tree_cache.json       ← same, for the harvested-icon source
│   └── sync_failed.json              ← asset-sync 7 d failed-URL TTL
├── icons/                            ← item icons: STOCD/SETS-Data, plus the
│                                       ones only the wiki has
├── ship_images/                      ← ship images mirrored from STOCD/SETS-Data
├── community_crops/                  ← HF crops tarball extracted here
│   ├── data/crops/<ab>/<sha>.png     ← sharded by the first two characters
│   ├── data/annotations.jsonl
│   └── crops_manifest.json           ← dataset SHA pin for idempotent refresh
├── backend_budget.json               ← requests sent today + any 429 received
├── knowledge.json                    ← community pHash entries
├── icon_equivalence.json             ← admin-curated equivalence classes
└── warp_*.log                        ← per-channel logs
```

### Why the crop mirror is sharded

`data/crops/` upstream reached the limit HF enforces — 10 000 files per
directory — and the promotion froze there on 2026-07-16 with an HTTP 400 the
server would not explain until the message was read in full. Crops are now
written under the first two characters of their sha, which is their content
hash, so the path is derivable on both sides with no index to keep in step.

The mirror follows the same rule for its own reasons: it held 12 274 files
after one merge and grows weekly, every sync globs it several times, and the
client also ships on Windows. `CommunityCropsClient._shard_local` runs on
every sync, is idempotent, and migrates an older install in place by rename —
no re-download. Nothing assumes it has already run: the counters, the cleanup
guard, the k-NN seeder and the backend's mirror lookup all accept either
layout.

An install that has not updated is unharmed. Its sync finds no crops under
the flat path upstream, and `_soft_delete`'s cleanup guard refuses to empty a
mirror over a removal that large, so it keeps everything it has and simply
stops receiving new crops until it updates.

The `.meta` sidecars next to cargo JSONs hold `{etag, fetched_at}` so
the next refresh can send `If-None-Match` and so the 24 h skip-window
has a reference timestamp. Wiping the JSON without its `.meta` is
safe — the next refresh treats the file as missing and re-downloads.

---

## 6. Log signal

Each cycle emits a clear "did verification actually happen?" trail.
Per-file lines now log at INFO (raised from DEBUG in commit `31308b2`
specifically so this is visible without enabling debug logging):

```
SyncCoordinator: cycle start (force=False)
SyncCoordinator: step=cargo — equipment/trait/ship JSONs
cargo.refresh: equipment.json fresh (3h old, TTL 24h) — skipped
cargo.refresh: traits.json unchanged (HTTP 304)
cargo.refresh: ships.json updated (793620 B)
SyncCoordinator: step=assets — GitHub icon/ship asset mirror
AssetSync: tree cache hit (12min old, 12450 files)
AssetSync [Item Icons]: 0/9821 need download
AssetSync [Ship Images]: 0/2629 need download
AssetSync: complete — {'checked': 12450, 'updated': 0, 'failed': 0}
SyncCoordinator: step=knowledge — community pHash download
WARPSync: knowledge fresh (3.2h old, TTL 24h) — reused 4521 entries from cache
SyncCoordinator: step=model — central model version check
ModelUpdater: local trained_at=2026-05-30T... remote=2026-05-30T... — up to date
SyncCoordinator: step=community — approved crops + labels mirror
CommunityCrops: tarball already current at d8f4c1b3 — nothing to do
SyncCoordinator: step=equiv — admin-curated icon equivalence
WARPSync: icon-equivalence fresh (3.2h old, TTL 24h) — reused 42 classes from cache
SyncCoordinator: step=seed — icon matcher community seed
SETSIconMatcher: seed mtime unchanged — skipped
SyncCoordinator: step=upload — confirmed-crop HuggingFace upload
SyncCoordinator: cycle done
```

Three states are visible per phase: `fresh — skipped` (TTL window
guard), `unchanged (304)` (server confirmed identical), `updated`
(new bytes written). Anything else logs at WARNING with the failure
reason.

---

## 7. The cargo-staleness incident (postmortem reference)

The bug class to look out for: a freshness mechanism that *would*
work, sitting behind a function nobody calls.

- **Symptom:** user observed `cargo/*.json` cache files with mtimes
  weeks old despite running sto-warp daily.
- **Root cause:** `cargo.refresh_all` and `cargo.refresh_async` had
  exactly one caller in the whole codebase — the cold-start splash —
  and the splash only ran on first install. Subsequent launches read
  the cached JSONs straight from disk and never re-verified.
- **Fix:** added `cargo` and `equiv` as steps in
  `_RefreshWorker.run`. The 24 h TTL inside the refresh function
  prevents wasted bandwidth; the dispatcher ensures the refresh is
  actually called.
- **Generalised lesson:** when auditing whether a refresh path
  works, *grep for callers*. "The function is implemented correctly"
  and "the function is wired into a runtime path that fires" are
  independent properties. The latter is the harder one to verify.

The chore companion commit `31308b2` bumped per-file verification
logs from DEBUG to INFO precisely so a future "is this actually
running?" question can be answered by reading the log instead of
re-running this investigation.
