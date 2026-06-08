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
| 1 | `cargo`    | `raw.githubusercontent.com/STOCD/SETS-Data/main/cargo/`   | HTTP `ETag` + `If-None-Match` per file; 24 h skip-window on top         | 24 h               | `warp.data.cargo._refresh_loop`                                 |
| 2 | `assets`   | `raw.githubusercontent.com/STOCD/SETS-Data/main/images/`  | GitHub Tree API SHA1 diff against local `_git_sha1`; 1 h manifest cache | 1 h (tree cache)   | `warp.data.asset_sync.AssetSyncManager.run`                     |
| 3 | `knowledge`| sto-warp Space backend `/knowledge`                       | local-mtime TTL; full re-download on expiry; stale fallback on 5xx      | 24 h               | `warp.knowledge.sync_client.WARPSyncClient._download_knowledge_bg` |
| 4 | `model`    | sto-warp Space backend `/model/version`                   | remote `trained_at` ISO timestamp comparison; embedder self-heal        | 15 min (rate-limit)| `warp.trainer.model_updater.ModelUpdater._bg_check`             |
| 5 | `crops`    | HF dataset `sets-sto/sto-icon-dataset` (tarball)          | dataset commit SHA recorded in `crops_manifest.json`                    | per-launch         | `warp.knowledge.community_crops.CommunityCropsClient.fetch`     |
| 6 | `equiv`    | HF dataset resolve URL → `icon_equivalence.json`          | local-mtime TTL; full re-download on expiry; stale fallback on 5xx      | 24 h               | `WARPSyncClient._download_icon_equivalence_bg`                  |
| 7 | `seed`     | derived from (5)                                          | mtime guard on `data/annotations.jsonl` from `community_crops`          | per-launch         | `warp.recognition.icon_matcher.SETSIconMatcher.seed_from_community_crops` |

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
    self.step.emit('cargo');     cargo.refresh_all(force=self._force)
    self.step.emit('assets');    AssetSyncManager().run()
    self.step.emit('knowledge'); sync_client._download_knowledge_bg(force=…)
    self.step.emit('model');     ModelUpdater()._bg_check(on_updated=None)
    self.step.emit('community'); CommunityCropsClient().fetch()
    self.step.emit('equiv');     sync_client._download_icon_equivalence_bg(force=…)
    self.step.emit('seed');      SETSIconMatcher.seed_from_community_crops()
    self.step.emit('upload');    sync_manager.check_and_upload()
    self.step.emit('done')
```

Each step is wrapped in `try/except` so an upstream 5xx never aborts
the next step. Failures log at WARNING and the cycle proceeds; the
next 60 min tick retries naturally.

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
│   ├── github_tree_cache.json        ← asset-sync 1 h tree manifest
│   └── sync_failed.json              ← asset-sync 7 d failed-URL TTL
├── icons/                            ← item icons mirrored from STOCD/SETS-Data
├── ship_images/                      ← ship images mirrored from STOCD/SETS-Data
├── community_crops/                  ← HF crops tarball extracted here
│   ├── data/crops/<sha>.png
│   ├── data/annotations.jsonl
│   └── crops_manifest.json           ← dataset SHA pin for idempotent refresh
├── knowledge.json                    ← community pHash entries
├── icon_equivalence.json             ← admin-curated equivalence classes
└── warp_*.log                        ← per-channel logs
```

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
