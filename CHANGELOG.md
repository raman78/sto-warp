# Changelog

All notable changes to **sto-warp** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.4] — 2026-05-24

Ground BOFF recognition + local bootstrap embedder workstream. Closes the
embedder's ground BOFF class-coverage gap (was 6/106) and ships a one-off
local-training path so future gaps can be patched without the 53-day HF
staging round-trip.

### Added
- `warp/trainer/synthetic_crop_generator.py` — generates augmented 64×64
  BGR crops from cargo wiki PNGs for embedder bootstrap. Per-class
  background synthesis (dark gradient + noise), alpha-aware composite,
  bbox / scale / colour jitter, optional radial cooldown overlay, JPEG
  re-encode. CLI: `python -m warp.trainer.synthetic_crop_generator
  --env ground -n 100`.
- `warp/trainer/embedder_trainer.py` — local ArcFace + EfficientNet-B0
  + PK sampler + k-NN gallery trainer (mirrors central
  `admin_train_metric.py`). Loads real crops via `crops/crop_index.json`
  and synthetic crops via cargo slug reverse-map; warm-starts from the
  existing `icon_embedder.pt` to preserve previously-learned classes.
  Outputs `icon_embedder.pt` + `embedding_index.npz` +
  `embedder_label_map.json` + `icon_embedder_meta.json` to
  `userdata.models_dir()`. CLI:
  `python -m warp.trainer.embedder_trainer --generate-synthetic
  --env ground -n 100 --train`.
- Ground BOFF seat code `G` parsed by `recognition/boff_keys.py`
  (regex + `is_ground_seat()` helper); `boff_marker.py` labels marker
  panels accordingly.
- `recognition/warp_importer._build_candidate_pools` routes ground
  seats to a ground-only ability pool (cargo `boff_abilities['ground']`),
  preventing space abilities from being suggested for ground BOFFs.
- `docs/ML_PIPELINE.md` §9 — full documentation of the one-shot local
  bootstrap workflow, hyper-parameters, data sources, output files, and
  the manual upload step to `sets-sto/warp-knowledge/models/`.

### Fixed
- Embedder trainer no longer filters out crops named `__inactive__` /
  `__empty__`; the original filter on `name.startswith('__')` silently
  dropped 433 + 113 real training crops, which made the embedder
  nearest-neighbour-snap empty slots to a random ability (typically
  Charged Particle Burst at conf≈0.93). System labels are now learned
  as proper classes so blank/greyed-out slots are recognised as empty.

### Trained model
Local bootstrap produced **956-class embedder** (was 524), val_recall@1
0.965, gallery 27 127 vectors. Uploaded to
`sets-sto/warp-knowledge/models/` (HF commit `1304198cc8f1a8e0`); other
installs pull on their next `ModelUpdater` tick. Coverage: ground BOFF
106/106 (was 6/106), space BOFF 122/122 (was 64/122).

---

## [1.0.3] — 2026-05-21

Targeted bug fixes on top of 1.0.2 — review-panel data integrity in
the trainer plus a new `Clear All BBoxes` action.

### Added
- `Clear All BBoxes` button alongside `+ Add BBox` / `- Remove BBox`
  in WARP CORE. Confirmation dialog offers three paths: **Yes**
  (remove everything, confirmed bboxes are also wiped from
  `annotations.json` via `_data_mgr.remove_annotation` + `save()`),
  **Spare All Confirmed** (only available when both pending and
  confirmed bboxes exist on the image), or **Cancel** (default).
  Logs the outcome to `warp_detection_core.log`.

### Fixed
- WARP CORE: clicking a Ship Tier (or Ship Type) row in the review
  list now actually populates the matching combo in the **Annotate
  Selected Icon** panel. `_on_review_row_changed` only wrote into
  `_name_edit`, which is hidden for NON_ICON_SLOTS — the tier/type
  combos stayed on whatever value they had last (typically T1 or the
  first ship in the dropdown), so the panel disagreed with the
  highlighted row. Now the NON_ICON_SLOT branch routes the row's
  `name` into `_tier_combo` / `_ship_type_combo` directly.
- WARP CORE: stale confirmed annotation no longer silently shadows a
  freshly-detected value for the same bbox+slot. `_populate_review_panel`
  used `ann_id = hash(bbox + slot)` (no `name`), so re-running
  Auto-Detect Slots and getting `Ship Tier`=T6-X2 was overwritten by
  an older confirmed T1 on disk. Now, when the fresh name disagrees
  with the saved one, the fresh value wins and the row is demoted to
  `pending` so the user re-confirms — with an `info` log line.
- WARP CORE: confirming Ship Type / Ship Tier with an empty combo no
  longer silently blanks the row. `_on_accept` falls back to the
  row's `name` / `orig_name` when the editor widget is empty,
  preventing the "after Confirm the Ship Type disappears from the
  bbox" footgun (OCR timing or an empty `_on_item_selected` payload
  used to cause this).

## [1.0.2] — 2026-05-21

Progress-bar unification across WARP and WARP CORE, plus a small
responsiveness fix so the X button closes the app immediately even
while a background sync or detection is mid-flight.

### Added
- Shared `StatusProgressBar` widget (`warp/gui/progress_bar.py`):
  status-bar progress bar with an embedded `Cancel` button. WARP and
  WARP CORE now use the same component, with the bar sized to its
  natural width and Cancel pinned to the window's right corner.
- WARP CORE: per-stage `%` progress during Auto-Detect Slots. The
  trainer's `RecognitionWorker` now forwards the importer's
  `progress_callback` and emits a `progress(int, str)` signal, so the
  status bar shows the same `image.png: Fore Weapon 1/4`-style
  breakdown as WARP. Cancel button stops the run at the next
  checkpoint (cooperative via `QThread.requestInterruption()`).

### Changed
- WARP CORE: the modal "Detecting Screen Types" and "Recognising
  Icons" popups are gone. Both flows now report progress in the main
  status bar, matching WARP. Cancel routes through
  `_cancel_active_run` so a single button covers whichever worker is
  live.
- WARP cancellation reworked to be cooperative — the importer's
  progress callback raises `InterruptedError` when the user clicks
  Cancel, which `RecognitionWorker.run()` translates into
  `failed('Cancelled')`. No more relying on `QThread.terminate()`.

### Fixed
- Close button no longer feels unresponsive while a sync cycle or
  detection is running. `SyncCoordinator.stop()` now caps its wait at
  200 ms (was 5 s) and asks the refresh worker to bail at the next
  step boundary via `isInterruptionRequested()`. The inner HF upload
  worker also got a bounded `wait(2000)` so a hung upload can't block
  app shutdown. WARP CORE's `closeEvent` likewise stopped per-worker
  `wait(500/2000/3000)` calls — one 200 ms grace window covers them
  all.

## [1.0.1] — 2026-05-21

UI polish, log-channel isolation, and small bug fixes on top of 1.0.0.
No data or pipeline changes — drop-in upgrade.

### Added
- Centralised theme registry (`warp/themes.py`) — chrome and semantic
  colours live in one `Theme` dataclass; `warp.style` reads from
  `themes.get_active()`. Swap themes by editing one file (or setting
  `WARP_THEME=<name>` before launch).
- Detection Logs tab now lives inside each tool's own window — WARP
  shows its run, WARP CORE shows its run. Backed by a thread-local
  channel router in `warp.debug` so worker threads in either tool
  write to their own file (`warp_detection.log` /
  `warp_detection_core.log`) without bleeding into the other view.
- `Copy All` and `Save As…` buttons on the Detection Logs tab. The
  default save name comes from the active screenshot
  (`{image}_{space|ground}_{YYYYMMDD-HHMMSS}.log`).
- Anchorless-rescue OCR matches now propagate their bbox through to
  the Preview tab, so the token that drove a rescue is visible.
- Opt-in wheel-event probe (`WARP_WHEEL_PROBE=1`) that traces which
  widget actually receives `QWheelEvent` inside the log view —
  diagnostic aid for the intermittent "dead strip" scroll issue.

### Changed
- WARP CORE: Screenshot and Detection Logs are top-level sibling
  tabs over the recognition workspace (previously Detection Logs sat
  under the canvas, which was inconsistent with WARP).
- Toolbar / button styling unified between WARP and WARP CORE —
  `QToolButton` and `QPushButton` share one QSS rule, default border
  is neutral grey (the previous gold border felt loud), hover lifts
  to the accent colour.
- `Force build type` combo is now visibly greyed when the checkbox
  is off (was technically disabled but indistinguishable).
- Selected `QTabBar` tab gets a muted slate-blue background so the
  active tab is obvious without shouting.
- Results tree: alternating-row colours off, slot/parent rows shown
  in a lighter background with bold font, single-entry slots (Ship
  Name / Type / Tier, or any slot that matched exactly one item)
  surface the value on the parent row so it's visible even collapsed.
- Per-bbox Preview labels show only the confidence — slot names move
  to a single row-level label per group (less visual noise).

### Fixed
- Alt+Tab while the cursor sat over the WARP CORE canvas left the
  bbox-draw cursor stuck system-wide. The annotation widget now
  releases its override cursor on `WindowDeactivate` instead of
  waiting for an Alt-release that already left for another window.
- `Auto-Detect Slots` in WARP CORE used to wipe the WARP window's
  Detection Logs view (it cleared the wrong channel). Now scoped to
  `detection_core`.

## [1.0.0] — 2026-05-20

First tagged release on PyPI. Establishes the standalone `sto-warp`
package — recognition pipeline, trainer (WARP CORE), launcher and
community-sync chain — as a self-contained pipx-installable tool.

### Added
- Auto-clear of the detection-log view on every Autodetect run
  (a `CLEAR` signal in `warp.debug` is honoured by `LogViewWidget`;
  on-disk log files are untouched).
- Three-pane file pickers with per-file image preview and a build-type
  badge so users can verify the autodetector picked the right screen
  type before committing.
- Startup community-seed step plus periodic sync tick; mirror of
  approved crops from Hugging Face for a shared k-NN baseline.
- Sync status surfaced in the status bar with a missing-token warning
  and a pre-commit summary log.
- Automated PyPI release workflow (Trusted Publishing / OIDC) plus
  `install.sh` for one-command pipx install/upgrade.

### Changed
- Consolidated all ship-type and tier-driven slot rules into a single
  `_apply_ship_and_tier_bonuses` helper mirroring SETS
  `get_variable_slot_counts`: Miracle Worker (`Innovation Effects`),
  Federation Intel Holoship, T6-X / T6-X2 bonuses to Universal /
  Devices / Starship Traits, and T5-U / T5-X `t5uconsole` bumps.
- Autodetect logs grouped into **EQUIPMENT**, **TRAITS & REPUTATION**,
  and **BOFF ABILITIES** sections; happy-path matches demoted to
  `DEBUG`, low-confidence / wrong-type / empty-crop cases promoted to
  `WARNING`.
- Constants and configuration values centralised in `warp/config.py`.
- Auto-sync interval raised from 5 minutes to 60 minutes to reduce
  background network traffic.
- Package version now derived from git tags via `hatch-vcs` — no more
  hand-edited `__version__` in `warp/__init__.py`. Release workflow
  fails loudly if the built wheel disagrees with the tag.

### Fixed
- Canvas click in the trainer prefers the current recognition result
  over a saved annotation, fixing the T1 / T6-X2 mismatch where
  clicking a freshly-detected bbox emitted the stale confirmed name.
- Trait-grid CC threshold raised from 30 to 50 so the Space Reputation
  row no longer merges with its bright header banner. Corpus sweep
  across 117 screenshots: +19 % panels, +18 % bounding boxes vs the
  previous threshold; no regressions.
- AUTO mode now trusts OCR over the ambiguous ML `TRAITS` class, so
  Personal Ground Traits screens are no longer mis-classified as
  Space Traits and dropped to zero detections.
- `WARPSyncClient` stops retrying on 4xx client errors and no longer
  trips the circuit breaker on them; thundering-herd on 503 fixed.
- Skip ML screen-type classification on already-confirmed screenshots
  in the trainer to cut needless GPU work.
