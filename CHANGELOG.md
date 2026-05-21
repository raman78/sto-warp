# Changelog

All notable changes to **sto-warp** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
