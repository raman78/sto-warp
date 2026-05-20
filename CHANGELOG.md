# Changelog

All notable changes to **sto-warp** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Pending
- Native packaging recipes (AUR, `.deb`, `.rpm`, Windows MSI/EXE).

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

### Removed
- `[Unreleased]` placeholders for items now shipped: `pyproject.toml`,
  `sto-warp` console entry point, cargo data loader, and the
  standalone WARP CORE trainer UI.
