# Changelog

All notable changes to **sto-warp** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial repository scaffold split out from `sets-warp`.
- `warp.debug` standalone logger replaces `src.setsdebug`; split into
  detection / system channels (`warp_detection.log` / `warp_system.log`).
- Foundation modules ported from `sets-warp/warp/`:
  - `warp.recognition` — `boff_keys`, `boff_marker`, `eq_geometry`,
    `ground_eq_geometry`, `icon_matcher`, `layout_detector`,
    `screen_classifier`, `text_extractor`, `trait_grid`.
  - `warp.trainer` — `ocr_diag`, `screen_type_trainer`.
  - `warp.knowledge` — `sync_client`.
- Standalone GUI (`sto-warp` / `sto-warp gui`) — `warp.gui.warp_window.WarpWindow`
  replaces the SETS-coupled `warp_dialog.py`:
  - **Open Screenshot…** (single-file picker) and **Open Folder…** routed
    through custom split-pane pickers; last-used dirs persisted.
  - **Force build type** toggle: off → AUTO mode (per-image classification);
    on → combo value forced on every image. State persisted via QSettings.
  - **Results** tab: tree grouped by slot, ship banner with name/type/tier.
  - **Preview** tab: bbox overlay coloured per slot family (weapons red,
    BOFF orange, traits green, spec magenta, consoles violet, EQ blue,
    ground EQ cyan).
  - **Export to SETS JSON…**: SETS v3.0.0-compatible build file.
- AUTO build-type mode + 3-factor cross-image merge in `WarpImporter`
  (0.3 · geometry + 0.3 · sibling + 0.4 · recog) replaces per-slot max-conf
  merging across same-typed screenshots.
- Launcher (`warp.gui.launcher.LauncherWindow`) hosts WARP, WARP CORE,
  Detection logs and System logs as tabs; `🔄 Refresh` button in the status
  bar manually re-runs the community sync.
- Detection logs view: terminal-like auto-tail with preserved horizontal
  scroll, fresh recognition runs wipe the live view; Open-folder button
  opens the log directory.
- Full `docs/` tree carried over from `sets-warp` (planning files pruned).

### Pending
- Cargo data loader fetching from `STOCD/SETS-Data` (strategy iii), cache
  under `~/.config/warp/cache/`.
- `pyproject.toml` for pipx distribution; `sto-warp` console entry point.
- Standalone replacements for the WARP CORE trainer UI files that still
  depend on the SETS Qt scaffolding.
- Native packaging recipes (AUR, .deb, .rpm, Windows MSI/EXE).
