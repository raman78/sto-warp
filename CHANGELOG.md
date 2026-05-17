# Changelog

All notable changes to **sto-warp** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial repository scaffold split out from `sets-warp`.
- `warp.debug` standalone logger replaces `src.setsdebug`.
- Foundation modules ported from `sets-warp/warp/`:
  - `warp.recognition` — `boff_keys`, `boff_marker`, `eq_geometry`,
    `ground_eq_geometry`, `icon_matcher`, `layout_detector`,
    `screen_classifier`, `text_extractor`, `trait_grid`.
  - `warp.trainer` — `ocr_diag`, `screen_type_trainer`.
  - `warp.knowledge` — `sync_client`.
- Full `docs/` tree carried over from `sets-warp` (to be pruned).

### Pending
- Cargo data loader fetching from `STOCD/SETS-Data` (strategy iii), cache
  under `~/.config/warp/cache/`.
- `pyproject.toml` for pipx distribution; `sto-warp` console entry point.
- Standalone replacements for the WARP CORE trainer UI files that still
  depend on the SETS Qt scaffolding.
- Native packaging recipes (AUR, .deb, .rpm, Windows MSI/EXE).
