# Changelog

All notable changes to **sto-warp** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Entries describe the user-visible changes in each release. Implementation
details live in the git history.

## [1.0.9] — 2026-05-28

### Added
- Icon equivalence classes. When two STO items share visually
  identical icon art (Mk variants, faction reskins, vanity
  duplicates), the trainer no longer raises a community-conflict
  prompt between them — there is nothing to disambiguate from the
  crop. The curated list is mirrored from the community dataset, so
  updates reach every install automatically.

### Changed
- Auto-Detect no longer clears the trainer's review list at the start
  of a run. Confirmed and pending rows survive, and re-running
  recognition only spends time on positions that aren't already
  tracked, so a second pass is much faster.
- Auto-detected items (yellow rows) are no longer treated as
  confirmed training data. They have to be explicitly accepted before
  they are added to the local training set or uploaded to the
  community dataset.

### Fixed
- Auto-detected items in the review panel now render in yellow as
  intended; some used to show green even though they had not been
  accepted.
- **Mark Done** / **Alt+D** no longer lets a session be finished
  while yellow auto-detected items are still pending; the status bar
  reports how many are blocking.
- Re-running Auto-Detect no longer creates duplicate side-by-side
  rows when a fresh detection drifts a pixel or two from an existing
  one.

## [1.0.8] — 2026-05-28

### Added
- Icons that were previously confirmed by the user but disagree
  with the current community recognition are now flagged as a
  "community conflict" (orange) in the trainer's review panel,
  instead of being silently overwritten on the next auto-detect.
  Re-confirming the icon records the rejection so the same
  community proposal stops nagging on later restarts — unless the
  community changes its pick to something new.

### Fixed
- Active equipment icons can no longer be auto-confirmed as empty
  or inactive slots based on earlier mistakes that had been saved
  as training examples. This stops the cycle where one wrong
  "empty" tag would re-poison itself across later sessions.

## [1.0.7] — 2026-05-26

### Changed
- Startup synchronisation of community-shared icon crops is now
  much faster. The mirror is checked against the upstream revision
  first, and only the crops that are actually missing locally get
  downloaded — instead of re-walking the whole file list every
  cycle.

### Fixed
- Crops that were removed from the shared dataset are now also
  removed from the local mirror, with a soft-delete safety net so
  nothing is lost permanently if something goes wrong.

## [1.0.6] — 2026-05-26

### Added
- Manual screen-type override in the **Preview** tab. A dropdown above
  the preview lets the detected screen type be changed per image, and
  a **Rerun Recognition** button re-processes the affected images
  with the new choices.

### Fixed
- Space equipment screenshots with a BOFF ability tooltip visible on
  top are no longer misread as a BOFF panel (which previously
  produced phantom BOFF seats out of the tooltip text).
- When two screenshots contribute to the same slot (e.g. a dedicated
  BOFFS screen and a mixed screen), the dedicated screen now wins
  ties, so a clean BOFFS capture isn't overridden by a noisier mixed
  one.

## [1.0.5] — 2026-05-25

Security release.

### Changed
- Uploads now go through a server-side endpoint, so an upload token
  no longer needs to be stored locally.

### Removed
- The shared upload token file is automatically deleted on first
  launch after upgrade.

## [1.0.4] — 2026-05-24

Ground BOFF recognition overhaul — earlier releases recognised only
a small fraction of ground abilities; this release brings full
coverage.

### Added
- Ground BOFF abilities are now fully recognised (earlier releases
  only handled a small fraction of them).
- Space BOFF coverage has been brought up to the full set of
  abilities as well.
- Ground BOFF seats are now detected separately from space seats —
  ground slots only receive ground-ability suggestions.

### Fixed
- Empty and greyed-out ability slots are now recognised as empty
  instead of being labelled as random abilities (the most visible
  symptom was every blank slot being identified as *Charged Particle
  Burst*).

## [1.0.3] — 2026-05-21

Trainer (WARP CORE) bug-fix round.

### Added
- New **Clear All BBoxes** button in the trainer with three options:
  clear everything, keep already-confirmed boxes, or cancel.

### Fixed
- The ship name and tier shown in the trainer's review panel now
  reliably reflect what was recognised: selecting a row updates the
  editor to match, re-running Auto-Detect applies the fresh
  recognition instead of being overridden by an older confirmation,
  and confirming an empty value no longer wipes the recognised one.

## [1.0.2] — 2026-05-21

Progress feedback and responsiveness improvements.

### Added
- Unified progress bar across WARP and WARP CORE with a working
  **Cancel** button for stopping a run in progress.
- Per-stage percentage during Auto-Detect in the trainer, showing
  which slot is currently being processed.

### Changed
- The blocking "Detecting Screen Types" / "Recognising Icons"
  dialogs are gone — progress now shows in the status bar instead,
  so the main window stays interactive while detection runs.

### Fixed
- Closing the app while a sync or detection is in progress is now
  immediate — the window no longer waits several seconds for the
  background work to finish before shutting down.

## [1.0.1] — 2026-05-21

User interface polish. No recognition-behaviour changes — drop-in
upgrade.

### Added
- Detection logs are now separate for WARP and WARP CORE — runs no
  longer overwrite each other.
- **Copy All** and **Save As…** buttons on the Detection Logs view.
  Saved files get a default name based on the active screenshot.
- Centralised theme — the whole app's appearance can be swapped via
  the `WARP_THEME` environment variable.

### Changed
- Toolbar and button styling unified between WARP and WARP CORE.
- The **Force build type** dropdown is now visibly greyed out when
  disabled (previously it looked active but wasn't).
- The active tab is now clearly highlighted.
- Results list: slot headings are bolder; single-value slots (Ship
  Name / Type / Tier) show their value on the heading row so it's
  visible without expanding.

### Fixed
- Alt+Tab while hovering the trainer canvas no longer leaves the
  cursor stuck in the drawing-cursor shape system-wide.
- Running Auto-Detect in WARP CORE no longer wipes WARP's Detection
  Logs view.

## [1.0.0] — 2026-05-20

First public release on PyPI. Install with `pipx install sto-warp`.

sto-warp is the standalone successor to the WARP / WARP CORE tools
that previously lived inside sets-warp.

### What's included
- **WARP** — screenshot recognition for Star Trek Online builds
  (equipment, traits, BOFF abilities, ship name & tier).
- **WARP CORE** — trainer for reviewing detection results and
  feeding corrections back into the model.
- **Community sync** — confirmed crops and the trained model are
  shared via Hugging Face, so corrections from every install
  improve the shared baseline.
- Three-pane file picker with image preview and a build-type badge
  for verifying the screen type before importing.
- One-command install / upgrade via `pipx`.

### Behaviour notes
- Background sync runs every 60 minutes (down from the 5-minute
  beta interval) to keep network use light.
- Recognition accuracy improvements for ship tier / type, traits,
  and edge cases such as T6-X2 detection, Personal Ground Traits
  vs Space Traits, and low-confidence trait grid rows.
