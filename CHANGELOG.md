# Changelog

All notable changes to **sto-warp** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Entries describe the user-visible changes in each release. Implementation
details live in the git history.

## [1.0.12] — 2026-05-31

### Added
- **Windows .exe installer** — sto-warp can now be installed on
  Windows by downloading the installer attached to the GitHub
  Release, with no Python toolchain required. The installer is
  per-user (does not need administrator rights), drops the app
  under `%LOCALAPPDATA%\Programs\sto-warp`, registers a Start Menu
  entry and the optional desktop shortcut, and ships a CPU-only
  PyTorch build so the bundle stays compact. The existing `pipx
  install sto-warp` path continues to work and is still the
  recommended channel on Linux.

## [1.0.11] — 2026-05-31

### Added
- **Fast Correction Mode** — a new lightweight bridge between WARP
  recognition and a clean SETS JSON export. Right-clicking a row in
  the WARP Results tree exposes *Open in WARP Fast Correction Mode*,
  which hands the entire current batch to WARP CORE in a temporary
  workspace. Wrong items can be corrected with the usual trainer
  flow, and a single **↗ Send to WARP** button pushes the corrected
  result back to WARP for export. Corrections made in this mode are
  ephemeral by design: the original screenshots' permanent training
  data is never overwritten, but the confirmed crops still feed the
  community sync queue. The trainer tab is clearly relabelled and
  themed while a Fast Correction session is active, and the launcher
  automatically cleans up sessions older than 14 days.
- The screen-type detection pass — both the automatic one in WARP
  on folder open and the manual *Detect Screen Types* button in
  WARP CORE — now shows a determinate progress bar with a working
  **Cancel** button. Files already classified keep their detected
  type; the remaining files stay on their previous value.
- While screen-type detection is running, the surrounding toolbar
  actions are temporarily greyed out in both WARP and WARP CORE so
  a second pass cannot be started on top of the first.
- The WARP CORE Recognition Review panel is now organised as a
  grouped tree — each slot family is one expandable header with a
  child count, items are listed in pipeline order underneath, and
  the currently selected row is rendered in bold across both
  columns to stay visually anchored while scrolling.

### Changed
- Sending the corrected batch back to WARP from Fast Correction Mode
  now closes the loop end-to-end: the trainer automatically exits
  Fast Correction, the tab title and accent are restored, and the
  trainer view is rewound to the folder, selection and filter that
  were active before the Fast Correction session was entered. The
  standing training work is no longer replaced by the ephemeral
  batch when the loop completes.
- The Item ↔ Conf column edge in the WARP CORE Recognition Review
  panel is draggable. The chosen width is remembered across
  sessions.
- The selection accent is now consistent across the file list, the
  Review tree and the Annotate panel, so the currently active item
  is obvious at a glance.
- The original screenshot filenames are shown throughout Fast
  Correction Mode — in the file list, dialogs and status messages —
  even though the files are staged into a hashed temporary folder
  underneath. The staging hash never surfaces in the UI.

### Fixed
- After a `pipx upgrade`, the launcher no longer leaves a fresh
  duplicate menu entry behind. On Linux any sibling
  `~/.local/share/applications/sto-warp-*.desktop` that points at
  the same `sto-warp` shim as the current install is removed on
  startup; on Windows the same cleanup is applied to the Start Menu
  `.lnk` shortcuts. Genuinely parallel installs targeting a
  different binary path are left alone.
- *Send to WARP* in Fast Correction Mode sends the entire batch of
  screenshots, not just the one that happened to be selected when
  the button was clicked.
- The Done colour in the Fast Correction file list no longer
  persists across sessions. Files start in the editable state every
  time a new batch is loaded, regardless of whether the originals
  were marked Done in an earlier normal trainer session.
- The confirmation checkbox next to a filename no longer disappears
  after a correction is accepted in Fast Correction Mode — it now
  correctly reflects either a manual confirmation or a model
  auto-classification, with the appropriate icon for each.
- Accepting an item on a screenshot already marked Done is now
  correctly blocked, matching the rest of the locked-screenshot
  guards (Remove, Clear All BBoxes, Auto-Detect, Delete key). The
  canvas was already visually locked and the button already read
  *↩ Back to Edit*, but Enter, the Accept button, picking from the
  autocomplete dropdown and the Ship Type / Ship Tier combos still
  let an annotation through. The status bar now explains why the
  action was skipped when this is attempted.
- The Annotate Selection panel (Slot dropdown, Item field, Accept
  button and the Ship Tier / Ship Type combos) is now disabled
  whenever the current screenshot is marked Done. Selecting a slot
  no longer reactivates the Item field on a locked screenshot, and
  the placeholder text inside the field changes to explain why
  editing is blocked. Toggling *↩ Back to Edit* immediately
  restores the panel.
- The WARP CORE Recognition Review tree's Slot and Idx columns now
  always render in the chrome's neutral white, regardless of the
  row's confidence or status. Previously these structural grouping
  columns inherited the confidence colour, so the labels shifted
  between white, pink and red depending on the row underneath. The
  Item, Conf and Status columns still carry the colour as before,
  matching the long-standing Fast Correction Mode convention.
- Hovering with **Shift** to resize a bbox edge is now responsive
  along the full length of every edge in the WARP CORE canvas, not
  only within a narrow zone around the midpoint handle. The cursor
  switches to the appropriate resize shape as soon as it crosses
  the edge band, regardless of how far along the side it is.
- **Alt** (the icon-crop cursor) and the other canvas modifier keys
  no longer react when the cursor is hovering a side panel while
  the image is zoomed above fit. The trigger area is now strictly
  the visible canvas panel, not the (potentially much larger)
  zoomed image rectangle.

## [1.0.10] — 2026-05-30

### Added
- A Start Menu entry is now created on Windows on the first launch,
  matching the behaviour that already existed on Linux. The shortcut
  is placed under *Start ▸ Programs ▸ sto-warp*, where it can be
  pinned to the taskbar or searched for from Start.
- Confirmed corrections are fed back into the recognition pipeline
  immediately, in the same session. A misrecognised icon that was
  just fixed will match correctly the next time it appears, without
  waiting for the next community-model download.
- The Detection logs now tag each recognised item with its match
  origin — **[USER]**, **[COMMUNITY]**, **[WARP CORE]** (local
  training set) or **[SESSION]** (matches accumulated during the
  current run) — so it is easier to see which source carried a
  given result.
- The Screenshots list in WARP CORE has a new **Filter by filename**
  box above the list. Typing narrows the visible files in real time;
  the ✕ button clears the filter.
- The WARP Results tree has a right-click menu on every row:
  **Copy filename**, **Copy full path** and (in the launcher window)
  **Open in WARP CORE**, which jumps straight to the trainer tab
  with the chosen screenshot loaded.
- Ship-class matching tolerates OCR-garbled type names. A class name
  with extra symbols or missing letters still resolves to the right
  in-game class as long as enough significant words survive.

### Changed
- Ship Tier OCR is more forgiving with misread bracket contents.
  When the tier reads as something like `[T6-Xz]` (a `2` misread as
  `z`) or `[TB-X2]` (a `6` misread as `B`), the result is snapped
  to the closest real tier, preferring the higher tier when two
  canonical values are equally plausible. Clean tier readings are
  unaffected.
- Tier corrections that would silently demote a higher tier to a
  lower one (for example `T6-X2 → T1`) are now rejected both when
  the community model is downloaded and when it is loaded locally.
  A bad correction can no longer slip in and pollute the running
  session.
- Decisions made in the training-data cleanup tool are remembered.
  Pressing **N (keep)** on a flagged crop during a
  `python -m warp.tools.scrub_training_data --review` session
  persistently marks that crop as reviewed, so the matching startup
  warning will not appear again.
- The Windows installer script installs Python 3.14 (which sto-warp
  requires) instead of an older version, so a fresh winget-driven
  install completes without a Python-version error on a clean
  Windows box.

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
