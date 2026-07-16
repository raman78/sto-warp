# Changelog

All notable changes to **sto-warp** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Entries describe the user-visible changes in each release. Implementation
details live in the git history.

## [1.0.24] — 2026-07-16

### Changed
- **The way a screenshot's type is set in WARP CORE has moved.** The
  screen-type menu now opens on a **double-click** of a file in the
  list. **Right-clicking** a file no longer changes its type — it now
  copies the file name or full path instead, matching the right-click
  menu on the WARP recognition tabs.

### Fixed
- The hover tooltip on a bounding box now shows the icon and name of the
  item that was confirmed, instead of keeping the originally detected
  icon after a correction. It also states whether the item was confirmed
  by the user or auto-accepted by the program.
- Trait tooltips now show the trait's reference icon instead of no icon
  (for example, Hive Defenses).

## [1.0.23] — 2026-07-10

### Added
- Hovering over a recognized item — on the canvas or in the review/results
  tree — now shows a tooltip with the reference icon thumbnail next to the
  item name, slot, and confidence. Comparing the reference icon with the
  screenshot bbox helps spot mismatches at a glance.
- Right-clicking a recognized item anywhere (canvas, review tree, results
  tree) offers two new actions: **Open on vger.stobuilds.com** opens the
  matching category page (equipment, traits) for quick search, and
  **Open on STO Wiki** opens the item's wiki page directly. Both features
  are available in WARP, WARP CORE, and Fast Correction Mode.

## [1.0.22] — 2026-06-29

### Fixed
- Equipment recognition is more accurate: items missing from the
  built-in gallery are now correctly identified using their wiki
  image instead of being labelled with the wrong nearest match,
  and icons in the top row are no longer cut off at the screenshot
  edge.

## [1.0.21] — 2026-06-19

### Fixed
- Universal Bridge Officer seats with mostly empty or inactive abilities
  are now identified by the correct profession instead of defaulting to
  Science.
- Changing a Bridge Officer group's type to a profession that already has
  a group no longer merges the two groups — each physical seat keeps its
  own group, and groups of the same type are numbered by their position
  on screen.
- The Kit slot on ground equipment screens is now detected even when OCR
  fails to read the short "Kit" label.

### Changed
- Ship Name is no longer shown as a separate slot in the results or on
  the preview canvas. The ship name is still used internally for layout
  detection.

## [1.0.20] — 2026-06-16

### Fixed
- Bridge Officer abilities on mixed screens (showing both equipment and
  officers) are no longer falsely detected when no profession colour
  markers are visible. Previously, unrelated UI elements could be
  misidentified as officer ability slots.
- Starship Trait icon positions on the second row are now correctly
  placed after the ship-name divider gap, instead of using a fixed
  spacing that could miss or misalign the trait boxes.

## [1.0.19] — 2026-06-16

### Added
- Right-clicking a Bridge Officer group header in the review tree shows
  a **Change Group Type** menu. Selecting a new type rematches all
  abilities in the group against the new profession at once, instead
  of requiring each ability to be corrected individually.
- Alt+Up / Alt+Down keyboard shortcuts navigate between screenshots in
  the trainer file list.
- Screenshots that are not build screens (menus, loading screens, etc.)
  are now classified as **Discard** and automatically marked Done — no
  manual review needed.
- Skills screens are now recognised as a separate screen type instead
  of being misclassified as equipment or traits.

### Fixed
- Empty and inactive slots — locked equipment positions, unequipped
  slots, locked Bridge Officer abilities — are now correctly recognised
  as empty or inactive instead of being mislabelled as random items.
- After changing a Bridge Officer group's type, the group stays in its
  original visual position instead of jumping to the wrong spot.
- When a numbered Bridge Officer group is removed or reassigned
  (e.g. "Boff Science #2" becomes Tactical), the remaining group drops
  its unnecessary "#1" suffix.

## [1.0.18] — 2026-06-11

### Added
- **Multi-language recognition.** Screenshots taken with a non-English
  STO client (German so far) are now recognised correctly. Equipment
  names, ship types and console categories that appear in the local
  language are translated to English automatically during detection, so
  WARP can match them against the reference database. The translation
  table is a plain CSV that can be extended by the community when new
  languages are needed.

### Fixed
- Items within the same slot group (e.g. several identical Isomag
  consoles in Engineering) now display in a stable top-to-bottom,
  left-to-right order instead of shuffling randomly between sessions.
- Removing a bounding box in Fast Correction Mode now correctly
  removes it from the saved data as well, so **Send to WARP** no
  longer re-emits an item that was already deleted.
- Near-overlapping bounding boxes (a 1–2 pixel shift between
  recognition runs) no longer create duplicate annotations for the
  same slot position.
- BOFF seat assignment is more accurate: abilities are better paired
  to the correct seat, and the seat tag is preserved through export.
- The screen-type confirmation checkbox in the file list now only
  reflects a manual confirmation by the user — automatic
  classification by the model no longer ticks it.
- The community upload filter blocks known-bad labels again after a
  brief gap where the check was inadvertently skipped.
- Ship tier and rank lookups use the live reference data instead of a
  stale snapshot that could drift as the upstream files were updated.

## [1.0.17] — 2026-06-10

### Fixed
- The bundled offline copy of the BOFF-ability table has been
  refreshed to the current upstream format. A clean install with no
  network would otherwise fall back to a stale copy left over from
  the previous data shape, which now silently breaks BOFF
  recognition on the first launch until reference data can be
  downloaded.

### Added
- Internal data caches (equipment, ships, traits, starship traits,
  BOFF abilities) are now checked against a shape contract whenever
  the reference data refreshes. Any future drift between the
  upstream files and what WARP expects is logged as a warning
  instead of silently degrading recognition — the same kind of
  drift that caused the BOFF regression fixed in v1.0.16.

## [1.0.16] — 2026-06-10

### Fixed
- BOFF abilities are recognised again. A reference-data loader bug
  introduced in v1.0.6 left the internal BOFF-ability lookup table
  as a flat list instead of the expected per-environment and
  per-profession buckets. Most recognition paths silently degraded:
  candidates were no longer filtered by profession (weaker matches),
  the slot-content check let any name through, and the SETS export
  dropped every BOFF ability from the written build. On screenshots
  containing a Universal BOFF seat the recognition worker also
  crashed outright with `'list' object has no attribute 'get'`.
  Re-importing any previously affected screenshot now writes BOFF
  abilities to the exported build correctly.

## [1.0.15] — 2026-06-08

### Added
- A setup splash now appears on the very first launch and blocks the
  main window until every piece of reference data is downloaded:
  cargo (equipment, traits, ships, BOFF abilities), item and ship
  icons, community knowledge, the recognition model, the community
  icon library, the icon-match template index, and the curated
  icon-equivalence list. Progress bars show what is happening for
  the slow steps. The splash only appears once — after every phase
  finishes successfully the launcher remembers it and starts
  silently on later runs. **Close** exits cleanly; **Cancel** lets
  the program start without the full library, at the cost of weaker
  recognition for that session. An interrupted splash will simply
  reappear on the next launch.

### Changed
- The community icon library is now downloaded as a single archive
  rebuilt weekly from the upstream dataset, instead of one HTTP
  request per crop. A clean install is now much faster and no
  longer stalls on shared-bandwidth limits. If the archive is
  unavailable the previous per-file download path is used as a
  fallback.
- **Export to SETS JSON** has been moved directly beneath the
  Results file list in WARP, so it stays reachable without
  scrolling on smaller windows. Behaviour is unchanged.

### Fixed
- Reference data now actually stays fresh between launches. Cargo
  data and the curated icon-equivalence list were configured to
  auto-refresh on a daily window, but nothing in the running
  program ever triggered that refresh — both files could sit
  frozen for weeks while the application was launched daily.
  They are now part of the regular startup-sync cycle, so each
  launch re-verifies them against the upstream source (with a
  one-per-day rate limit so the actual network call still happens
  at most once a day per file). The system log prints one line
  per file per cycle confirming the result, so verification is
  visible rather than implied.
- The Recognition Review panel no longer mixes up bounding boxes
  between two screenshots that happen to share the same file name
  (e.g. `overview.png` from two different builds). Screen-type
  labels follow the same rule.
- The trait review tree now lists **Personal Ground Traits** above
  **Personal Space Traits** in WARP CORE, matching the in-game
  order and the rest of the review panel's slot ordering.
- **Mark Done** is greyed out until every review row on the
  current screenshot is confirmed, so a half-corrected screenshot
  cannot be locked by accident.
- A corrected BOFF slot now stays inside its original seat group
  instead of jumping into a different rank's seat when the rank
  label changes.
- The Space Reputation extrapolation no longer fires when the
  predicted band is too dark to read, removing a class of
  phantom-trait suggestions on screenshots with the reputation
  panel partly off-screen.
- Moving a bbox row in the review list keeps the row at its
  original position, so the slot order on screen and the order in
  the tree stay in sync.
- The system log no longer fills with noise from harmless
  upstream-read warnings and per-batch upload rejections during
  normal community sync.

## [1.0.14] — 2026-06-02

### Added
- The WARP CORE Recognition Review panel now groups BOFF abilities by
  physical bridge officer seat instead of by profession. Each
  expandable header names the seat (e.g. **Boff Tactical #1**, **Boff
  Engineering+Temporal**) and lists its abilities in the same
  left-to-right order they appear on the in-game UI. Clicking a group
  header highlights every bbox inside that seat on the canvas preview
  at once, so the seat layout can be verified at a glance.

### Changed
- Adding a bbox or changing a row's slot now moves the row to its
  correct position in the review tree, matching the in-game reading
  order, instead of leaving it at the bottom of the group it ended
  up in.
- Pressing **Enter** on the review list now skips past rows already
  auto-accepted by recognition and lands on the next row still
  waiting for confirmation, so working through a batch is faster.
- When a build has multiple BOFF seats of the same profession at
  different ranks (e.g. two Tactical seats — Lieutenant Commander
  and Lieutenant), the larger detected cluster of abilities is now
  assigned to the higher-rank seat on SETS export. The first cluster
  by index used to always take the first seat, which often swapped
  the two on Avenger-class builds.
- The **↗ Send to WARP** button is now hidden when the trainer is
  not in Fast Correction Mode. It used to sit visible but inert on
  the standing trainer tab.

### Fixed
- BOFF review groups now sort in proper top-to-bottom,
  left-to-right reading order even on screenshots where equipment
  row pitch is close to the gap between BOFF seat rows. On affected
  builds the middle-row left seat could previously appear before
  the top-row left seat.
- Starship Traits are no longer miscounted when the visible top row
  of the trait panel has only 4 active icons instead of the
  expected 5 — the layout merge no longer drops or duplicates icons
  from the row underneath.
- Old confirmed annotations saved before per-seat grouping shipped
  now backfill their seat assignment automatically on the next
  Auto-Detect of the matching screenshot, so the review tree
  immediately reflects per-seat grouping without needing a manual
  re-confirm.

## [1.0.13] — 2026-06-01

### Added
- **Native macOS launcher** — on first launch sto-warp now drops a
  `sto-warp.app` bundle into `~/Applications/`, so the app surfaces in
  Launchpad, Spotlight, and the Dock alongside other Mac applications.
  Previously the app could only be started from a terminal. The bundle
  is a thin wrapper around the pipx-installed binary; no code signing
  or notarisation is involved, so it launches without the "damaged
  app" Gatekeeper warning that signed-but-unnotarised downloads would
  trigger. After a `pipx upgrade` any stale bundle pointing at the
  same install is pruned automatically, so no duplicates accumulate
  in Launchpad.

### Changed
- **Rounded-corner app icon across all platforms** — the application
  icon now has a soft squircle shape that matches the look of native
  applications on macOS (Big Sur+), KDE Plasma, and modern GNOME.
  The change applies consistently to the Linux `.desktop` icon, the
  macOS `.icns` bundle icon, the Windows Start Menu shortcut `.ico`,
  *and* the live window / taskbar / title-bar icon visible while
  sto-warp is running. Linux and Windows installs on 1.0.12 pick up
  the rounded icon automatically on the next `pipx upgrade` or after
  re-running `sto-warp install-desktop`.

### Fixed
- Clicking a slot on the canvas in WARP CORE (and in Fast Correction
  Mode) now reliably bolds the matching row in the review list. The
  bold previously stayed on whichever row was selected before the
  click.
- Switching between items on a Mark Done-locked screenshot no longer
  re-enables the locked Item name field — the lock now persists for
  every item in the screenshot, not just the one selected when the
  screenshot was marked done.
- Starship traits no longer get counted twice when a nearby noise
  area is misclassified as a second "Starship Traits" section. The
  structural detection wins and the duplicate is dropped.

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
