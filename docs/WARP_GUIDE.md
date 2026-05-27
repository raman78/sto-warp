# WARP & WARP CORE — User Guide

WARP reads your Star Trek Online screenshots, recognises every slot (weapons, traits, BOFF abilities, …) and exports the result as a SETS v3.0.0-compatible build JSON.
WARP CORE lets you review, correct, and confirm what WARP found — and feed those corrections back into the community model so future recognition improves.

---

## Table of contents

1. [Preparing screenshots](#1-preparing-screenshots)
2. [Launcher window — tabs and global controls](#2-launcher-window--tabs-and-global-controls)
3. [Using WARP — recognise a build](#3-using-warp--recognise-a-build)
   - [Force build type vs AUTO mode](#force-build-type-vs-auto-mode)
   - [Run recognition](#run-recognition)
   - [Results tab](#results-tab)
   - [Preview tab — bbox overlay](#preview-tab--bbox-overlay)
   - [Export to SETS JSON](#export-to-sets-json)
4. [WARP CORE — interface overview](#4-warp-core--interface-overview)
5. [Reviewing and correcting recognition](#5-reviewing-and-correcting-recognition)
6. [Confirming items and accepting results](#6-confirming-items-and-accepting-results)
7. [Detection logs / System logs tabs](#7-detection-logs--system-logs-tabs)
8. [Community model — how it works](#8-community-model--how-it-works)
9. [Community sync details](#9-community-sync-details)
10. [Keyboard shortcuts](#10-keyboard-shortcuts)
11. [Tips and troubleshooting](#11-tips-and-troubleshooting)

---

## 1. Preparing screenshots

### What to capture

WARP reads the standard STO build screens. Open your ship/character loadout in-game and take full-screen screenshots of:

| Screen | Contains |
|--------|----------|
| Space Equipment | Weapons, shields, engines, deflector, devices, consoles |
| Ground Equipment | Ground weapons, armor, kit, kit modules, devices |
| Space Traits | Personal space traits, starship traits, reputation traits |
| Ground Traits | Personal ground traits, reputation traits |
| Bridge Officers | Boff seats and abilities (space or ground) |
| Specializations | Primary and secondary specialization trees |

> **Screenshot tip:** Use the default STO screenshot key (default: **Print Screen**) to save full-resolution screenshots. Cropped or resized images may reduce recognition accuracy.

### How many screenshots per folder

**One build per folder.** Each folder is one import session. You can mix screen types freely:

- **Separate screens** — one screenshot per game tab. WARP identifies each screen type automatically.
- **Mixed screen** — a single screenshot that combines multiple tabs (assembled view). WARP detects the layout automatically.
- **Partial** — only some screens (e.g. equipment only, or traits only). Slots for missing screens are left empty.

### Recommended folder structure

```
my_build/
    space_equipment.jpg
    space_traits.jpg
    boffs.jpg
    ground_equipment.jpg
```

<!-- screenshot: example folder with 4 screenshots in file manager -->

---

## 2. Launcher window — tabs and global controls

`sto-warp` opens a single launcher window with four tabs:

| Tab | Purpose |
|-----|---------|
| **WARP — Recognition** | Open one screenshot or a whole folder, run the recognition pipeline, review results, export to SETS JSON |
| **WARP CORE — Trainer** | Review and confirm detected items so they feed back into the community model |
| **Detection logs** | Terminal-like live tail of the current detection run; auto-clears when a fresh run starts |
| **System logs** | Background activity (asset sync, model updates, knowledge cache, desktop integration) — separate from detection noise |

A `🔄 Refresh` button in the bottom-right status bar manually re-runs the
community sync (re-download knowledge, check for newer central models, upload
pending confirmed crops). It is disabled while a sync is already in flight.

Window geometry and tab state are persisted across runs.

---

## 3. Using WARP — recognise a build

The **WARP — Recognition** tab has a single toolbar:

| Control | What it does |
|---------|--------------|
| **Open Screenshot…** | Single-file picker (split-pane dark dialog). The file is staged into a temporary directory so the folder pipeline picks it up unchanged. |
| **Open Folder…** | Folder picker for multi-image builds. Every screenshot in the folder is processed in one run. |
| **Force build type** (checkbox + combo) | See below. |
| **Export to SETS JSON…** | Enabled after a successful run. Writes a SETS v3.0.0-compatible build file. |

Both pickers remember the last directory you used and reopen there on the
next click.

### Force build type vs AUTO mode

When the **Force build type** checkbox is **off** (the default), WARP runs in
**AUTO mode**: every screenshot in the folder is classified independently by
the screen classifier + OCR, and per-image build type is derived from that.
This is the right mode for mixed folders where some screenshots are equipment
and others are traits, BOFFs, or specialisations.

When the checkbox is **on**, the combo's value is forced on every image — useful
when the classifier mis-reads a deliberately cropped capture and you want to
short-circuit it. Available values: `SPACE_MIXED`, `GROUND_MIXED`, `SPACE`,
`GROUND`, `BOFFS`, `SPACE_BOFFS`, `GROUND_BOFFS`, `SPACE_TRAITS`,
`GROUND_TRAITS`, `SPEC`.

Both the checkbox state and the last selected build type are persisted.

**Cross-image merge (AUTO mode).** When multiple screenshots in the folder
classify to the same build type (e.g. two SPACE_EQ frames), WARP no longer
keeps the highest-confidence item per slot. Instead each slot's candidates
compete on a blended score:

```
0.3 · geometry  (X-pitch regularity across the block)
0.3 · sibling   (panel-group coverage from the same source image)
0.4 · recog     (mean confidence; low-confidence virtuals halved)
```

This prevents a confident `__empty__` from a mis-classified frame from
overwriting a real item from the correctly classified one.

### Run recognition

Click **Open Screenshot…** or **Open Folder…**. WARP processes each
screenshot through this pipeline:

```
   [screenshot.png]
        │
        ▼
  ┌─────────────────────┐
  │ 1. Screen classify  │  MobileNetV3-Small  →  SPACE_EQ / GROUND_EQ /
  └─────────────────────┘                        TRAITS / BOFFS / MIXED / …
        │
        ▼
  ┌─────────────────────┐
  │ 2. OCR              │  EasyOCR  →  ship_name / ship_type / ship_tier
  │    (space screens)  │              (anchored on T6-X2 / T6-X / Tx tokens)
  └─────────────────────┘
        │
        ▼
  ┌─────────────────────┐
  │ 3. ShipDB lookup    │  type-first match (783 ships)  →  ship profile
  │                     │  • exact type  • word-subset (boff-Jaccard tiebreak)
  │                     │  • fuzzy 0.68  • keyword fallback
  └─────────────────────┘
        │
        ▼
  ┌─────────────────────┐
  │ 4. Layout detection │  per build_type, in priority order:
  │                     │   • Strategy 0:  BOFF marker grid (BOFFS / MIXED)
  │                     │   • Strategy 1:  EQ geometry detector (OCR-anchored)
  │                     │   • Strategy 1G: Ground EQ geometry
  │                     │   • Strategy 2:  pixel analysis (legacy)
  │                     │   • Strategy 3:  learned anchors.json
  │                     │   • Strategy 4:  default calibration
  └─────────────────────┘
        │
        ▼
  ┌─────────────────────┐
  │ 5. Icon matching    │  per slot crop  →  RecognisedItem
  │                     │   template / histogram k-NN / ArcFace embed /
  │                     │   EfficientNet softmax / community pHash
  └─────────────────────┘
        │
        ▼
  ┌─────────────────────┐
  │ 6. Render to GUI    │  RecognisedItem stream → Results tree + Preview tab
  │                     │  + ship banner. SETS write happens only on demand
  │                     │  via Export to SETS JSON…
  └─────────────────────┘
```

A progress bar in the status bar shows the current step. Per-image progress is
forwarded as `[done/total] file.png` text, and the sub-stage progress bar moves
smoothly within each image's OCR / classify / layout / per-slot matching window
so a single-image run does not just jump 0 → 100%.

Recognition typically takes 5–30 seconds per screenshot, depending on the
number of screens, image resolution, and whether your hardware is CPU- or
GPU-accelerated.

### Results tab

After a run finishes, the **Results** tab shows a tree grouped by slot:

```
Slot              Idx   Item                           Conf   Source
─────────────────────────────────────────────────────────────────────
Ship Name         1     U.S.S. Enterprise              –      eq.jpg
Ship Type         1     Fleet Heavy Cruiser            –      eq.jpg
Ship Tier         1     T6-X                           –      eq.jpg
Fore Weapon       4     ▸ (expanded)
                  1     Phaser Beam Array Mk XV          0.94   eq.jpg
                  2     Phaser Beam Array Mk XV          0.91   eq.jpg
                  …
```

Order: ship metadata first (the three OCR signals SETS needs), then the
canonical pipeline order (equipment → BOFFs → traits → spec), then anything
else sorted alphabetically. A bold ship banner appears above the tree with
the recognised name / type / tier so you can sanity-check OCR at a glance
before exporting.

### Preview tab — bbox overlay

The **Preview** tab visualises *what WARP saw*. The left pane lists every
source file from the latest run; the right pane paints the selected
screenshot fitted to the viewport with bounding boxes drawn over each
detected slot. Each box is stamped with the slot name and confidence.

Box colour is keyed to slot family — deterministic so the same slot always
gets the same hue across screenshots in a batch:

| Family | Colour |
|--------|--------|
| BOFF abilities | Orange |
| Traits | Green |
| Specializations | Magenta |
| Consoles | Violet |
| Deflector / engines / warp core / shields / devices | Blue |
| Weapons | Red |
| Ground armor / kit / kit modules / personal shield / ground devices | Cyan |
| Everything else | Stable hashed hue derived from the slot string |

Use Preview to spot mis-aligned bboxes (wrong row, off-by-one column) at a
glance, then jump into WARP CORE to correct them. The preview is read-only —
it never re-runs detection.

### Export to SETS JSON

When a run finishes the **Export to SETS JSON…** button enables. It writes a
SETS v3.0.0-compatible build file (via `warp.build_writer` + `warp.sets_export`)
that you can load in the SETS build planner via `File → Load Build`.

The status bar reports a one-line summary on success:

```
SETS build → /path/to/build.json  ·  ship=Fleet Heavy Cruiser
            eq=24  traits=11  boff_ab=12  ·  3 unmatched
```

---

## 4. WARP CORE — interface overview

The **WARP CORE — Trainer** tab in the launcher has three panels.

```
+------------------+------------------------------+----------------------+
|   LEFT PANEL     |       CENTER PANEL           |    RIGHT PANEL       |
|                  |                              |                      |
|  Screenshots     |   [canvas / screenshot]      |  Recognition Review  |
|  ----------      |                              |  ----------          |
|  screen1.png  [ok]|  (zoom with Ctrl+wheel)     |  Slot: Fore Weapon 1 |
|  screen2.png  [?] |  (bboxes drawn on items)    |  Item: Phaser Array  |
|  screen3.png  [ ] |                              |  Conf: 94%  [green]  |
|                  |                              |                      |
|  [progress bar]  |  +------------------------+  |  Slot: Console Sci 1 |
|  3/6 confirmed   |  |  Slot:  [combo box  ]  |  |  Item: ???  [red]    |
|                  |  |  Item:  [name field ]  |  |  Conf: 31%           |
|                  |  |  [  Accept (Enter)  ]  |  |                      |
|                  |  +------------------------+  |  [+ Add BBox] [- Rm] |
|                  |                              |  [x] Auto >= [0.75]  |
|                  |                              |  [  Accept (Enter) ] |
+------------------+------------------------------+----------------------+
```

### Toolbar

| Button | Action |
|--------|--------|
| **Detect Screen Types** | Classifies every screenshot in the folder using the MobileNetV3-Small screen classifier (Equipment / Traits / Boffs / Specializations / Mixed). Runs automatically when you open a folder — use the button to re-run it manually if you rename or replace files. Files you have already confirmed with a checkmark are skipped. |
| **Auto-Detect Slots** | Re-runs the full recognition pipeline on the **currently selected screenshot**. Items you have already confirmed are preserved and used as seeds for icon matching — only unconfirmed slots are re-processed. Use this after correcting a few items to let WARP retry the remaining ones with better context. |

---

### Left panel — Screenshots list

Lists every screenshot file in the folder. Each entry shows the detected screen type (e.g. *Space Equipment*, *Traits*, *Boffs*) and a **checkbox**:

- **Checked (✓)** — screen type confirmed by you; Detect Screen Types will not overwrite it
- **Unchecked** — screen type was auto-detected and is still tentative; may be updated on re-detect

**How to confirm a screen type:**
- **Tick the checkbox manually** next to the filename — confirms whatever type is currently shown.
- **Change the type via the Screen Type dropdown** (top of the center panel) — the correct type is set and the checkbox is ticked automatically.

Auto-detected types start unchecked. If the classifier guesses wrong, change the type in the dropdown and it will be confirmed immediately. Un-ticking a checkbox removes the manual override so the classifier can re-classify that file next time.

Click a filename to load it into the canvas.

#### Screenshot colour coding

| Colour | Meaning |
|--------|---------|
| White | No annotations yet |
| Light blue | Has annotations — in progress |
| Green | Marked Done — fully annotated and locked |

#### Marking a screenshot as Done

When you have finished annotating all items on a screenshot, click **✓ Mark Done** (below the progress bar) or press **Alt+D**. This:

- Locks the screenshot — no new bounding boxes can be added
- Saves the confirmed slot layout to the local layout database (used to improve auto-detection on similar screenshots)
- Colours the entry green in the list

If you need to make changes, click **↩ Back to Edit** (same button) or press **Alt+D** again to unlock it.

### Center panel — Canvas

Displays the current screenshot with coloured bounding boxes drawn over each detected item slot:

| Box colour | State | Meaning |
|------------|-------|---------|
| Red          | `pending`         | Detected but not yet reviewed — needs your attention |
| Green        | `confirmed (user)`| Accepted by you (Enter / autocomplete pick / Accept button) |
| Yellow/gold  | `confirmed (auto)`| Auto-accepted by the program because confidence ≥ Auto threshold (default 0.75). Persists across restarts so you can tell at a glance what *you* confirmed vs what the program did. Editing the name re-flags it as user-confirmed (green). |
| Orange       | `community conflict` | You previously confirmed this slot as one item, but the current community model now proposes a different name. The bbox waits for you to re-verify instead of being silently overwritten. See [Community conflicts](#community-conflicts) in section 6. |
| Cyan         | `text slot`       | Ship Name / Ship Type / Ship Tier — read by OCR, no icon matching, no confidence score |
| Grey (empty name) | `pending, no match` | The grid found this slot but the icon matcher had low confidence (< 0.35) or the match name had the wrong type for the slot. The bbox is kept so you can correct it manually — type the right name and Accept. |
| Gold crosshair | (drawing)       | While Alt+LMB drag is in progress — the bbox you're currently drawing |

Diagram:

```
   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
   │░░░░░░░░░░░░░░│   │██████████████│   │▓▓▓▓▓▓▓▓▓▓▓▓▓▓│
   │░░ pending  ░░│   │██  user OK ██│   │▓▓ auto OK  ▓▓│
   │░░░ (red)  ░░░│   │██ (green)  ██│   │▓▓ (yellow) ▓▓│
   └──────────────┘   └──────────────┘   └──────────────┘

   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
   │░░░░░░░░░░░░░░│   │┄┄┄┄┄┄┄┄┄┄┄┄┄┄│   │▒▒▒▒▒▒▒▒▒▒▒▒▒▒│
   │░ text slot ░│   │  no match    │   │▒▒ conflict ▒▒│
   │░░░ (cyan) ░░│   │  (grey)      │   │▒▒ (orange) ▒▒│
   └──────────────┘   └┄┄┄┄┄┄┄┄┄┄┄┄┄┄┘   └──────────────┘
```

#### Ship Name / Ship Type / Ship Tier bboxes

Cyan bboxes are special — they are not matched against the item database. Instead, the text inside them is read by OCR and used to identify the ship and its tier. They behave differently from equipment bboxes:

- **No confidence score** — there is no "correct/incorrect" percentage; the OCR result is shown as the item name.
- **No autocomplete** — type the text as it appears in the screenshot if you need to correct it.
- **No duplicate warning** — Ship Type and Ship Tier bboxes are allowed to overlap each other (see below).

#### Ship Type and Ship Tier overlap

In STO screenshots the ship type label (e.g. *"Fleet Temporal Science Vessel"*) and the tier label (e.g. *"T6-X"*) appear close together and sometimes on adjacent lines.

When drawing these bboxes manually:

- **Ship Type** — draw the bbox over the **full ship type text**, even if it spans two lines. A two-line Ship Type bbox will physically overlap with the Ship Tier bbox. This overlap is **intentional and expected** — the duplicate warning does not trigger for cyan text slots.
- **Ship Tier** — draw a **separate, smaller bbox** that covers **only the tier token** (e.g. just the `T6-X` fragment). Do not include the ship type text in this bbox.

Example:

```
┌─────────────────────────────────────┐  ← Ship Type bbox (full 2 lines)
│ Fleet Temporal Science Vessel       │
│ T6-X                   ┌───────┐   │  ← Ship Tier bbox (tier token only)
└────────────────────────┤  T6-X ├───┘
                         └───────┘
```

If both bboxes are confirmed correctly, WARP will extract the ship class from Ship Type and the upgrade tier from Ship Tier independently.

| Action | How |
|--------|-----|
| Zoom in / out | **Ctrl + scroll wheel** (1× – 6×, anchored to cursor) |
| Select a box | **Left click** on the box — highlights it in the review list |
| Draw new box | **Alt + LMB drag** — hold Alt, click and drag over an item icon |
| Draw mode toggle | **Alt+A** button in the right panel — cursor stays as crosshair until toggled off |

Below the canvas is the **Annotate panel**:
- **Slot** — dropdown to select the slot type for the current box
- **Item** — text field with autocomplete; type the item name
- **Accept** — confirms the item (also triggered by Enter)

### Right panel — Recognition Review

Lists all items detected in the current screenshot, one row per slot. Each row shows:
- Slot name (e.g. "Fore Weapon 3")
- Recognised item name (or "???" if not matched)
- Confidence percentage, colour-coded:
  - **Green** ≥ 75% — confident match
  - **Yellow** 40–74% — uncertain, review recommended
  - **Red** < 40% — poor match, manual correction needed

At the bottom:
- **Add BBox** — enter draw mode to add a missing box (Alt+A)
- **Remove** — delete the selected box (Alt+R or Del)
- **Auto ≥ [threshold]** — automatically accept items above the threshold; adjust the spinner to change it (default 0.75)

<!-- screenshot: WARP CORE window with a loaded screenshot and mixed confirmed/pending items -->

---

## 5. Reviewing and correcting recognition

### Step-by-step user journey

The intended WARP CORE workflow, end to end:

```
   START: open WARP CORE → pick a folder of screenshots
        │
        ▼
   ┌─────────────────────────────────────────────────┐
   │ A. SCREEN TYPES                                 │
   │    On folder open, every file is classified     │
   │    (SPACE_EQ / GROUND_EQ / TRAITS / BOFFS /     │
   │     SPECIALIZATIONS / SPACE_MIXED / …).         │
   │    For each file: check the type next to the    │
   │    filename. If correct → tick the checkbox.    │
   │    If wrong → change it via the Screen Type     │
   │    dropdown above the canvas (auto-ticks).      │
   └─────────────────────────────────────────────────┘
        │
        ▼
   ┌─────────────────────────────────────────────────┐
   │ B. AUTO-DETECT SLOTS                            │
   │    Select a screenshot in the left panel.       │
   │    Click "Auto-Detect Slots" (toolbar) to run   │
   │    the recognition pipeline on it.              │
   │    Items appear in the right panel's review     │
   │    list, sorted by confidence (lowest first).   │
   │    Items above the Auto-accept threshold are    │
   │    auto-confirmed (yellow) immediately.         │
   └─────────────────────────────────────────────────┘
        │
        ▼
   ┌─────────────────────────────────────────────────┐
   │ C. REVIEW & CORRECT                             │
   │    For each remaining pending (red) item:       │
   │      • Click it → canvas highlights the box.    │
   │      • Correct name? Press Enter / click Accept │
   │        → box turns green.                       │
   │      • Wrong name? Type the right one (autocom- │
   │        plete) and Accept.                       │
   │      • Box in wrong place? Del to remove, then  │
   │        Alt+drag a new one on the right icon.    │
   │    For grey (empty-name) bboxes: type the name. │
   │    For missing slots: Alt+drag directly on the  │
   │    canvas to add a new bbox.                    │
   └─────────────────────────────────────────────────┘
        │
        ▼
   ┌─────────────────────────────────────────────────┐
   │ D. MARK DONE                                    │
   │    When the screenshot is fully reviewed, click │
   │    "✓ Mark Done" (or Alt+D).                    │
   │    Locks the screenshot, saves its layout to    │
   │    anchors.json (improves future detection on   │
   │    similar screens), colours the entry green    │
   │    in the file list.                            │
   └─────────────────────────────────────────────────┘
        │
        ▼
   Repeat B–D for the next screenshot in the folder.
        │
        ▼
   END: confirmed crops upload automatically to HF
        every 10 min in the background.
```

### Typical micro-loop (per item)

1. Pick the lowest-confidence red item in the review list (sorted automatically).
2. Compare what's in the Item field against the icon on the canvas.
3. **Correct** → **Enter**. **Wrong** → type the correct name, pick from autocomplete (instant accept, no Enter needed).
4. **Bbox in wrong position** → **Del** to remove, **Alt+drag** to redraw.
5. Move to the next item.

> **Tip:** Auto ≥ 0.75 already handles the easy cases. In practice, you usually only need to review red items (< 40%) and the occasional yellow-confidence item.

### Correcting item names

The Item field has **autocomplete** — start typing the item name and a dropdown shows matches from the STO item database. Select with arrow keys or mouse.

If the correct item is not in the autocomplete list, type the full name manually. It will still be saved as training data for that icon.

> **Tip:** You don't have to correct every item. Focus on red (< 40%) and yellow items. Green items (≥ 75%) are usually correct — Auto-accept handles these automatically if enabled.

### Adding a missing bounding box

If WARP missed a slot entirely (no box drawn over an item):

1. Hold **Alt** and drag directly on the canvas — or click **Add BBox** / press **Alt+A** to lock draw mode on.
2. The cursor changes to a gold crosshair.
3. Drag over the item icon to draw a box.
4. The box is recognised immediately. Correct the name if needed, then Accept.

### Annotating empty and inactive Bridge Officer slots

Bridge officer seats sometimes contain slots that are visually empty (nothing assigned yet) or inactive (locked/unavailable at this rank). Annotating these helps WARP correctly position recognised abilities even when some slots are unoccupied.

**How to annotate:**

1. Draw a bounding box over the empty or inactive slot icon as you would for a normal ability.
2. In the **Item** field, type `__empty__` (for an unoccupied slot) or `__inactive__` (for a locked/unavailable slot).
3. Press **Enter** to confirm.

The review list shows these with grey labels `[empty slot]` / `[inactive slot]`. They write nothing to your SETS build but are uploaded to the community dataset so the model learns to recognise these slot states.

> **Why bother?** WARP uses the position of all detected icons — including empty and inactive ones — to determine which ability belongs in which seat slot. Annotating gaps makes the positioning more accurate, especially for Commander seats (4 slots) where only 1–2 abilities are present.

### Removing a wrong bounding box

If a box covers the wrong area or a non-slot area:

1. Click the box on the canvas (or the item in the review list) to select it.
2. Press **Del** or **Backspace**, or click **Remove**.

<!-- screenshot: WARP CORE canvas with a red item selected and the Item field showing the correct name typed -->

---

## 6. Confirming items and accepting results

### Manual accept

With an item selected:
- Press **Enter**, or
- Click **Accept** in the right panel or bottom panel.

A confirmed item turns green in the review list and on the canvas.

### Auto-accept

Enable the **Auto ≥** checkbox to automatically accept items above the confidence threshold without pressing Enter for each one. The threshold spinner sets the cutoff (default 0.75 = 75%).

Auto-accept applies:
- When a screenshot is loaded (immediately marks high-confidence items)
- After drawing a new bounding box
- After running Auto-Detect

The checkbox state and threshold value are saved across sessions (per user).

### Selecting from the autocomplete dropdown

Choosing an item from the autocomplete dropdown confirms it immediately — no Enter needed. This is the fastest way to correct a wrong item: type a few letters, pick from the list, done.

### Duplicate warning

If you confirm an item into a slot that already has a confirmed item at the same position (>70% overlap), WARP CORE shows a warning. This prevents accidentally confirming the same physical slot twice.

### Community conflicts

The community model occasionally disagrees with an item you have
already confirmed locally — usually because a hard-to-tell pair of
icons looks almost identical to the embedder, or because the community
vote count for that crop tilts the other way.

When this happens, the affected slot is **not** silently overwritten.
It appears in the review panel as:

```
⚠ Tactical Consoles  ->  [CONFLICT] disk: Covert Warhead Module | community: Crystalline Absorption Matrix  [100%]
```

with an **orange** bounding box on the canvas and an orange row label.
Auto-accept skips conflicted rows even at 100 % confidence, so the
program never decides this one on its own.

**What to do:**

1. Open the screenshot in WARP CORE and look at the icon.
2. If the community proposal is correct, pick that name and **Accept**
   — the slot turns green and the community gets one extra vote in
   that direction.
3. If your previous confirmation was correct, leave it as is and
   **Accept** — the slot turns green and WARP CORE remembers that you
   already rejected this specific community proposal for this bbox.

After step 3, the next Auto-Detect will silently keep your name and
*not* flag the conflict again, as long as the community keeps
proposing the same (rejected) name. If the community later changes
its mind to a *different* name, a fresh conflict appears so you can
re-verify against the new proposal.

---

## 7. Detection logs / System logs tabs

The launcher's last two tabs surface what is happening under the hood.

### Detection logs

Live tail of the current recognition run — OCR results, classifier picks,
layout-detector strategy choices, per-slot match scores. The view auto-scrolls
to the newest line (vertical), but **horizontal scroll position is preserved**
so a long line doesn't bounce the view sideways every time it appears.

A fresh **Open Screenshot / Open Folder** wipes the live view automatically
(but not the underlying log file) so each run starts on a clean slate. The
`Source:` combo at the top switches between the current session and the
previous session loaded from `warp_detection.log.bak`. **Open folder** opens
the log directory in the system file manager.

### System logs

Background activity: asset sync (cargo / ship DB downloads), model updates,
knowledge cache, desktop entry installation, sync coordinator. Kept on a
separate channel so detection noise stays focused.

Both views share the same controls — `Auto-scroll`, `Clear view`, `Reload`,
and `Open folder`. On-disk log files (rotated):

| File | Channel | Path |
|------|---------|------|
| `warp_detection.log` | Detection | `~/.config/warp/` |
| `warp_system.log` | System | `~/.config/warp/` |

---

## 8. Community model — how it works

### Architecture

WARP uses a central **EfficientNet-B0** icon classifier trained on crops contributed by all users.
There is no local training — the model is trained once per hour by the community pipeline and
automatically downloaded to your installation.

### Your contribution

Every time you confirm an item annotation in WARP CORE, the icon crop is queued for upload to
HuggingFace (`sets-sto/sto-icon-dataset`). The upload happens automatically in the background
**every 10 minutes** (first run 15 s after app start) when a HuggingFace token is configured.

**What is sent:** Only the icon crop image (small PNG, ~64×64 px) and its label (item name +
slot type). No screenshots, no ship names, no personal data.

### Model update

A new model is trained hourly on GitHub Actions using all community crops. Your installation
checks for updates **every 15 minutes** (rate-limit cache; uses `requests` with 5 s connect /
60 s read timeouts to survive Render free-tier cold-starts) and downloads the new
`icon_classifier.pt` automatically if a newer version is available.

| File | Source | Update interval |
|------|--------|----------------|
| `warp/models/icon_classifier.pt` (EfficientNet-B0 softmax) | `sets-sto/warp-knowledge` (HF) | 15 min check |
| `warp/models/icon_embedder.pt` (ArcFace embedder) | `sets-sto/warp-knowledge` (HF) | 15 min check |
| `warp/models/screen_classifier.pt` (MobileNetV3-Small) | `sets-sto/warp-knowledge` (HF) | 15 min check |
| `warp/models/community_anchors.json` (learned layouts) | `sets-sto/warp-knowledge` (HF) | 15 min check |
| `warp/models/ship_type_corrections.json` (OCR correction map) | `sets-sto/warp-knowledge` (HF) | 15 min check |

---

## 9. Community sync details

### What is sent to HuggingFace

- The icon crop image (small PNG, ~64×64 px) — just the item icon, cropped from your screenshot
- The item name and slot type you confirmed
- An anonymous installation ID (random UUID, generated at first launch, stored locally)

No username, account information, full screenshots, or ship names are ever transmitted.

### Privacy note for Ship Name bbox

If you draw a bbox over the ship name text in WARP CORE, the **position** (coordinates) is saved
locally in `annotations.json` for layout learning. The **text content** is never saved as a crop
and never uploaded — ship names are treated as personal data.

---

## 10. Keyboard shortcuts

### WARP CORE

| Shortcut | Action |
|----------|--------|
| **Enter** | Accept current item |
| **Del** / **Backspace** | Remove selected bounding box |
| **Alt + A** | Toggle Add BBox draw mode |
| **Alt + D** | Toggle Mark Done / Back to Edit |
| **Alt + R** | Remove selected bounding box |
| **Alt + LMB drag** | Draw new bounding box directly |
| **Ctrl + Wheel** | Zoom canvas in/out (1× – 6×, anchored to cursor) |

### Bounding box colours (canonical)

| Colour | Meaning |
|--------|---------|
| Red | Pending — awaiting your review |
| Green | User-confirmed (Enter / autocomplete pick / Accept) |
| Yellow / gold | Auto-confirmed by program (conf ≥ Auto threshold). Persists across restarts. |
| Orange | Community conflict — community model disagrees with your earlier confirmation. Re-verify and Accept. |
| Cyan | Text slot (Ship Name / Type / Tier) — OCR, no icon matching |
| Grey (empty name) | Detected bbox without a usable match — type the name and Accept |
| Gold crosshair | Currently being drawn (Alt + LMB drag in progress) |

See section 4 for the full colour legend with diagram.

---

## 11. Tips and troubleshooting

### WARP didn't detect my ship

- Make sure at least one **Space Equipment** screenshot is included (ship name and type are read from the equipment screen).
- The ship name must be visible at the top of the screen in the screenshot.
- If OCR fails, WARP falls back to a keyword-based match using the slots it finds. Check the log for the detected ship name.

### Wrong ship was recognised

The ship banner above the Results tree shows what WARP picked up from OCR. If
it's wrong, the underlying OCR tokens are in the Detection logs tab — useful
when the name is a near-miss (e.g. *"Legondary Bortasqu'"* instead of
*"Legendary Bortasqu'"*). Open the screenshot in WARP CORE and confirm a
**Ship Name / Ship Type / Ship Tier** bbox manually; on the next recognition
run the confirmed text wins.

### Recognition accuracy is low on first use

On a fresh install, WARP uses the community model. If you have unusual items, high-resolution screenshots, or a non-standard UI scale, accuracy may be lower initially. Run a few imports and confirm the results in WARP CORE — your confirmed crops are uploaded to the community dataset, and the model improves as more users contribute.

### "???" items after recognition

Items shown as "???" have a confidence below the minimum threshold (40%). They were not matched to any known item. To fix:
1. Switch to the **WARP CORE — Trainer** tab.
2. Find the ??? item in the review list (shown in red).
3. Type the correct item name in the Item field.
4. Accept.
5. Repeat for all ??? items — confirmed crops will be uploaded automatically and help improve future recognition.

### Recognition is slow

The first run after a fresh install may take 30–60 seconds because EasyOCR initialises its language model. Subsequent runs are faster (model stays in memory while sto-warp is open).

On CPU-only hardware, the ML inference step adds 2–5 seconds per screenshot. This is normal.

### "Duplicate bbox" warning

This appears when two confirmed items overlap by more than 70% in the same screenshot. Usually this means you accidentally drew a bbox over an area that already has a confirmed item. Remove the duplicate using Del and re-confirm if needed.

### Training completes but accuracy doesn't improve

- Make sure you have confirmed items from multiple different screenshots, not just one.
- If you only have a handful of unique items, accuracy metrics may fluctuate — this is normal with small datasets.
- More data always helps. Confirm items from 5–10 screenshots before training for the best results.

