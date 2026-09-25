# Skill-tree recognition

## Purpose

Reads a captain's **Space** or **Ground** skill tree from a screenshot and
turns it into the `space_skills` / `ground_skills` fields of a SETS build.
It is owned by `warp/recognition/skill_grid.py`. WARP folds the result into
**Export to SETS JSON**, and WARP CORE displays it read-only.

Skill trees are a different problem from equipment and traits. Every
player's tree has the **same fixed layout**. Only each node's state differs:
**ON** (trained) or **OFF** (untrained, or locked behind a padlock). Icons
repeat from one rank to the next, so an icon cannot say which node it is.
Identity therefore comes from **position**, and recognition comes down to
two questions: where is the grid, and is each node lit?

## Context

```
screenshot ──► screen classifier ──► SKILLS / SPACE_SKILLS / GROUND_SKILLS
                  (trained model)                 │
                                                  ▼
                               skill_grid  (deterministic, no model)
                               1. anchor the fixed template
                               2. read ON/OFF per node
                                                  │
                    ┌─────────────────────────────┴──────────────┐
                    ▼                                            ▼
      WARP: Preview boxes, summary line,           WARP CORE: canvas boxes,
      Export to SETS JSON                          review summary, read-only
```

The screen classifier is the only part of this that learns. It learns
*which kind of screen* this is, from the types users confirm (see
`ML_PIPELINE.md`). Everything after it has fixed thresholds.

The recognition pipeline in `warp_importer.py` skips skill screens: they
produce no `RecognisedItem`s (`_SKIP_STYPES`). Skill results reach the user
through the three paths listed under Components.

## Step by step

### 1. Which tree: space or ground

The screen type decides. `SPACE_SKILLS` means space and `GROUND_SKILLS`
means ground. A generic `SKILLS` screen is resolved by `env_of` from the
shape of the node grid. The space tree is tall (width / height about 0.62)
and the ground tree is wide (about 1.4), so a ratio below 1.0 means space.
If no grid is found, the environment is unknown and nothing is read.

### 2. Anchor the template

`skill_template.json` holds every node's centre, normalised 0..1 to the
bounding box of the node centres: 90 positions for space (30 each for eng,
sci, tac) and four trees of 6, 6, 4, 4 for ground. The template was
calibrated once, by clicking the node centres on full captures.

`_node_extent` finds that bounding box on the screenshot in three steps:

1. `_tile_centres` looks for square, lit blobs on the near-black
   background. It keeps blobs 12–70 px on a side with an aspect of
   0.55–1.45. It needs only the outermost nodes, so a missed node in the
   middle does not matter.
2. With fewer than 12 tiles, the grid is not found and nothing is read.
3. `_robust_bound` trims a lone tile that sits well outside the grid (more
   than 1.4× the typical tile spacing away). It was added because one
   ground capture had a 21st, stray tile that pulled the left edge out and
   shifted every box.

Scaling the template onto that box gives every node's centre in pixels.
Anchoring by extent rather than by absolute pixels is what makes margins
and resolution irrelevant, provided the whole grid is in the picture.

### 3. Read each node

`_is_on` samples a 26×26 px tile around each centre and counts **vivid**
pixels: saturation > 0.5 and brightness > 0.45. The node is ON when more
than 5% of the tile is vivid.

A lit node shows its icon in full career colour. A greyed or padlocked node
has almost no vivid pixels, so the two groups are far apart. Saturation
alone was tried first and rejected: grey icons still carry some tint, and
they fell into a band (0.2–0.4) that overlapped the lit ones.

### 4. Output in SETS order

| Field | Shape | Index |
|---|---|---|
| `space_skills` | `{'eng': [30], 'sci': [30], 'tac': [30]}` of bool | `i = rank*6 + sub*3 + node` — rank 0..4 top to bottom, sub 0 = left / 1 = right column, node 0..2 |
| `ground_skills` | `[[6], [6], [4], [4]]` of bool | trees top-left, top-right, bottom-left, bottom-right; node order follows SETS' ground layout |

The full grid is always written, OFF nodes included. SETS maps by array
index, not by name, and does not check lengths, so a short list would
silently shift every node after it. The export contract is in
`SETS_FORMAT_CONTRACT.md`.

## Components

| Construct | Role | Used by |
|---|---|---|
| `detect_space`, `detect_ground`, `detect` | node states in SETS shape | `skills_from_files` |
| `detect_boxes` | `[(x, y, w, h, on), ...]` for a canvas | WARP `ResultsView._set_skill_overlay`, WARP CORE `WarpCoreWindow._show_skill_recognition` |
| `on_counts`, `group_sizes` | ON count and node count per career / tree, in `detect_boxes` order | WARP CORE summary |
| `env_of` | space vs ground from grid shape | both GUIs, for generic `SKILLS` |
| `skills_from_files`, `skill_env_counts` | a folder's skill screens → build fields, and the duplicate count | WARP export and summary line |
| `to_skill_tree` | a standalone skill-tree file for SETS | not wired into a GUI |

## In WARP

- **Preview** draws each node's box: green for ON, red for OFF. As with
  every screen in WARP, a changed screen type takes effect after **Rerun
  Recognition**.
- The **summary line** after a run lists ON counts per career (space) and
  per tree (ground).
- **Export to SETS JSON** adds `space_skills` / `ground_skills` to the
  build. A folder is one build, so only the first space and first ground
  screen are used. When there are more, the summary line shows a warning
  badge.

## In WARP CORE — display only

When a skill screen is open, `WarpCoreWindow._show_skill_recognition` draws
the same green/red boxes on the canvas. It also puts the counts in the
review summary, for example `Space skills ON — Eng 10/30 · Sci 9/30 ·
Tac 27/30`. For a generic `SKILLS` screen, the summary also says which
tree the grid shape pointed to. The display is refreshed at three moments,
because each of them rewrites the summary: when the screenshot opens, when
its type changes, and after Auto-Detect. A type change shows immediately,
since in WARP CORE picking a type confirms it.

When the grid cannot be found, the summary says so and asks for the type to
be set to Space Skills or Ground Skills, or changed. Neither the canvas nor
the summary is left blank.

**No editing, by design.** Node states come from a fixed threshold, so no
model learns from them, and WARP CORE does not export builds. A corrected
node state would therefore have no consumer. The review fields stay
disabled on skill screens (`_update_screen_type_ui`, `_NO_BBOX_TYPES`).

**Auto Mark Done.** Once a *user* confirms a skill screen's type, by picking
it from the type menu or ticking the file-list checkbox,
`_auto_mark_done_skill` marks the screenshot Done and locks it. The type is
the only thing on a skill screen left to review. A type the classifier
only guessed is **not** marked Done, because that type is the training
signal for the screen classifier and still needs a human look. **Back to
Edit** reopens the screenshot as for any other screen.

## Failure modes

| What you see | Cause | Where to look |
|---|---|---|
| WARP CORE: "no skill node grid found" | fewer than 12 tiles found: a cropped, scrolled or heavily scaled capture | `_node_extent`; the detection log line `WarpCore: no skill node grid found` (in the launcher, WARP CORE's tab writes to `warp_detection_core.log`) |
| Every box shifted by the same amount | a stray tile stretched the extent more than `_robust_bound` trims | `_tile_centres` output for that image |
| Generic `SKILLS` read as the wrong tree | grid shape distorted by the same cause | `env_of`; set the type explicitly |
| All nodes OFF on a trained tree | anchoring failed in `detect_space` / `detect_ground` | log `SkillGrid: could not anchor ... template` |

## Not covered

- **Milestone choices** (`skill_unlocks`, the ⇑/⇓ picks at rank
  milestones) are exported as `None`. SETS loads the file but does not
  restore those choices.
- **Partial captures.** The template needs the whole grid on screen. The
  `extent=` parameter of the detect functions accepts a known box, but no
  GUI supplies one.
