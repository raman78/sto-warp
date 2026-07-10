# Fast Correction Mode — Technical Reference

> User-facing description: [`WARP_GUIDE.md` §6.5](WARP_GUIDE.md#65-fast-correction-mode).
> This document covers staging, lifecycle, the snapshot/restore mechanism, and
> the rules that keep Fast Correction Mode isolated from the standing
> training data.

Fast Correction Mode (FCM) is an *ephemeral* WARP CORE session that lets
the user polish a batch of recognition results before exporting SETS JSON,
without overwriting the long-term training data those screenshots may
already be referenced from. It exists because the community model improves
asynchronously: when a build needs to ship today and two slots came out
wrong, the heavy "open every screenshot in WARP CORE, correct, mark Done,
upload" loop is the wrong shape — FCM is the lightweight loop.

Introduced in **1.0.11**. End-to-end loop closure (auto-exit, view rewind)
landed in the same release; the Send-to-WARP visibility fix landed in
**1.0.13**. Bbox dedup and FC-removal disk-propagation fixes landed in
**1.0.18** (see §5½).

---

## 1. Lifecycle

```
   WARP recognition run finishes
            │
            ▼
   user right-clicks any Results row → "Open in WARP Fast Correction Mode"
            │
            ▼
   ┌─ Trainer: set_fast_correction_mode(files, items_by_file, …) ─────┐
   │                                                                  │
   │  1. fast_session.prepare(orig_paths) — stage files                │
   │     ~/.cache/warp/fast_correction/<12-hex>/                       │
   │         session.json                                              │
   │         fc_<hash>__<orig_basename>.png   (one per input)          │
   │                                                                   │
   │  2. Snapshot the training-mode view into _pre_fc_snapshot         │
   │     (only on FIRST entry — re-entry keeps the original snapshot)  │
   │                                                                   │
   │  3. Reset _screenshots / _screen_types / _recognition_cache to    │
   │     the staged batch                                              │
   │                                                                   │
   │  4. Seed _recognition_cache[staged.name] from WARP's items_by_file│
   │     (basename match — resolves symlink / trailing-slash mismatch) │
   │                                                                   │
   │  5. Theme: amber accent + banner; tab title becomes               │
   │     "WARP CORE — Fast Correction"; hide Open Folder / Open File   │
   └──────────────────────────────────────────────────────────────────┘
            │
            ▼
   user reviews / corrects (normal WARP CORE loop — same hotkeys,
   colours, Mark Done semantics)
            │
            ▼
   user clicks "↗ Send this to WARP"
            │
            ▼
   ┌─ Trainer: _on_send_to_warp() ───────────────────────────────────┐
   │  1. Re-run SETS pipeline on the staged batch with confirmed     │
   │     items as ground truth                                       │
   │  2. send_to_warp.emit(result)                                   │
   │  3. exit_fast_correction_mode()                                 │
   └─────────────────────────────────────────────────────────────────┘
            │
            ▼
   ┌─ Launcher: _on_send_to_warp(result) ────────────────────────────┐
   │  1. Install result into the WARP tab as if recognition had just │
   │     finished there                                              │
   │  2. Switch focus to the WARP tab                                │
   └─────────────────────────────────────────────────────────────────┘
            │
            ▼
   ┌─ Trainer: exit_fast_correction_mode() restores _pre_fc_snapshot │
   │  (folder, selection, screen types, done set, filter text) and   │
   │  emits fast_correction_exited                                   │
   └─────────────────────────────────────────────────────────────────┘
            │
            ▼
   user clicks "Export to SETS JSON…" in WARP
```

---

## 2. Staging — `warp/trainer/fast_session.py`

Each batch is keyed by `sha256(sorted basenames)[:12]`. The choice is
deliberate:

| Decision | Rationale |
|---|---|
| Basename-only, not full path | Survives the user moving the screenshot folder. Same set of `space.png + traits.png` keeps the same session even after a move/copy. |
| Sorted before hashing | Input order from WARP is not stable; sorting normalises identity so re-entering Fast Mode resumes instead of forking. |
| 12 hex characters | Long enough to avoid collisions across a single user's history, short enough to stay readable in log lines. |
| Sha256 (not md5) | No external constraint forces md5, sha256 is on every Python install, future-proofs the namespace. |

### Filesystem layout

```
~/.cache/warp/fast_correction/
└── <12-hex-hash>/
    ├── session.json                       # last_used_at, hash, orig basename → path map
    ├── fc_<hash>__<orig>.png              # snapshot copy (not hardlink)
    ├── annotations.json                   # written by TDM under staged basenames
    └── crops/<sha256>.png                 # crops confirmed within the FC session
```

`session.json` carries `last_used_at` (UNIX timestamp) and the
basename→original-path map. Anything else can be reconstructed from disk.

### Why copy, not hardlink

`shutil.copy2`, not `os.link`. Hardlinking would be faster and cheaper, but
fails on cross-filesystem layouts (e.g. screenshots on an external drive,
cache on `$HOME`) and on Windows when the source is on a different volume.
The copy is a single-file operation per screenshot and dominated by I/O
the user has already paid in WARP, so the cost is negligible against the
correctness gain.

### Garbage collection

`fast_session.gc_old_sessions(max_age_days=14)` is called by the launcher
at startup. It walks every subdir of the staging root and removes any
whose `session.json.last_used_at` is older than the cutoff. Sessions with
missing or unreadable metadata are also removed — preferring a small risk
of GC'ing a corrupt session to leaking disk forever.

The default 14-day window matches the user-visible promise in
[`WARP_GUIDE.md` §6.5 "Cleanup"](WARP_GUIDE.md#65-fast-correction-mode).
Bumping it requires updating both this number and the user-facing doc.

---

## 3. Isolation rules

| Resource | Behaviour in Fast Correction Mode |
|---|---|
| `annotations.json` for staged files | Written under `~/.cache/warp/fast_correction/<hash>/` keyed by **staged** basename (`fc_<hash>__<orig>.png`), never under the original basename. |
| `annotations.json` for *original* files in `~/.local/share/warp/training_data/` | **Never touched.** TDM resolves the staged basename to its own staging dir — original file's permanent annotations are not loaded, not modified, not deleted. |
| Crops in the community upload queue | Confirmed crops still flow into `SyncWorker`'s upload path, so the community model still benefits from the corrections. The staged dir contributes the same way a normal training dir does — the staging hash never leaves the user's disk. |
| Anchors / layout DB | Each **Mark Done** inside FC writes the confirmed layout to `community_anchors`-equivalent files so future Auto-Detect on similar layouts improves. Layout learning is not isolated; it does not carry identifying information about *which* build was used. |
| Local model (`icon_classifier.pt`) | Never written. Production model files are downloaded only, never overwritten by FC. |
| `DISCARD` screenshots | Auto-marked Done on entry (both ML-detected and manual override). `RecognitionWorker` emits `[]` — no recognition pipeline runs. Discarded files contribute no items to `_on_send_to_warp()` and are excluded from the exported result. |

The "what gets saved" matrix in [`WARP_GUIDE.md` §6.5](WARP_GUIDE.md#65-fast-correction-mode)
is the user-readable version of this table; this table is the source of
truth, the user doc the projection.

---

## 4. The snapshot/restore mechanism

When the trainer enters Fast Correction Mode for the first time it
captures a `_pre_fc_snapshot` dict containing:

- `screenshots` — the original folder file list
- `screen_types`, `screen_types_manual`, `screen_types_ml_auto` — the
  user's screen-type confirmation state
- `recognition_cache`, `recognition_items` — pending Auto-Detect results
  the user was mid-review
- `screenshots_done` — the set of Mark-Done filenames
- `current_idx` — which file was selected
- `file_filter` — the **Filter by filename** box content

`exit_fast_correction_mode()` restores every field, re-populates the file
list widget, restores the filter, re-selects the previously active file
and emits `fast_correction_exited`. The launcher picks the signal up,
flips back to the trainer tab, and clears the warm-amber theme.

**Re-entry guard.** If the user opens a second Fast Correction batch
while the first is still active, `set_fast_correction_mode` does **not**
re-snapshot. Without that guard, the second FC entry would snapshot the
first FC's *staged* batch, and exit could never escape Fast Mode — the
"previous" state to restore would itself be ephemeral. With the guard,
the first snapshot stays valid all the way back to real training mode.

---

## 5. Themed UI cues

The tab title becomes `WARP CORE — Fast Correction`. A warm-amber accent
is applied via `apply_fast_correction_style()` in `warp/themes.py`. The
**Open Screenshot…** and **Open Folder…** toolbar entries are hidden
because Fast Mode's input set is fixed by what WARP handed over —
loading anything else would invalidate the staging hash.

The **↗ Send this to WARP** button is only visible in Fast Correction
Mode. The 1.0.13 fix in commit `fa75388` hides it during normal training
to remove a click target that was inert outside FC.

All generic review-panel UI features carry over unchanged into FC:

- **Hover tooltips with reference icon** on both the canvas
  (`annotation_widget.py:_show_hover_tooltip`) and the review tree
  (`trainer_window.py:_populate_review_item`) — HTML tooltip embeds a
  base64-encoded QImage loaded from `cargo.ref_icon_path(name)`.
- **Right-click → external links** on canvas bboxes
  (`annotation_widget.py:contextMenuEvent`) and on review-tree leaf
  items (`trainer_window.py:_show_item_link_menu`) — "Open on
  vger.stobuilds.com" (slot → category page via `cargo.vger_url`) and
  "Open on STO Wiki" (`cargo.wiki_url`). Uses
  `QDesktopServices.openUrl`.

These are not gated by mode — the same code paths execute in both
normal training and FC.

---

## 5½. Bbox dedup and removal — IoU matching (1.0.18)

Two problems existed prior to 1.0.18 around near-overlapping bounding boxes
(a 1–2 px shift between detector runs):

1. **Duplicate creation.** `TrainingDataManager.add_annotation()` step 2
   matched existing annotations by exact bbox tuple equality. A 1 px shift
   produced a different tuple → step 2 missed → a second annotation was
   appended for the same physical slot position.

2. **Stale disk on FC removal.** `_on_remove_item()` only deleted the disk
   annotation when `state == 'confirmed'`. In Fast Correction Mode every
   review item is seeded as `state='pending'`, so deleting a bbox removed
   it from memory but left the confirmed annotation on disk.
   `_on_send_to_warp()` reads from disk → the deleted item re-appeared.

### Fix 1 — IoU-based dedup in `add_annotation`

`_bbox_iou()` (`warp/trainer/training_data.py:82`) computes
Intersection-over-Union for two `(x, y, w, h)` tuples. Step 2 of
`add_annotation()` (line 329) now:

- Scans all existing annotations for the same image key.
- Exact bbox match → update in-place immediately (original behaviour).
- Same-slot annotation with IoU ≥ 0.5 → treat as the same physical
  position and update in-place.
- The IoU path is restricted to same-slot comparisons to avoid
  merging legitimately adjacent slots (e.g. `Fore Weapons` slot 1 and
  slot 2 on a tight grid).

### Fix 2 — disk removal for pending items

`WarpCoreWindow._on_remove_item()` (`warp/trainer/trainer_window.py:2371`)
now runs the disk-removal block unconditionally (not gated behind
`state == 'confirmed'`). It searches disk annotations with the same
IoU fallback (exact match first, then IoU ≥ 0.5) so a shifted bbox
is still found and removed. The confirmation dialog for confirmed
items is preserved — the user is still prompted before deleting a
confirmed annotation.

### IoU threshold

| Context | Threshold | Rationale |
|---|---|---|
| `add_annotation` step 2 dedup | 0.5 | Catches 1–3 px shifts (typical IoU 0.85–0.95) without merging adjacent slots on tight grids. |
| `_on_remove_item` disk lookup | 0.5 | Same reasoning — match the same physical position even when the detector grid drifted slightly. |
| `_merge_recognition` overlap filter | 0.3 | Broader — prevents any meaningful overlap between new detections and existing items. Unchanged. |

---

## 6. Failure modes and recovery

| Symptom | Cause | What happens |
|---|---|---|
| `fast_session.prepare` returns an empty `paths_map` | Every `shutil.copy2` failed (permission, disk full, source unreadable) | Trainer aborts FC entry, status bar shows "failed to stage any input file"; no state changed. |
| User closes the window mid-correction | Standing process killed | Staging dir survives on disk. Next FC entry with the same file set returns to the same hash → resumes annotations. After 14 days `gc_old_sessions` removes it. |
| User enters FC twice with different batches | Re-entry path | Second batch overwrites `_fast_session` but **keeps** `_pre_fc_snapshot`. Exit still drops back to the original training-mode state. |
| Send-to-WARP fires with no confirmed items | User clicked the button before correcting anything | The re-run of the SETS pipeline still uses WARP's original items (seeded into `_recognition_cache` at entry) — equivalent to closing FC and clicking Export from the original WARP results. Not destructive. |

---

## 7. What deliberately is **not** here

- **No retention period configuration.** 14 days is the only value, deliberately. Surfacing it as a setting tempts users to set "never delete", and abandoned staging dirs accumulate fast on iterative correction sessions.
- **No "save FC corrections to permanent training data" button.** If a user wants the corrections to persist, the right path is the normal *Open in WARP CORE* right-click — see [`WARP_GUIDE.md` §3 Results tab](WARP_GUIDE.md#results-tab). Adding a "promote" button would duplicate that path and blur the FC promise of isolation.
- **No staging hash in any UI string.** `fast_session.display_name()` strips the `fc_<hash>__` prefix before any path reaches the file list, dialog, or status bar. The hash is an internal addressing scheme; surfacing it would force the user to reason about it. Log lines are the only place it appears.
