# Review panel merge — confirmed rows vs fresh detection

## Purpose

WARP CORE's review panel never shows raw detector output. Every time a
screenshot is opened or Auto-Detect finishes, the detector's items are merged
with the annotations already confirmed for that image in `annotations.json`,
and the merged list becomes `_recognition_items` — the list the tree, the
canvas and the auto-accept pass all read.

This doc covers that merge only. Where the detector's items come from is in
[`EQ_DETECTION.md`](EQ_DETECTION.md), [`BOFF_DETECTION.md`](BOFF_DETECTION.md),
[`TRAIT_DETECTION.md`](TRAIT_DETECTION.md) and
[`SHIP_INFO_DETECTION.md`](SHIP_INFO_DETECTION.md); what happens to a
confirmation afterwards is in [`DATA_LIFECYCLE.md`](DATA_LIFECYCLE.md).

## Two merge passes, different jobs

```
Auto-Detect ──► RecognitionWorker ──► items
                                        │
   _recognition_items (visible rows) ──►│  pass 1: _merge_recognition
                    (preserve_existing) │  trainer_window.py:1663
                                        ▼
                                     merged ──► pass 2: _populate_review_panel
                                                trainer_window.py:1833
   annotations.json (confirmed) ───────────────► _recognition_items
```

| Pass | Compares | Purpose |
|---|---|---|
| `_merge_recognition` (`trainer_window.py:1663`) | fresh items vs rows already on screen | Auto-Detect is "find what is not here yet", so a new item that collides with a visible row is dropped |
| `_populate_review_panel` (`trainer_window.py:1833`) | fresh items vs `annotations.json` | a confirmed annotation carries the user's name/state and must survive re-detection |

Pass 2 runs on a cold open too (with an empty item list), which is why the
panel shows confirmed rows before any detection has run.

## Matching a fresh item to a confirmed annotation

`Annotation.ann_id` is `md5(f"{x}_{y}_{w}_{h}_{slot}")` truncated to 12 hex
chars (`training_data.py:127`) — bbox and slot, not name. Matching walks three
steps, in order:

1. **Exact `ann_id`.** Same bbox, same slot.
2. **Bbox IoU ≥ `IOU_RECOVER` (0.5), any slot** — `_find_legacy_aid`
   (`trainer_window.py:1933`). Absorbs detector drift of a few pixels and slot
   renames (a seat-keyed BOFF slot has a different `ann_id` than the
   `Boff Tactical` it was saved as).
3. **Single-instance twin** — rule 1 below.

A match makes the confirmed name and state win, unless the fresh name
disagrees, in which case the item goes to `community_conflict` or back to
`pending` (`trainer_window.py:2041`ff) so the user re-verifies rather than
having either side silently swallowed.

Confirmed annotations that no fresh item matched are appended after the loop
(`trainer_window.py:2159`) — this is what keeps 31 confirmed equipment rows on
screen when the detector, skipping already-tracked bboxes, emits two items.

## Precedence rules

A confirmed annotation outranks a fresh detection of the same thing: it is
what the user already settled, and its bbox is the one crops were exported
from.

**Rule 1 — a single-instance slot holds exactly one row.** When such a slot is
already confirmed, the fresh detection folds onto the confirmed entry —
keeping the confirmed bbox — instead of adding a second row
(`_single_instance_twin`, `trainer_window.py:88`, called at `:2006`). The fold
reuses the normal match path, so a name disagreement still surfaces as a
conflict. A second fresh item for the same slot (`:1960`) and a second
confirmed entry from legacy data (`:2167`) are both dropped.

`SINGLE_INSTANCE_SLOTS` (`training_data.py:57`) is the membership list. It
must stay equal to the set of slots the importer caps at `'max': 1`, per
build type:

| Source | Slots capped at one |
|---|---|
| `SPACE_SLOT_ORDER` (`warp_importer.py:218`) | `Deflector`, `Sec-Def`, `Engines`, `Warp Core`, `Shield`, `Experimental` |
| `GROUND_SLOT_ORDER` (`warp_importer.py:235`) | `Kit`, `Body Armor`, `EV Suit`, `Personal Shield` |
| `SPEC_SLOT_ORDER` (`warp_importer.py:281`) | `Primary Specialization`, `Secondary Specialization` |
| ship info (no `max` entry — one per screenshot by definition) | `Ship Name`, `Ship Type`, `Ship Tier` |

`tests/test_review_merge_dedup.py` asserts that equality for `SPACE`,
`GROUND`, `SPEC`, `SPACE_MIXED` and `GROUND_MIXED`, so a slot added to
`SLOT_ORDER` with `'max': 1` and forgotten here fails the suite.

**Rule 2 — icon bboxes may not overlap.** A fresh equipment / trait / BOFF box
whose IoU against an unconsumed confirmed box reaches `_MERGE_IOU_OVERLAP`
(0.3, `trainer_window.py:84`) is dropped; the confirmed row is re-added by the
leftover pass, so the position keeps exactly one row. The threshold matches
`IOU_THRESHOLD` in pass 1 so both passes agree on what "collides" means.
Ship-info text slots are exempt on both sides (`_confirmed_overlap`,
`trainer_window.py:104`): `Ship Type` spans the class line and `Ship Tier` the
`[T6-X2]` badge inside it, so their boxes overlap by design — rule 1 already
caps each at one row.

Storage enforces the same cap independently: `add_annotation` step 3
(`training_data.py:354`) drops any other confirmed annotation for a
single-instance slot before inserting. The panel rules exist because the
display list is assembled before anything is written back.

## Failure modes

| Symptom | Cause | Where to look |
|---|---|---|
| `order_for_review groups: … Ship Tier(2) …` | fresh tier bbox and confirmed tier bbox overlap below 0.5, so step 2 missed them — the case rule 1 closes | `:2006`, log line `folded onto confirmed bbox=` |
| Two rows on one icon, one confirmed and one pending | IoU between 0.3 and 0.5 — matched by neither step 2 nor rule 2 before it existed | `:2020`, log line `overlaps confirmed` |
| A confirmed row silently replaced by a fresh name | equivalence class or a previously rejected community proposal | `trainer_window.py:2052` |
| Confirmed rows vanish after Auto-Detect | item filtered out of `preserve_existing` by `_is_legacy` (no `slot_index`) **and** not present on disk | `trainer_window.py:1624` |

## Decisions and trade-offs

**The confirmed bbox wins, not the fresh one.** Rule 1 could have adopted
today's geometry the way `_find_legacy_aid` does (`:1989`). It does not: the
ship-info bbox moves with every OCR pass, the confirmed one is what the
exported crop and the community vote were cut from, and re-anchoring it on
each detection would rewrite the annotation for no gain.

**Rule 2 drops the fresh item instead of merging it.** Merging would have to
pick a winner for slot and name anyway, and the confirmed row already carries
both. Dropping keeps the row count equal to the number of physical positions,
which is what the canvas overlay assumes.

**`Ship Name` was added to `SINGLE_INSTANCE_SLOTS`.** It is one row per
screenshot like the other two ship-info slots, and it was the only one of the
three left out — so it was capped by neither the panel nor `add_annotation`.
Side effect: positional slot suggestion (`trainer_window.py:3871`) now skips
`Ship Name` once confirmed, which matches how the other capped slots behave.
