# Equipment panel detection

Production modules: `warp/recognition/eq_geometry.py` (panel geometry),
`warp/recognition/layout_detector.py` (`LayoutDetector._detect_via_pixel_analysis`
— row → slot labelling and bbox emission), `warp/warp_importer.py` (slot
profile). The detector locates the 6-cell × N-row equipment matrix on a
SPACE / SPACE_MIXED screenshot, decides which slot each row is, and emits
one bbox per slot the ship owns.

> **`dev/` is the maintainer's local working set** — gitignored in its
> entirety and absent from a checkout. Every `dev/*.py` path in this
> document is a reproduction pointer for whoever has the corpus on disk,
> not a script you can run from the repo. Measurements are quoted inline,
> so nothing here depends on having those files.

Sibling documents: [BOFF panel detection](BOFF_DETECTION.md),
[Trait panel detection](TRAIT_DETECTION.md). Slot counts per ship and the
game's own rules live in [STO slot rules](sto_slots_rules.md).

## Two independent sources, and why both are needed

A row's **position** and a row's **size** come from different places:

```
  screenshot ──► eq_geometry ──► panel_x, panel_right, dx, row_pitch, row_cys
                     │                            │
                     └──► eq_label_cys ───────────┤ WHICH slot each row is
                          (OCR labels)            │
                                                  ▼
  ship name ──► ShipDB ──► profile ──────────► HOW MANY icons in that row
                           (slot counts)
```

Geometry cannot know how many console slots a ship has — an empty cell and
a missing cell look identical. The profile cannot know where the panel is.
Every defect in this pipeline so far has come from one source silently
substituting for the other.

## Invariants

- **EQ-1** — One screen row carries exactly one slot. A slot never spans
  two rows, and two rows never share a slot name.
- **EQ-2** — An OCR-read row label outranks a positional guess. The guess
  exists only to fill rows whose label OCR missed.
- **EQ-3** — Column membership decides which OCR hit is a row label.
  Confidence only breaks ties *within* the label column.
- **EQ-4** — A pixel count is a lower bound on reality. It sees filled
  cells only, so it may under-report and must never be treated as an upper
  bound or as an authority over a known profile.

## 1. Panel geometry — `eq_geometry.detect_eq_geometry`

`detect_eq_geometry` in `warp/recognition/eq_geometry.py`. Returns an `EQGeometry`
(`EQGeometry` in `eq_geometry.py`) or `None` when OCR yields no usable labels.

| Field | Meaning |
|---|---|
| `panel_x_start` | left edge of the matrix, from an HSV stripe scan per label |
| `panel_right` | right edge, from single-slot icon right edges |
| `final_dx` | cell pitch in x — `(panel_right - panel_x_start) / 6` |
| `row_pitch` | cell pitch in y |
| `row_cys` | visible row centre-Y values, top → bottom |
| `eq_label_cys` | `{canonical slot index → cy}` — the OCR anchors |
| `mode` | `v8` when the right edge landed on real icons, else `MATH_FALLBACK` |

The module docstring carries the full step list and the derivation of
`DX_RATIO = 0.725` in `eq_geometry.py`. Two steps matter for correctness
beyond what the code states:

### Every keyword hit survives to the column filter

`_collect_single_hits` (`eq_geometry.py`) returns **all** single-line
keyword hits, including several for the same row name. `_cluster_by_x1`
(`_cluster_by_x1` in `eq_geometry.py`, tolerance `X_CLUSTER_TOL = 30`) then keeps the
largest x-cluster as the label column, which is what discards off-panel
text.

Collapsing to one hit per row name *before* that filter violates **EQ-3**
and loses rows outright. Observed failure: the tooltip word `Field`
fuzzy-matches the `Shields` keyword at 0.727, above `_FUZZY_CUTOFF = 0.65`
(`_FUZZY_CUTOFF` in `eq_geometry.py`). Both it and the real `Shields` label read at OCR
confidence 1.00, so a confidence-ranked collapse kept whichever came first
in OCR order — the tooltip — and the column filter then dropped that as
off-panel. The Shields row ended with no anchor at all.

Raising `_FUZZY_CUTOFF` is the wrong lever: it trades one class of error
for another on noisy OCR, while the geometry already carries the answer.

### Duplicate canonical indices

When two hits in the label column map to the same canonical index
(`STD_ORDER` in `eq_geometry.py`), `detect_eq_geometry` averages their
`cy` only if they are co-located — no further apart than the taller
label's own height. Further apart they are different rows, and averaging
would place the row where no row exists; the more confident hit wins
instead.

## 2. Row → slot labelling — `_detect_via_pixel_analysis`

`LayoutDetector._detect_via_pixel_analysis` in `warp/recognition/layout_detector.py`.

```
for each cy in geom.row_cys:
    slot = cy_to_slot.get(cy)              # OCR anchor, authoritative (EQ-2)
    if slot is None:
        slot = extended_order[row_index]   # positional guess
        if slot already anchored on another row:
            skip this row                  # EQ-1
    count = profile[slot]                  # how many icons to emit
    emit `count` bboxes right-to-left from panel_right
```

- `cy_to_slot` (`layout_detector.py`) is built from
  `geom.eq_label_cys` through `_STD_IDX_TO_PROD_SLOT`
  (`layout_detector.py`), which maps the geometry module's canonical
  names to production slot names (`Shields` → `Shield`).
- `extended_order` (`layout_detector.py`) starts from
  `SPACE_SLOT_ORDER_STANDARD` (`layout_detector.py`), drops slots the
  profile counts as 0, and inserts optional ones (`Sec-Def` after
  `Deflector`; `Experimental` / `Hangars` after `Aft Weapons`).
- The collision guard (`LayoutDetector._detect_via_pixel_analysis` in `layout_detector.py`) enforces **EQ-1**.
  Without it, `result[slot_name] = bboxes` replaced an OCR-anchored row's
  bboxes with a guessed row's, and the anchored row was left with nothing
  to confirm.

`self.last_row_pixel_counts` (`layout_detector.py`, reset at
reset in `LayoutDetector.detect`, written in `_detect_via_pixel_analysis`)
records what `_count_icons_in_row`
(`LayoutDetector._count_icons_in_row` in `layout_detector.py`) measured per row. It is written **before** the
profile decides what to emit, so rows the profile counts as 0 — and
therefore skips — still leave their measurement behind. That is the input
to §4.

## 3. Slot profile — `ShipDB._entry_to_profile`

`ShipDB.resolve` in `warp/warp_importer.py`, then `_apply_ship_and_tier_bonuses`
(`_apply_ship_and_tier_bonuses` in `warp_importer.py`).

Cargo stores slot fields as **strings**, so truthiness tests are wrong on
them: `bool('0')` is `True`. That gave 652 of 797 ships a phantom
`Sec-Def` and 568 a phantom `Experimental`. The phantom shifted
`extended_order` by one position, so every row below `Deflector` whose
label OCR missed took the name of the row above it. Slot presence now goes
through the same `_int` conversion as every other count, and profile
totals match cargo exactly (145 `Sec-Def`, 229 `Experimental`).

Bonuses applied on top of the cargo entry:

| Source | Effect |
|---|---|
| `Innovation Effects` in ship abilities (Miracle Worker) | +1 Universal Console |
| `Federation Intel Holoship` | +1 Universal Console |
| `T6-X` | +1 Universal Console, +1 Device, +1 Starship Trait |
| `T6-X2` | +2 to each of the above |
| `T5-U` / `T5-X` | +1 to the console type named in `t5uconsole` |

## 4. Tier recovery when the badge is off-screen

Many screenshots do not show `[T6-X2]` — the header is cropped, or the
player captured a view without it. `ship_tier` is then empty, no bonus is
applied, and `Universal Consoles` falls to 0. A row counted as 0 is skipped
entirely (§2), so the slot disappears from the output rather than appearing
empty; `Devices` and `Starship Traits` come up short by as much as 2.

`_infer_x_bonus` (`warp_importer.py`) recovers the upgrade level from
three measurements the same run already produced:

| Evidence | Source |
|---|---|
| `last_row_pixel_counts['Devices'] - profile['Devices']` | §2 |
| `last_row_pixel_counts['Universal Consoles'] - profile['Universal Consoles']` | §2 |
| `len(layout['Starship Traits']) - 5` | [trait grid](TRAIT_DETECTION.md) |

By **EQ-4** each is a lower bound, so the answer is their **max**, not a
majority. Measured against tiers read from the screenshots
(`dev/diag_tier_inference.py`, 22 SPACE_EQ / SPACE_MIXED screens):

| Rule | Correct | Wrong |
|---|---|---|
| majority vote | 8 | 1 |
| **max** | **21** | **0** |

Majority failed where two device slots were left unfilled: the device
evidence read `+1` while the console and trait evidence read `+2`. Of the
21 correct, 18 were unanimous and 3 were splits that max resolved. The set
includes a `T1` and a `T6` hull, both correctly inferred as `+0` — the rule
does not bias towards adding slots.

Evidence outside `0..2` is discarded rather than clamped: the game grants
at most +2, so a larger reading means the measurement is wrong and must not
size a row.

`_compose_inferred_tier` (`warp_importer.py`) turns the level into a
tier string. Slot evidence measures the **upgrade**, not the tier — `T5-X2`
and `T6-X2` both grant +2 — so the base number comes from cargo's `tier`
field, which is populated for all 797 ships. The result is validated
against `SHIP_TIER_VALUES` (`warp/recognition/text_extractor.py`).

The call site (`WarpImporter._process_image` in `warp_importer.py`) runs only when OCR found no tier at
all, only for `SPACE` / `SPACE_MIXED`, and re-runs detection so the
recovered rows get bboxes. A tier OCR did read stays authoritative.

**Asymmetry to preserve:** `x_bonus == 0` means *no evidence of an
upgrade*, never *not upgraded* — an unfilled build looks identical to an
un-upgraded one. The inference may therefore only ever add slots, never
assert their absence, and `_compose_inferred_tier` returns `''` for 0 so no
tier is claimed.

Downstream, the recovered tier is indistinguishable from a read one
(`ImportResult.ship_tier`, so `_apply_alien_species` in `warp/build_writer.py` stops writing a
plain `T6` for an upgraded ship). The one difference is
`RecognisedItem.src == 'inferred'`, which the trainer renders as an
`Inferred` row status — see the user manual,
[WARP guide § Right panel](WARP_GUIDE.md).

## Failure modes

| Symptom in logs | Cause | Where to look |
|---|---|---|
| `row N [Slot] … kept 0 within grid` absent for a visible row | `profile[slot] == 0`; row skipped before any bbox is projected | §3, §4 |
| `positional guess 'X' already anchored by OCR on another row` | OCR missed a label; the guess collided. Row left unlabelled by design | `LayoutDetector._detect_via_pixel_analysis` |
| `pixel_count=N profile=M` with `N > M` | profile under-counts — usually a missing tier bonus | §4 |
| Two slots emitted on one row cy | pre-`EQ-1` regression | `LayoutDetector._detect_via_pixel_analysis` |
| `mode=MATH_FALLBACK` | no single-slot icon right edge found; `panel_right` extrapolated | `detect_eq_geometry` |

## Measured baseline

14 EQ screens with `annotations.json` ground truth, greedy IoU ≥ 0.5
matching of emitted bboxes against confirmed annotations
(`dev/diag_eq_row_labels.py`; the baseline run used a git worktree at the
previous commit so both versions ran through the same harness):

| | TP | FP | FN | precision | recall |
|---|---|---|---|---|---|
| before §1–§3 fixes | 368 | 47 | 38 | 0.887 | 0.906 |
| after | 390 | 22 | 16 | 0.947 | 0.961 |

12 screens unchanged, 2 corrected to zero errors, none regressed.

### Why this stage reads the image a second time

`detect_eq_geometry` runs its own full-image OCR pass, after
`TextExtractor.scan_image` has already read the same screenshot. That looks
like waste and it is expensive — measured 2026-09-05 over five screen types
(`dev/probe_ocr_passes.py`), the second pass costs 2.1–3.0 s on a mixed screen,
roughly a third of all the OCR a screenshot needs.

Reusing the tokens `scan_image` produces was tried and **rejected on
measurement**. The two reads are not the same read: `scan_image` reads five
horizontal strips at higher effective resolution, `_run_ocr` reads the whole
frame at once, and their token boxes land in slightly different places. Feeding
the strip tokens to `detect_eq_geometry` over the 154 SPACE screens in the
training store (`dev/probe_eq_geometry_tokens.py`, both runs through the
shipped function):

| | images |
|---|---|
| identical geometry | 6 |
| **different geometry** | **119** |
| geometry lost (found before, not after) | 0 |
| geometry gained | 0 |
| no geometry either way | 29 |

Nothing is gained or lost outright — every screenshot that had a panel still
has one. What moves is *where* it is: row centres shift by 1–2 px, `panel_right`
by up to 2 px, `row_pitch` by 1. On `12.png`, `row_pitch` 51 → 50 and
`panel_right` 751 → 749.

A pixel or two sounds negligible against a 44–56 px icon box, and it may well
be. But it moves **every crop on 95 % of screenshots**, and crop geometry is
known to matter here by measurement rather than intuition — see the icon
resolution work, where stretching beat letterboxing by 10.7 points. Geometry
alone cannot say whether the shift helps or hurts; only the match rate can, and
that is a whole-corpus recognition run, not a geometry diff.

So the second pass stays until someone spends that measurement. The saving is
real and so is the risk; what is not acceptable is trading one for the other on
the strength of "the numbers look close".

## Open questions

1. `_cluster_by_x1` keeps the **largest** x-cluster as the label column. On
   a screenshot whose tooltip column contains more canonical-looking hits
   than the panel itself, the wrong column would win. Not observed on the
   105-screenshot corpus; no guard exists. Decision needed on whether to
   tie-break by proximity to `panel_x_start` before this becomes a real
   failure.
2. Universal Console slots are modelled as coming only from the Miracle
   Worker ability, the Federation Intel Holoship, or the X upgrade (§3). If
   a ship grants them by another route, §4 would read the surplus as an X
   bonus and inflate `Devices` and `Starship Traits` with it. Needs a
   cargo-side check of whether that route exists.
3. Whether the second OCR pass can be dropped (see above). Blocked on one
   measurement: the icon match rate over the whole corpus with strip tokens
   against the current full-frame read. Worth 2–3 s per mixed screenshot.
