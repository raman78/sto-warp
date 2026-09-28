# Equipment panel detection

Production modules: `warp/recognition/eq_geometry.py` (panel geometry from
the row labels), `warp/recognition/eq_stack.py` (panel geometry when there
are no labels, §6), `warp/recognition/layout_detector.py` (`LayoutDetector._detect_via_pixel_analysis`
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
  bound or as an authority over a known profile. A *cell* count is not the
  same measurement and not a lower bound — see §2, "Is there a cell here at
  all?" — which is why it, and only it, may narrow a row.
- **EQ-5** — Where the game writes, there is no slot. Enforced for every
  panel by `drop_boxes_on_text`, the last thing `LayoutDetector.detect`
  does; the rule and its three cases are described under "A slot never sits
  on a label" in `TRAIT_DETECTION.md`. The equipment panel is the reason
  that rule needs a position test at all: **it writes each row's label to
  the left of its icons, on the same line**, where the ground and trait
  panels put a heading above the block. A label level with its own row is
  therefore normal here and never a section boundary.
- **EQ-6** — The crop the matcher reads is exactly the bbox that is
  reported and drawn. Nothing moves it after the grid is built; see §5.

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
`DX_RATIO = 0.725` in `eq_geometry.py`. Three steps matter for correctness
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

### Row pitch counts only the rows it can be sure of

`row_pitch` is read from the labels OCR found: for two consecutive labels,
the gap between them divided by the number of rows that separate them. That
number comes from `STD_ORDER`, the rows in the order the game draws them —
a view of `space_eq_rows.ROWS`, keyed by the printed label (table in
[STO slot rules](sto_slots_rules.md)). It is only known when no
optional row (`OPTIONAL_ROWS`: Sec-Def, Aft, Experimental, Universal
Consoles, Hangars) lies between the two labels — such a row is drawn on
some ships and not on others. `_row_steps` makes that call, and only pairs
with a certain count feed the median. If OCR read no such pair, every pair
is used with its optional rows counted absent, and the log says so
(`row pitch from N label pair(s) with an optional row between them`). On
the 109 annotated space screenshots that fallback never fired.

Until 2026-09-27 `STD_ORDER` followed the slot *groups* in the rules
document — Aft and Experimental second and third — and every pair was
used. Pairs such as Fore → Aft or Shield → Devices then gave 5× or 2× the
real pitch, and Aft → Deflector a negative one. The median still landed
within a pixel, which is why no row went missing (1236 of 1240 GT rows
before and after), but the pitch read about 1 px short of the truth, and
now equals it.

The right-edge scan's search window, `RIGHT_SEARCH_END` cells of `est_dx`
(`row_pitch × DX_RATIO`) from `panel_x_start`, was tuned against that short
pitch. With the true pitch it reaches ~4 px further and on 9 screenshots
picks up bright pixels just past the icon. Measured through the shipped
importer, OCR on, 3206 confirmed equipment boxes
(`dev/measure_eq_right_edge.py`):

| | located | right slot | right item |
|---|---|---|---|
| before the fix | 3188 | 3185 | 3146 |
| fixed, `RIGHT_SEARCH_END = 6.05` | 3188 | 3185 | 3146 |
| fixed, `RIGHT_SEARCH_END = 6.00` | 3184 | 3182 | 3135 |

So the window stays at 6.05. Mean IoU against the confirmed boxes fell
from 0.991 to 0.977; most of those boxes are the old detector's output,
accepted, so that figure favours the old geometry by construction and
cannot separate a real shift from the bias. `DX_RATIO` was re-derived
against the true pitch and came out 0.727, so it is unchanged.

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
  (`layout_detector.py`), which maps a row's index to its slot name.
- `extended_order` is `space_eq_rows.row_sequence(slot_order, profile)`:
  the rows this ship draws, in the order the game draws them. It drops rows
  the profile counts as 0 and puts each optional row the profile grants at
  its place in `space_eq_rows.ROWS` — Sec-Def under Deflector, Experimental
  under Aft Weapons, Hangars last. The OCR-anchored fallback
  (`_detect_via_ocr_anchored`) builds its sequence with the same function;
  until 2026-09-27 it had its own copy, which put Hangars after Aft
  Weapons and so placed an unread Hangars label between Aft and Devices.
- The collision guard (`LayoutDetector._detect_via_pixel_analysis` in `layout_detector.py`) enforces **EQ-1**.
  Without it, `result[slot_name] = bboxes` replaced an OCR-anchored row's
  bboxes with a guessed row's, and the anchored row was left with nothing
  to confirm.

`self.last_row_pixel_counts` (`layout_detector.py`, reset in
`LayoutDetector.detect`, written in `_detect_via_pixel_analysis`)
records what `_count_icons_in_row`
(`LayoutDetector._count_icons_in_row` in `layout_detector.py`) measured per row. It is written **before** the
profile decides what to emit, so rows the profile counts as 0 — and
therefore skips — still leave their measurement behind. That is the input
to §4.

### When OCR does not deliver the label

`cy_to_slot` is authoritative (EQ-2), but it is only as complete as the read.
OCR fails on labels in three distinct ways, and each has its own recovery
because each has a different amount of evidence to work from. The order below
is the order of preference: a real reading always beats a projection.

| What went wrong | Recovery | Where |
|---|---|---|
| The label was read, with a wrong character | Match it to the nearest keyword | `ground_eq_geometry._fuzzy_slot` (ground), `eq_geometry._fuzzy_best` (space) |
| The label was not read at all, but others were | Project its position from the ones that were | `ground_eq_geometry._fill_missing_labels` |
| A row has no label and no projected position | Name it from the slots its neighbours bracket | `layout_detector.fill_unanchored_rows` |

**Near-miss matching.** The ground detector compared keywords by exact string
equality until 2026-09-05, so one wrong character dropped a whole row without
a trace. On `Screenshot_2025-03-19_122129.png` the reader returned
`Kil Modules` for a perfectly legible *Kit Modules* — one character in eleven
— and six cells were lost. The space detector had fuzzed all along; the ground
one simply never did. Both now use cutoff `LABEL_FUZZY_CUTOFF` 0.65 with a
length guard: only keywords within two characters of the token are considered,
which is what stops tolerance becoming invention (`devlces` scores 0.444
against `kit modules`, `weapors` 0.222).

**Projecting a missing label.** Ground rows are not evenly spaced — `Weapons`
holds two stacked cells, so what follows sits about one and a half rows lower.
But the spacing is a fixed *proportion* of the panel's scale. Measured over 17
ground screenshots with row pitches from 58 to 106 px, each row's offset from
`Kit Modules` in units of row pitch was constant to a standard deviation of
0.013–0.024 — about 1.2 px on a typical panel. Those constants are `ROW_RATIO`,
so any one label positions all the others. The **median** of the candidates is
taken, not the first, so a single label OCR placed slightly low cannot drag the
whole panel with it.

**Naming a row from its neighbours.** Where the ship and tier are known, the
panel's slot sequence is known too, so a run of unnamed rows between two
anchored ones can be filled from the slots that lie between those anchors.
`fill_unanchored_rows` does this **only when the count is unambiguous**: a run
of two unnamed rows is filled only if exactly two expected slots sit between
its neighbours. When the numbers disagree the profile cannot name the run, and
a wrong name would silently write an item into the wrong slot.

**Naming a row from what is in it — or guessing, and saying so.** Such a run
goes to `eq_row_naming.name_run` (`LayoutDetector._name_rows_by_content`). The
slots that can sit there are those between the run's named neighbours in
`space_eq_rows.ROWS`, not already used on the panel, and every mandatory one
among them — except, with no named row above the run, the mandatory rows
above the arrangement's own first row: the panel's top may be cut or
covered, and requiring Fore there once shifted a whole panel by one row and
turned Engines, Warp Core and Shield, read at 97-100%, into Unknown. Each
arrangement of them is scored by the items the matcher reads
in the run's filled cells — a cell whose item could not sit in the row the
arrangement gives it counts against it, "could sit" coming straight from the
cargo groups. Each row then comes out one of four ways:

| Outcome | When | What the user sees |
|---|---|---|
| named | one arrangement fits best and leaves at most 10% of the read cells out of place, or every best one agrees on the row | an ordinary group |
| guessed | an arrangement fits, but not certainly — ties go to the one with fewer optional rows, most ships lacking them | *"Devices (guess)"* |
| Unknown | no arrangement exists, or most of the row's items cannot sit in the slot the best one gives it | *"Unknown"*, *"Unknown #2"*… |
| outside | at the panel's open bottom, a trailing row the best arrangement leaves out; leaving a row out costs its items, so a real last row is kept and another panel's icons are not | no boxes |

A guessed or Unknown row keeps its boxes — one per cell the game drew, not
the profile's count, which says nothing about a row nobody can name — and its
items carry `RecognisedItem.row_guess` ('guess' / 'unknown'). Its
measurements stay out of `last_row_pixel_counts`, so a guessed Devices row
cannot raise a ship's upgrade tier. The importer puts a line in
`ImportResult.errors` naming the guessed and Unknown groups, which WARP CORE
shows under the review list.

In WARP CORE — and so in Fast Correction, the same review panel — such a
group's items cannot be accepted, by hand or by auto-accept, until the user
right-clicks the group and confirms the guessed type or chooses another
(`WarpCoreWindow._settle_eq_group`, which rematches the items against the
chosen slot). A confirmed item goes into the training data and the upload
queue with its slot, so a guessed slot confirmed item by item would be filed
as fact. Mark Done waits for the same reason, and its tooltip names the
groups still open. The grid of such a row is drawn on the canvas, and only
of such rows, so the user sees where WARP was unsure. The SETS export leaves
their items out and says how many (`WriteReport.unsettled_items`).

Measured 2026-09-28 over the 109 annotated space screenshots (55 SPACE_EQ,
54 SPACE_MIXED; 3230 confirmed equipment boxes), counting boxes whose slot
matches the annotation. With OCR switched off, so every row has to be named
without its label: 3093 boxes found; 3059 in rows named with certainty and
right, 3 named with certainty and wrong; 28 in guessed rows, 20 of them
right (the 8 wrong ones all on one screenshot); 3 in Unknown rows. The 3
certain-and-wrong boxes are wrong with OCR on as well, so they do not come
from naming by content. Also with OCR off, 101 boxes land where no item was
annotated; 34 of them on one screenshot whose panel the stack places
wrongly (none of its 33 items found), and two more screenshots get no
equipment boxes at all. With OCR on
no row is guessed or Unknown — the labels name them all — and 3215 boxes
are right, 14 land where nothing was annotated.

When the user draws a missing box, WARP CORE suggests its slot from the rows
around it (`WarpCoreWindow._suggest_slot_from_position`). It walks the same
order, from `space_eq_rows`, skipping rows the identified ship lacks — only
an identified one: the fallback profile for an unknown ship sets every
optional row to 0 and is a guess (`ImportResult.ship_matched`). And a box in
the equipment column takes only an equipment row as the one above it. Before
both, it offered Sec-Def for the Engines row of a ship with no secondary
deflector, and a BOFF seat's successor for a Shield row that had the seat
level with it in the next panel.

This replaced indexing a flat list by row number, which broke whenever the
list's order did not match the panel's. On `image-4e7c6849dd28da67.png`, where
the *Devices* and *Universal Consoles* labels are covered, the list placed
`Hangars` at position 6 — it is inserted after `Aft Weapons` while the game
draws it last — so row 6 was left empty, row 7 took `Devices`, every row below
shifted down one, and `Universal Consoles` was never placed at all.

### Is there a cell here at all?

`count = profile[slot]` above is the weak point when the profile is a guess.
A row is emitted **right-justified** from `panel_right`, so asking for too
few pushes the row's own leftmost cells off the panel, and asking for too
many puts a box on bare panel — which then reads as a blank cell and is
auto-confirmed as `__empty__` at confidence 1.00, teaching the models that
the panel background is an empty slot.

`_classify_cell` cannot help: it answers *what is in* a cell and takes for
granted that the crop is one. Measured 2026-09-06 over 1596 grid positions
that hold no slot, it called 60% of them `empty` and 35% `inactive` — 95% of
the places where nothing exists came back as something.

`LayoutDetector._cell_exists` is the missing question, and it reads the
screenshot against itself so that nothing has to be assumed about
resolution, UI scale or colour theme:

| Test | What it uses | Why |
|---|---|---|
| frame | Canny edge density over the cell | The game draws a border round every slot, filled or not. Existing cells: 22.6% of area at the 5th percentile, 32.4% median. Bare positions: 0.0% at the 95th. |
| band | mean absolute difference against the same region half a cell to the left | The panel is uniform along its length and a cell is not. Bare positions: 0.9 median, 18.9 at the 95th. Cells: 49.8 at the 5th percentile, 72.7 median. |

Both are required. The frame test alone is fooled where a row is short,
because the game draws **one outline around the whole run** of missing cells:
on `SovBuild.png` a two-cell `Universal Consoles` row read as six, the bare
positions scoring 4.1–7.5% purely from that outline. Colour was tried and
rejected — the reading "bare is flat navy, an empty cell is near-black"
holds on some screenshots and not others, and sized rows correctly 62% of
the time against 96% for the frame.

`_count_icons_in_row` stops at the first position the pair rejects, and the
number of cells it walked is recorded as `last_row_cell_counts`. Rows are
right-justified, so the missing cells are always a left prefix and stopping
is the whole answer.

It also stops where the screenshot does. A position whose window starts
left of the image cannot be judged: the band test has nothing to compare
with, and the frame test alone reads whatever sliver is left. On
`image-c8be3f34ec234254.png`, cropped at the panel's left edge, the sixth
position's window began 2 px off the image and held only the panel's edge
and the fifth icon's border, so a five-cell Fore Weapons row counted six.
The walk now ends there and logs `row [slot] stops at position N: its window
starts K px left of the image`. The guard had always been in the code, but
after a clamp to 0, so it never fired. Raising the frame test's bar for this
case was tried and rejected: confirmed empty and inactive cells carry only
5.7% and 7.7% edges at the median, the same range as the sliver (3.8%) and
the gap outline (4.1-7.5%), so no bar separates them. Across the 109
annotated space screenshots this was the only one with a position off the
image, and no confirmed box touches the left edge.

#### When the measurement is allowed to reach the boxes

A changed profile has to be re-projected or the measurement is a number in
the log and nothing else. That re-projection used to be skipped whenever the
screenshot carried *any* confirmed annotation, on the reasoning that
re-detecting would overwrite the user's pixel-perfect bboxes. It does not:
the confirmed merge puts them back, preferring a confirmed box wherever one
overlaps a detected one, and keeping the rest so a corrected row can gain a
cell.

What the old condition did instead was freeze a screenshot's layout the
moment its first row was confirmed. A row that gained a cell could never gain
a box, and a phantom the user deleted came back on the next run because the
profile still asked for it — the correction was undone by the program.

The state that means "this layout is settled" is the one the user sets and
can see: the screenshot marked done (`screenshots_done.json`, written by the
trainer beside `annotations.json`). Anything else is work in progress and
gets a fresh scan with the confirmed boxes merged back on top. Measured on
`image-939dc3ed9dd1eb95.png`, a cropped panel the maintainer had already
annotated: not done → Devices 6, Science Consoles 2, Aft Weapons 4 (the four
confirmed ones surviving a detection that found three); marked done → the
previous 5 and 3.

**This count is not a lower bound**, which is what separates it from
`last_row_pixel_counts` and from **EQ-4**: a cell the game drew is there
whether or not anything is in it. So `WarpImporter._process_image` may take
it as the row width in both directions when no ship was identified, where
the filled count could only ever raise the guess. Measured over 666
confirmed rows on that path: 531 correct before, 623 after, with phantom
cells down from 93 to 16 and rows short of a cell from 42 to 27. On the
screenshots where a ship *is* identified the profile stays authoritative and
nothing changes — 2868 of 3085 confirmed boxes either way.

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
| `last_trait_icon_counts['Starship Traits'] - profile['Starship Traits']` | [trait grid](TRAIT_DETECTION.md) |

By **EQ-4** each is a lower bound, so the answer is their **max**, not a
majority.

All three have to be **measurements**, and the third was not until
2026-09-06: it counted the boxes the finished layout carried for
`Starship Traits`, and those are drawn from the profile at the game maximum
so that no slot goes undrawn — seven, whatever the ship. The evidence
therefore read `7 - profile`, which is at least 1 for every ship below
`-X2`, and the max rule promoted it. Measured on
`image-817e2e37c01aed8c.png`, a `T6-X` Terran Adamant: devices `3-3 = 0`,
universal consoles `1-1 = 0`, traits `7-6 = 1` → raised to `T6-X2`. The
count that *was* a measurement sat in the same run — `trait_grid` had found
5 icons — and gives `-1`, out of range and discarded, leaving the tier
alone. `LayoutDetector.last_trait_icon_counts` now carries it, filled by
`merge_trait_boxes`, which is the one place the grid's own boxes are known.

The distinction is the same one **EQ-4** draws for equipment rows: a
detector's count and a profile-sized row answer different questions, and
only the first is evidence about the ship. Measured against tiers read from the screenshots
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

The call site (`WarpImporter._process_image` in `warp_importer.py`) runs for
`SPACE` / `SPACE_MIXED` only, and re-runs detection so the recovered rows
get bboxes. It fires in two situations: no tier on screen at all, and a tier
the OCR did read that the rows overshoot. The second is the newer and less
certain of the two, which is why the log tells them apart — a raise names
the tier it started from — and why the evidence is printed with it, as
`slot counted-profile` for each measurement that spoke. That line reports
what was measured *before* the raise; reading it off the profile afterwards
described the consequence and made the defect above invisible.

The surplus is measured against the profile, which already carries whatever
the read tier granted, so what comes back is the amount by which the screen
exceeds that tier — never the absolute level.

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

## 5. From bbox to crop

`WarpImporter._process_image` cuts each slot's crop straight from the bbox
the layout emitted, and that same bbox is what the trainer draws over the
screenshot. The drawn box is therefore an exact record of what was
recognised. If a box sits a few pixels off an icon, the matcher saw it
the same few pixels off. This holds for every panel, because the slot loop
is shared.

Until 2026-09-25 that was not true. A stage called P5 ("icon-to-layout
feedback", from the time when slot positions were predicted rather than
measured) worked like this. If `Deflector`, `Engines`, `Warp Core` or
`Shield` matched below 0.85, it scanned crops up to 40 px above and below.
The first one to score above 0.96 set a Y offset, and that offset was added
to **every later crop** on the screen, traits, BOFFs and reputation
included. The reported bbox stayed where it was, so the overlay looked
right while the reading came from somewhere else.

It was removed on measurement (`dev/p5_measure.py`, shipped importer,
134 SPACE_EQ / SPACE_MIXED screens):

| | screens |
|---|---|
| P5 scanned | 9 |
| P5 moved the grid | 3 distinct (+16, +12, −4 px) |
| move caused by a community pHash hit on the shifted crop | 3 of 3 |
| unshifted crop hit the same table | 0 of 3 |

A pHash hit is a hard override at 1.00, so it cleared the threshold on a
crop that was mostly the wrong cell. On one screen it named the wrong
deflector as well. With P5 off, mean confidence on the moved screens rose
from 0.67 to 0.88, from 0.76 to 0.93 and from 0.81 to 0.88. Against ground
truth one screen went from 47 to 52 of 62. The only slot that got worse
(a warp core correct only through the shift) came from two similar icons
being confused, not from misalignment: the grid box sits on the icon.

Rows that differ between ship types need no offset. They are handled by
the profile, which adds or drops whole rows (§2, §3).

## 6. When the panel has no labels — `eq_stack.detect_eq_stack`

§1 places the panel from its row labels. A screenshot cropped past the label
column — a collage of panels, say — has none, and neither does one whose
OCR pass read nothing. Such a panel used to get no equipment boxes at all.
`LayoutDetector._get_eq_geometry` now asks `eq_stack.detect_eq_stack` when
`detect_eq_geometry` returns `None`, so labels still come first and the
label-less path runs only where they are missing. Its geometry carries
`mode='stack'`.

### Finding the panel

The idea is that the grid itself is recognisable. Every row is
right-justified, so the panel's rightmost column has a cell in every row,
one row pitch apart, filled or not. And every ship draws Deflector, Engines,
Warp Core and Shield — Sec-Def on some — as one-cell rows directly under
Fore Weapons. No other panel has that shape.

1. Icon-sized bright components (`trait_grid._detect_icon_ccs`) are grouped
   by right edge and size and chained at whole multiples of one row pitch.
   Dark or merged icons are missed by a brightness test, so a chain may skip
   up to four rows.
2. Every skipped row inside the chain must pass `_cell_exists`, which answers
   for empty cells too. The stack is then extended up and down while
   `_cell_exists` still sees a cell.
3. `panel_right` is the median right edge of the chain — within 3 px of the
   confirmed boxes on 98 of 109 screenshots; the adaptive-background scan of
   §1 made it worse here. The cell pitch in x is each row's span from first to
   last cell over its steps, a fraction of a pixel: the median of single steps
   (34 or 35 px) drifted 2 px by the fifth cell.
4. Each row's cells are counted by `_count_icons_in_row`, called with the
   arguments `_detect_via_pixel_analysis` builds from a geometry.
5. The stack is the panel only if it is one row, then a run of 3-6 one-cell
   rows, then 3-8 more rows (`_fit_panel`). Rows outside that are trimmed. A
   trait grid never fits — the game draws the frame of every empty trait
   slot, so its rows are full — and a side list fails the rule that the run
   starts on the second row. A run of 4 or 5 is exactly Deflector..Shield,
   so Fore is the row above it; a run of 3 or 6 may be off by one because
   something covered a row, so two rows are kept above it. On
   `Hirogen Predator.png` with OCR off, a weapon tooltip over the Deflector
   row made it read five cells; trimming to one row above the run cut Fore
   Weapons off the panel.

### Naming the rows

Rows are named by writing them into `eq_label_cys`, the channel OCR labels
use, so everything after it — `fill_unanchored_rows`, the profile, bbox
emission — is unchanged.

- **From the shape.** A run of four one-cell rows is Deflector, Engines,
  Warp Core, Shield; five adds Sec-Def after Deflector. The row above is
  Fore Weapons, the row below Aft Weapons. A run of three or six is
  ambiguous and names nothing.
- **From the content, under Aft Weapons.** There the game draws
  [Experimental], Devices, [Universal], Engineering, Science, Tactical,
  [Hangars] (`space_eq_rows.ROWS`). The number of rows fixes how many of the
  optional ones are present but not which. Each filled cell is read by the
  matcher, and every arrangement the table allows is scored by the cells
  whose item could not sit in the row it assigns. "Could sit" comes straight
  from the cargo groups: a Universal console is in every console group, and
  the Universal row's group holds every console, so only typed consoles,
  devices, experimental weapons and hangar pets tell rows apart.
- **Only where the answer is unique.** The best arrangement may leave at most
  10% of the read cells out of place (`MAX_MISFIT`) — a sole arrangement is
  not right because it is the only one. Trailing rows in which nothing was
  read may lie past the panel's bottom, so arrangements without them are
  scored too, and where several share the best score only the rows they all
  agree on are named. A row left unnamed is logged with the reason
  (`eq_stack: N row(s) under Aft Weapons left unnamed — …`) and is handled
  like any row whose label OCR missed: named from its content, guessed or
  called Unknown (§2, "When OCR does not deliver the label").

The importer hands the matcher to `detect()` for `SPACE` builds for this
alone (`_needs_matcher`); nothing else in the space chain uses it.

### Keeping what it names

`detect()` for a `SPACE` build accepts the pixel analysis only when it covers
70% of the profile's slots, and otherwise falls back to learned layouts and
fixed positions. A label-less grid names only the rows it is sure of, so it
can fall short of that while every box it names is right; its result is kept
regardless. Measured with OCR off over the 55 annotated SPACE_EQ screenshots,
before content naming existed: 1092 correctly named boxes keeping it, 973
sending those screens on to the fallbacks, whose fixed positions misname
rows.

### Measured

The shipped importer over the 109 annotated space screenshots, 3206
confirmed equipment boxes (`dev/measure_eq_stack_e2e.py`,
`dev/measure_eq_right_edge.py`). A box counts when it overlaps a confirmed
one at IoU ≥ 0.5; "right slot" when it also carries that box's slot.

| OCR switched off (no labels anywhere) | confirmed | right slot before | right slot now | wrong slot now | boxes with no confirmed box, before → now |
|---|---|---|---|---|---|
| SPACE_EQ | 1668 | 899 | 1588 | 6 | 384 → 25 |
| SPACE_MIXED | 1538 | 0 | 1266 | 2 | 0 → 92 |

With OCR on, every screenshot with labels comes out identical; the one
without them, `image-c8be3f34ec234254.png`, went from no equipment boxes to
all 11 rows, its 6 confirmed boxes found with the right slot and item.

Naming by content costs about 1.2 s per label-less screenshot — some 16
matcher calls — and nothing where labels were read. The added recognition
time on such screens is mostly the importer matching the slots that now
exist.

## Failure modes

| Symptom in logs | Cause | Where to look |
|---|---|---|
| `row N [Slot] … kept 0 within grid` absent for a visible row | `profile[slot] == 0`; row skipped before any bbox is projected | §3, §4 |
| `positional guess 'X' already anchored by OCR on another row` | OCR missed a label; the guess collided. Row left unlabelled by design | `LayoutDetector._detect_via_pixel_analysis` |
| `pixel_count=N profile=M` with `N > M` | profile under-counts — usually a missing tier bonus | §4 |
| Two slots emitted on one row cy | pre-`EQ-1` regression | `LayoutDetector._detect_via_pixel_analysis` |
| `mode=MATH_FALLBACK` | no single-slot icon right edge found; `panel_right` extrapolated | `detect_eq_geometry` |
| `eq_stack: no equipment panel found without labels` | no labels, and no stack of cells with the equipment panel's shape — a cropped fragment, or no panel at all | §6 |
| `eq_stack: N row(s) under Aft Weapons left unnamed — …` | the rows' items fit no single arrangement; the reason is in the line | §6, `_content_anchors` |
| `row N → 'X' (guess)` / `row N → 'Unknown' (unknown)` | nothing established the row's type; it is shown to the user to settle | §2, `eq_row_naming` |
| `row N left out — past the panel's bottom` | a label-less stack ran on under the panel; the row's items fit nothing that can follow | §2, `eq_row_naming` |

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

### And the reverse — one whole-frame read for everyone — was tried and drawn

The other direction is more promising on paper: keep the whole-frame read this
stage already does, make it the *only* read, and retire
`TextExtractor.scan_image`'s five strips. That would leave space EQ geometry
byte-identical, since it is already the whole-frame reader, and remove the
strip splitting, the boundary stitching and the cross-strip dedup outright.

Measured over 52 screenshots spanning all 14 screen types
(`dev/probe_one_read.py`):

| | changed |
|---|---|
| screen type | 2 of 52 |
| ship class and tier | **0 of 52** |
| ship header bboxes | 4 of 52, by 1 px |
| ground equipment geometry | 9 of 52 |

The two screen-type changes are noise: one screenshot is read better, one
worse, and on the second both readings produce only mangled tokens
(`sterahip`, `srorahi`) with no keyword either way.

The grid was the open question, and a table could not answer it — this grid was
calibrated by eye in the first place. So both candidate grids were drawn on the
same screenshot, today's in green and the whole-frame one in red
(`dev/draw_grid_compare.py`). **Today's sits tight on the icons; the
whole-frame grid sits 2–3 px low**, clipping the top of the Body Armor and EV
Suit cells on both ground screenshots examined.

So the strips stay, and the reason they stay is now the measured one rather
than the one that used to be written in `scan_image` — which claimed
resolution, and was false. Why a narrow band locates a label row more
accurately than the whole frame is not established; that it does, is.

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
