# Ship info detection

Production module: `warp/recognition/text_extractor.py`, entry point
`TextExtractor.extract_ship_info` (`warp/recognition/text_extractor.py`).
It reads three fields off the top band of a space screenshot — ship **name**,
ship **class** (called `ship_type` throughout the code) and **tier** — plus a
bounding box for each. `ShipDB.resolve` (`warp/warp_importer.py`) then
turns the OCR'd class string into a canonical entry from the ship database,
which decides the slot profile for the whole build.

> **`dev/` is the maintainer's local working set** — gitignored in its
> entirety and absent from a checkout. Every `dev/*.py` path below is a
> reproduction pointer for whoever has the corpus on disk, not a script you
> can run from the repo. The measurements are quoted inline, so nothing here
> depends on having those files.

Sibling docs: [`EQ_DETECTION.md`](EQ_DETECTION.md) (equipment panel),
[`BOFF_DETECTION.md`](BOFF_DETECTION.md), [`TRAIT_DETECTION.md`](TRAIT_DETECTION.md).
This doc covers only the text block above those panels.

---

## Why an anchor at all

The ship-info block has no fixed position: UI scale, resolution and window
mode all move it, and on a merged (`SPACE_MIXED`) screenshot it shares the top
band with the traits panel legend, the active-duty header and whatever tooltip
was open at capture time. Cropping a fixed ROI is therefore unreliable —
`SHIP_INFO_ROI` exists but is only used by the re-OCR fallback in
`refine_ship_info`.

Instead the extractor finds one token it can trust (the **anchor**) and reads
the rest of the block relative to it. Everything downstream — which tokens may
join the class string, which row holds the name — is expressed as a constraint
against that anchor.

## Invariants

- **S1** — A `ship_*_bbox` is the union of exactly the tokens that went into
  the corresponding string. Nothing is added for padding, nothing is dropped.
  A bbox that spans emptiness therefore means the string is wrong too.
- **S2** — The bboxes are consumed, not merely displayed. The meta-slot
  block at the end of `WarpImporter._process_image` emits `Ship Type` and
  `Ship Tier` as review items carrying them, the trainer uploads the
  matching image strips as text crops (`_TEXT_CROP_PREFIXES` in
  `warp/trainer/sync.py`), and `refine_ship_info` re-OCRs from them. A wrong
  bbox is a wrong training sample, not just a cosmetic defect.
- **S3** — `Ship Name` is anchor-internal. It is never emitted as a slot —
  the comment guarding that same block says why — because it identifies the
  player, not the build.
- **S4** — A non-empty `ShipResolution.type` does **not** mean the class was
  recognised. When nothing matched, the OCR string is echoed back;
  `ShipResolution.matched` is the only honest signal.

## 1. Anchor strategies

Four passes, tried in order, all inside `extract_ship_info`. The first three
look for a tier badge and share the `tier` anchor kind; the fourth falls back
to the ship-name line:

| Pass | Trigger | Anchor token | Section comment |
|---|---|---|---|
| 1 — loose | `RE_TIER_LOOSE` matches, or one token holds a snappable `[...]` | the token carrying it | `Anchor 1` |
| 1b — fused | A whole `Name [TB-X2]` line came back as one token | that token | `Anchor 1b` |
| 1c — split | Two or three x-adjacent tokens in one row **join** into a closed `[...]` that snaps | a synthetic token spanning the fragments | `Anchor 1c` |
| 2 — name | A token that looks like a ship-name line, per `_is_name_prefix_token` | that token | `Anchor 2` |

With a tier anchor the class is assembled from the tier row, or from the row
above when that row held nothing; with a name anchor, from the row below the
name. No anchor means no ship info at all — the log says
`TextExtractor: no anchor, ship info unset` — and the extractor emits
`anchorless_candidates` instead and `ShipDB.find_class_by_candidates_ex` gets
a last-resort attempt.

Pass 1c exists because OCR sometimes cuts the badge itself rather than the line
around it, and neither half is then recognisable: see
[Decision 2026-08-30](#decision-2026-08-30-a-tier-badge-cut-in-half-is-still-a-tier-badge).

`_is_name_prefix_token` is misleadingly named: it does not require a prefix.
`U.S.S. ENTERPRISE` matches, and so does a bare `Henrik Lindstrom`, which is
why ground/character screens produce a name at all. Measured on the 254-shot
corpus, 48 screenshots have a `ship_name` with no `U.S.S.`-style prefix.

## 2. The column window

The HUD stacks name / class / registry in one left-aligned column, so the
extractor only accepts tokens inside a horizontal band around the anchor
computed inside `extract_ship_info`:

```python
# warp/recognition/text_extractor.py
col_pad = max(80, int(min(anchor_w, _COL_PAD_ANCHOR_CAP) * 2.0))
col_lo  = anchor_x - col_pad
col_hi  = anchor_x + anchor_w + col_pad
```

`_COL_PAD_ANCHOR_CAP = 150` is the part worth explaining. The pad used
to be `anchor_w * 2.0` uncapped, which is fine while the anchor is a short
`U.S.S.` token (~80 px → ±160 px) but collapses when OCR returns a whole class
line as one token:

```
anchor 'Legendary Scimitar Intel Dreadnought Warbird [T6-X2]'
       x 46..450, w=404
  → col_pad 808 → window x ∈ (-762, 1258) on a 1544 px image (81 %)
  → 'Thebe officers' at x=1219 (traits legend) accepted as ship class
  → ship_type_bbox (46, 21, 1255, 31), across the equipment column
```

The anchor's width is not a property of the column, so it stops widening it
past the cap. Anchors narrower than 150 px are unaffected — which is why the
cap costs nothing (see [Measured baseline](#measured-baseline)).

## 3. Class assembly

| Step | Function | Rule |
|---|---|---|
| Same row | `_adjacent_left_of` | tokens left of the tier token, contiguous in x; stops at a gap > `max(40, 4 × token height)`. A token may overrun the anchor's left edge by `_LEFT_OVERLAP_RATIO × anchor height` and still count — EasyOCR boxes for neighbouring glyph runs overlap |
| Row above | `_row_to_type` | whole row, **only when the tier's own row yielded nothing**, and if it is in the column, is not a name row, and holds no ALL-CAPS proper noun |
| Token filter | `_valid_type_tok` | `_valid_type_tok_nearby` — length > 2, not in `_HUD_BLACKLIST`, not a section header, not a registry number — **plus** inside the column |

`_adjacent_left_of` calls `_valid_type_tok_nearby`, i.e. it deliberately skips
the column window. The window rejects text that shares the anchor's y-band but
sits far away in x, and the gap rule already does that, strictly and on one row
— applying both cut the class line short instead. See the decision below.

`_HUD_BLACKLIST` gates the class only. Name assembly does not consult it,
which is how `Kit Modules` and `Starship Selection Dry Dock` survive as
`ship_name` on some screenshots.

## 4. Resolution against the ship database

`ShipDB.resolve` runs a ladder of strategies and promotes the canonical class
string from the DB over the OCR string whenever a real entry was found. The
strategy that won is reported in `last_match_strategy`
— all of it inside `ShipDB.get_profile` — in the order tried:

| # | Strategy | Index it searches | Identifies |
|---|---|---|---|
| 1 | `exact-type` | `_by_type` | class |
| 2a | `word-subset`, `word-subset-best` | `_by_type` keys | class |
| 2b | `display-name`, `display-name-best` | `_display_index` + tier | **ship** |
| 2c | `fuzzy-display` | `_display_strings` (names) | **ship** |
| 2d | `token-overlap` | `_display_index`, weighted | **ship** |
| 2e | `fuzzy-type` | `_by_type` | class |
| 3 | `keyword-fallback` | — | nothing |

`anchorless-rescue` is reported in place of whichever strategy won when
`resolve` first recovered the class string from loose OCR candidates.

### Class-only strategies come last, on purpose

The **Identifies** column is the thing to keep in view. `_by_type` is keyed on
the generic `type` field, so 797 ships collapse into 38 dict slots and the
entry under a key is whichever ship the load loop wrote last — `cruiser` holds
one of 93, `battlecruiser` one of 91. A hit there names a *class* and then
returns an arbitrary member of it. Measured over the roster, that member's
slot profile is wrong on 4.9 of 11 slots for the ship actually on screen; a
class consensus, the best any redesign of the index could manage, still misses
4.1. The information is not in a class name.

So every strategy that can name a real ship is tried before either class-only
stage that could pre-empt it. Two failures came from getting that order wrong:

- `fuzzy-type` (cutoff 0.68) used to sit above `fuzzy-display` (cutoff 0.85).
  A single dropped letter — `Lexington` read as `exington` — scored 0.7333
  against the class `heavy dreadnought cruiser` and answered
  `Universe Temporal Heavy Dreadnought Cruiser`, while the true ship sat one
  stage below at 0.9859.
- `fuzzy-type` also used to sit above `token-overlap`. On `Screenshot_96.png`
  the junk token `Decails` (OCR of the **Details** button) entered the class
  line; `token-overlap` scores the right ship at 10.5, but `fuzzy-type`
  answered first with the same wrong dreadnought.

`fuzzy-display` is additionally guarded against the opposite error. A read
that is *itself* a class name must not be attributed to a ship — 9 of the 27
multi-word class names sit above 0.85 from some ship name, e.g.
`warbird battlecruiser` is 0.9048 from `arbiter battlecruiser`. The guard
tests that property directly (a ≥0.90 match against the class index), rather
than the old proxy of demanding ≥3 OCR words, which also discarded every
two-word ship name.

Measured over all 797 ships after both reorderings: 790/797 clean names,
745/746 with one letter dropped, 792/797 with a junk token prefixed, and 65/65
class reads correctly *not* attributed to a ship.

### Confidence reflects which column won

`ship_type_confidence` grades the emitted `Ship Type`
item by strategy: 1.0 when a ship was identified, 0.45 when only the class
was, 0.30 when nothing matched. WARP CORE's auto-accept threshold is
user-settable between 0.50 and 1.00, so the two low bands can never be
auto-confirmed at any setting. Before this, `Ship Type` was emitted at a flat
1.0 and a class-only guess was written into `annotations.json` as ground truth
without the user seeing it.

This layer is forgiving by design, which is why a polluted class string often
still resolves correctly, and why measuring the extractor by string equality
overstates its errors.

## Failure modes

| Symptom in the log | Cause | Where to look |
|---|---|---|
| `type='<junk> <real class>'` | a token from another panel entered the column | `_valid_type_tok` / column window, §2 |
| `ship_type_bbox` far wider than the class line | same cause — the bbox is the union of what got in (S1) | §2 |
| `ship_type_bbox` *narrower* than the class line, `Ship Tier` sharing it | no anchor fired, so the box is one rescue token and the tier borrowed it | §1 pass 1c, and `no anchor` in the log |
| `[name anchor]` on a screen with no ship | `RE_NAME_PREFIX` matches a plain word: dots are optional and the separator class accepts a space, so `Die E` matches | §Open questions, item 2 |
| `no anchor, ship info unset` | no tier bracket and no name-shaped token | expected on ground/BOFF/skills screens |
| Class resolves to a different ship of the same family | fuzzy lookup fed a polluted string | §4, and `last_match_strategy` in the log |

## Measured baseline

Corpus: 254 screenshots under `~/Shared/STO_screens`, one OCR pass each, tokens
cached so policy variants can be re-scored offline
(`dev/diag_ship_bbox_span.py`, `dev/diag_ship_col_sweep.py`). The oracle is
`ShipDB.resolve(...).matched` — a real DB hit, not string equality (S4).

| Variant | DB matches | Resolution changes |
|---|---|---|
| uncapped pad (pre-fix) | 111 | — |
| cap 100 / 120 / **150** / 200 | 111 | 1, a correction (`Screenshot_96.png`) |
| cap 250 | 111 | 0 — too loose to fix the case |

Any cap in 100–200 px behaves identically, so the shipped value sits in the
middle of the plateau. Ten screenshots lose a junk prefix in `ship_type`
(`Thebe officers`, `Decails`, `Persona`, `Filters`, `0.5.5. schuberT`, and six
`AWAY TEAM`). `ship_*` bboxes wider than 40 % of the image: 15 → 11; the
remainder are genuinely long class names (41–48 %) plus `doff.png`.

### Decision 2026-08-26: the tier tiebreaker is confined to equal-length candidates

**Change.** `_fuzzy_tier` picks the
canonical tier with the highest list index among candidates scoring within
0.10 of the best ratio. That pool is now filtered to candidates of the **same
length as the OCR token**, falling back to the unfiltered pool when nothing
matches on length. One line:

```python
same_length = [v for v in pool if len(v) == len(cand)]
return max(same_length or pool, key=SHIP_TIER_VALUES.index)
```

**Why.** The tiebreaker exists to rescue `'T6-XZ'` (a misread `2`) from being
demoted to `T6-X`. On noisy input it did the opposite. `'TG-X'` — a `6` read
as `G`, seen on `b1-faa648cbbf29a0f9.png` — scores 0.750 against `T6-X` *and*
`T5-X`, and 0.667 against both `-X2` forms. The 0.10 buffer reaches 0.65, so
the `-X2` variants entered the pool and "prefer higher" promoted the token two
steps to `T6-X2`.

The cost is not cosmetic. `_apply_ship_and_tier_bonuses`
grants +1 Universal Console, Device and Starship
Trait for `-X`, +2 for `-X2`, so the layout asked for rows the ship does not
have — measured on that screenshot, 2 Universal Consoles where the pixels
showed 1, and 5 Devices where they showed 4. Under `T6-X` the profile matches
the pixel counts exactly.

**Alternatives measured.** Thirteen OCR variants seen in the wild, three
rules:

| Input | Expected | Buffer 0.10 (old) | Buffer 0.05 | Equal length (shipped) |
|---|---|---|---|---|
| `TG-X`, `TB-X`, `T8-X`, `TS-X` | `T6-X` | ✗ `T6-X2` | ✓ | ✓ |
| `T6-XZ`, `T6-XL` | `T6-X2` | ✓ | ✗ `T6-X` | ✓ |
| `T6-X`, `T6-X2`, `T6`, `TB-X2`, `T8-X2`, `T5-U`, `TS-U` | itself | ✓ | ✓ | ✓ |
| **errors** | | **4** | **2** | **0** |

Narrowing the buffer alone breaks the case the tiebreaker was written for.
The length rule rests on OCR substituting characters far more readily than
dropping them, so a four-character bracket is not a five-character tier.

**How to revert.** Delete the `same_length` line and return
`max(pool, key=SHIP_TIER_VALUES.index)`. `tests/test_tier_snapping.py` then
fails on the four `test_a_mangled_digit_does_not_promote_the_suffix` cases and
passes everything else — that split is the signature of the old behaviour, so
a revert is visible rather than silent.

**Not covered by this change.** A badge this rule snaps to the wrong value is
still wrong. Two things catch that afterwards: a tier confirmed in WARP CORE
(the decision below), and — since 2026-08-27 — the measured row counts, which
may raise a tier read too low.

### Decision 2026-08-26: a confirmed class or tier outranks OCR, in WARP CORE only

**Change.** `_process_image` consults `_load_confirmed_ship_info(source)`
before ShipDB resolves the ship, gated on `_is_trainer_call` — the same
`_use_confirmed` gate that already loads the confirmed layout. A user-confirmed
`Ship Type` or `Ship Tier` replaces what OCR read, and the swap is logged with
both values:

```
WarpImporter: tier 'T6-X2' (OCR) → 'T6-X' — confirmed by user, taken as
truth; slot bonuses follow the confirmed tier
```

**Why.** The tier decides how many boxes get drawn, and the correction had
nowhere to go. `_merge_recognition` keeps the confirmed row — the freshly
detected one is dropped as a `SINGLE_INSTANCE_SLOTS` duplicate — but the grid
had already been sized from the OCR value earlier in the same run. The user
could correct the tier as often as they liked and the surplus rows returned on
every Auto-Detect.

**A second defect surfaced while implementing it.** `_load_confirmed_layout`
and `_load_confirmed_profile` look the image up by **filename**, while
`TrainingDataManager` has keyed `annotations.json` by the first 16 hex chars
of the image's sha256 for some time. Measured on one maintainer's store: 168
of 178 entries are hash-keyed and therefore invisible to those two functions,
including 130 confirmed `Ship Type` / `Ship Tier` rows. `_use_confirmed` has
been substantially dead. The new `_annotations_for` helper reads both schemes,
hash first, and all three loaders now go through it — so the confirmed
*layout* (pixel-perfect bboxes instead of estimated ones) starts being applied
where it never was.

**Boundaries.**

| | |
|---|---|
| Who counts | user-confirmed rows only, via `_user_confirmed`. `auto_confirmed` is the detector accepting its own answer on a threshold — feeding that back would let a misread tier confirm itself and defend the slots it invented. Settled 2026-08-27 for the layout and profile loaders too, which had accepted them: measured at 2 of 2623 confirmed rows in one store, with no screenshot losing its layout, because the flag clears when the user accepts the row |
| When | `_is_trainer_call` only. WARP proper never reads annotations; the ARCHITECTURE RULE in `_process_image` is untouched |
| Visibility | both values go to the log, so a misread stays measurable instead of being papered over |
| Empty names | a confirmed bbox with no text says *where* the tier is, not *what* it is, and is ignored |

**How to revert.** Delete the `if _is_trainer_call:` block before
`resolution: ShipResolution | None = None`. `tests/test_confirmed_ship_info.py`
keeps passing — it tests the loader, which is independently useful — so the
revert is only visible in behaviour, not in a red suite. That asymmetry is
deliberate: the loader is the part worth keeping either way.

### Decision 2026-08-27: measured rows may raise an OCR tier, never lower it

**Change.** The X-bonus inference (`warp_importer._infer_x_bonus`) used to run
only when OCR found no tier at all. It now runs whenever the build is a space
one, and when the rows on screen hold more slots than the read tier grants,
the tier is raised to match.

**What is actually measured.** Not the badge — that is OCR's job and nothing
else reads it. `-X` grants +1 Universal Console, Device and Starship Trait,
`-X2` grants +2, so *counting those rows* measures the upgrade:

```
pixel_count(Devices)            - profile['Devices']
pixel_count(Universal Consoles) - profile['Universal Consoles']
detected(Starship Traits)       - profile['Starship Traits']
```

The profile already carries whatever the read tier granted, so each figure is
a surplus **over the tier the build claims**, not an absolute level. That is
what makes the same function serve both cases with one arithmetic.

**Why only upwards.** Every reading is a lower bound: a pixel count sees
filled slots, so an empty one makes the evidence too small and never too
large. Measuring *less* than the badge claims is therefore not evidence of a
lower tier — it is what an unfilled slot looks like. Raising is safe in the
same sense: a surplus cannot be produced by a player leaving something empty.

**A trap found while implementing it.** The Starship Traits figure used to be
measured against the constant `_BASE_STARSHIP_TRAITS = 5` rather than against
the profile. Those are the same number while the function only ran with no
tier read, and different the moment a read tier has already added its bonus:
a correct `-X` screen showing six starship traits would have read as one more
upgrade and promoted the ship to `-X2`. `tests/test_tier_bonus_inference.py`
holds that case specifically.

**How to revert.** Restore the `not ship_tier and` guard at the head of the
block. The tier then comes from OCR alone whenever OCR produced one, which is
the behaviour before this decision. The logs distinguish the two events —
`no tier on screen — inferred …` against `tier 'T6-X' (OCR) raised to …` — so
how often the raise fires, and how often it is wrong, can be counted before
deciding.

**Rejected: a font-height guard.** One line of text is one font, so a token
whose height differs from the anchor's looks like a different piece of UI —
the stray token is h=16 against the class line's h=24. Measured, it does not
work at token granularity: OCR box height depends on which glyphs a token
contains (ascenders/descenders), so the only bands loose enough to be safe
(≤ 0.6–1.6) do not catch a ratio of 0.67, and every tighter band costs a real
match — `h0.7-1.45` loses `image6.png` (`Exploration Cruiser Retrofit`).

**Rejected for now: "a ship bbox must not overlap a slot bbox".** Measured
after the cap across 10 folders: 18 meta bboxes, **0** overlapping a detected
slot bbox in 2-D. Before the cap, `Ragna/Untitled.png` had `Ship Type` over 10
slots. The rule would currently be dead code; a warning log would be worth
more than silent clamping, since the bbox also feeds crops (S2).

### Decision 2026-08-30: a tier badge cut in half is still a tier badge

**Change.** Anchor pass 1c re-joins
runs of two or three x-adjacent tokens in one OCR row and re-tests the result
for a closed `[...]` that `_fuzzy_tier` can snap. Two supporting changes make
the class line come out right once the badge anchors: `_adjacent_left_of` no
longer applies the column window and tolerates an overlapping neighbour, and
the row-above extension only runs when the tier's own row yielded nothing.

**Why.** On `l1.png` (694×622) OCR cut the badge itself:

```
x  36– 94  'Terran'                        conf 1.00
x 103–323  'exington Dreadnought Cruiser'  conf 0.79   ← dropped 'L'
x 318–353  '[Te-'                          conf 0.27   ← '[T6-'
x 351–383  'X2]'                           conf 0.49
```

Neither half carries a closed bracket, so pass 1b saw nothing; `'Te-'` has no
digit, so `RE_TIER_LOOSE` saw nothing either. The screenshot fell through to
the anchorless path, where `ship_type` became whichever single token won the
ShipDB lookup — `ship_type_bbox` was `(103, 37, 220, 18)`, missing `Terran` and
the whole badge — and `Ship Tier`, having no box of its own, borrowed that same
wrong one — see the `no badge on screen` comment in `_infer_x_bonus`'s
caller. Joined, `'[Te-' + 'X2]'` → `'[Te-X2]'` →
`T6-X2`, boxed at `(318, 30, 65, 25)`.

**Shortest run wins.** Sweeping by start index instead lets a long class-name
token join a badge it merely abuts: the triple
`'exington Dreadnought Cruiser' + '[Te-' + 'X2]'` snaps to the same `T6-X2` but
boxes 280 px instead of 65.

**Why the column window had to go from `_adjacent_left_of`.** `col_pad` is
derived from the anchor's own width (§2), which is right for a *wide* anchor
and useless for a narrow one: a 65 px badge opens the window at x=188, and the
class line it is meant to bound starts at x=36. `Terran` fell outside and the
class came back as one word short. The gap rule is the stronger locality test
anyway — same row, contiguous in x — so the window is now applied only where
it was designed to help.

**Why the row-above rule became conditional.** With the class line read off the
badge's own row, whatever sits above it is something else. Here it was the FPS
watermark `1.5.5. Pure Immersion` plus the `Fore` column header — neither is in
`_HUD_BLACKLIST` — and prepending them would have produced
`'1.5.5. Pure Immersion Fore Terran exington Dreadnought Cruiser'`. The rule is
for a class that OCR wrapped onto two lines, which is exactly the case where
the badge's row comes back empty.

**Measured.** Full OCR pass over 451 screenshots (`dev/shipinfo_corpus_run.py`,
`dev/shipinfo_corpus_diff.py`), before and after:

| | changed | gained | lost |
|---|---|---|---|
| `ship_type` | 1 | 1 | 0 |
| `ship_tier` | 1 | 1 | 0 |
| `ship_name` | 0 | — | — |
| `ship_*_bbox` | 2 | 2 (`None` → correct) | 0 |

The one screenshot that changed is `l1.png`. Nothing else in the corpus moves,
because the failure needs OCR to cut inside the bracket.

**A consequence, not a regression.** Filling in `ship_type` stops the screenshot
from taking the anchorless path, so it reaches `ShipDB.get_profile` — which
promptly returned the wrong ship, because the fuzzy match against the generic
`type` field ran ahead of the one against real class names. That is a separate,
pre-existing bug, fixed in `cf63086`; without it this change trades a bad box
for a bad ship.

**How to revert.** Delete the 1c block; the two supporting changes are then
unreachable for split badges but still apply to passes 1 and 1b, so revert them
too if the corpus is re-measured. `tests/test_tier_badge_split.py` holds the
real token dump for the case.

### Decision 2026-08-31: the review row shows the tier the build was sized from

**Change.** The `Ship Tier` item emitted for WARP CORE now carries the
resolved `ship_tier` — the same value ShipDB, the profile and the exporter
were given — instead of the raw `text_info['ship_tier']` the OCR pass
produced, and its confidence says which of the three sources answered.

**Why.** The two decisions above gave the tier two ways to change after OCR
read it, and neither reached the review row. A screenshot whose badge read
`T1` while the user's confirmed `T6` sized the grid offered `T1` at a flat
1.00 — above every auto-accept setting. Accepting it writes `T1` into
`annotations.json`, and because a confirmed tier now outranks OCR, the next
Auto-Detect sizes the grid from it. The header said one thing and the row
offered another:

```
WarpImporter: tier 'T1' (OCR) → 'T6' — confirmed by user, taken as truth
  Tier  : T6
RecognitionWorker:   slot='Ship Tier'   name='T1'   conf=1.00
```

Checkable without the source: the `Ship Tier` row in WARP CORE now reads the
same value as the `Tier  :` line of the resolution header above it in the
detection log. On that screenshot it reads `T6-X2` — the confirmed `T6` plus
the upgrade the measured rows found — marked as inferred and left pending.

**The grades.**

| Source | Confidence | Reasoning |
|---|---|---|
| `SHIP_TIER_CONF_CONFIRMED` | 1.0 | the user's own row; it *is* ground truth |
| `SHIP_TIER_CONF_BADGE` | 0.90 | measured, see below |
| `SHIP_TIER_CONF_INFERRED` | 0.45 | below the 0.50 auto-accept floor |

The badge figure is measured rather than chosen. Running the shipped
`extract_ship_info` + `refine_ship_info` over all 80 screenshots in one
maintainer's `annotations.json` that carry a user-confirmed `Ship Tier`, the
badge agrees with the user on 73 of the 79 it answered at all. The misses are
not spread evenly:

| read | confirmed | n | Verdict |
|---|---|---|---|
| `T1` | `T6` | 4 | genuine misreads |
| `T6-X2` | `T5` | 1 | genuine — `B'rel Bird-of-Prey Retrofit`, a T5 hull |
| *(nothing)* | `T6-X2` | 1 | badge not read at all |
| `T6-X2` | `T1` | 1 | **the store is wrong**, not OCR |

The last row was audited by eye: `1-ba54d6e861e08f02.png` carries a confirmed
`Ship Tier = T1` while its badge plainly reads `Verne Temporal Science Vessel
[T6-X2]`. Discounting it the rate is 74/79 — 93.7%, rounded down to 0.90 for
the sample size.

Four of the five real misses are a `[T6]` bracket read as `T1`, all on
screenshots narrower than 1700 px (`Kor Bird-of-Prey [T6]` on
`Kor-casual-17430cee006ca2ca.png` was checked against the pixels). That is the
case the grade exists for: `T1` is a legal tier, so nothing downstream rejects
it, and it costs slots.

The inferred grade sits below the floor deliberately. The raise is a lower
bound argued from pixel counts, and unlike the badge its accuracy has never
been measured — so it may reach the training data only through a human.
`src='inferred'` continues to mark the row in the UI, and now does so whenever
the raise fired, not only when the badge was off-screen.

**Also corrected.** The raise logged `tier 'T6-X2' (OCR) raised to T6-X2` —
it read `ship_tier` *after* the reassignment, so it printed the new value as
if OCR had read it. It now captures the previous value. The companion branch
announced `no tier on screen` whenever the badge carried no `-X` suffix, which
is most screens; it is now selected by whether a tier existed at all.

**How to revert.** Restore `_val = text_info.get('ship_tier', '') or
_inferred_tier` and the flat `1.0`. `tests/test_ship_tier_confidence.py`
drives `_process_image` itself, so four of its six tests fail on the revert —
the two that do not are the negative controls.

## Open questions

1. **`_HUD_BLACKLIST` does not gate name assembly.** `Kit Modules`,
   `Starship Selection Dry Dock` and `SHIP & CREW` survive as `ship_name`.
   Harmless today (S3 keeps the name out of the results), but it is the same
   class of leak as the one fixed in §2. Decide whether the name path should
   share the filter.
2. **`RE_NAME_PREFIX` is far too loose.** Of 264 matching tokens across the
   corpus, only 88 (33 %) carry two real separators; the rest are plain words
   such as `Dry Dock`, `Run Speed:`, `Set Active`. A ≤ 4-word cap on the
   candidate token was measured as zero-cost (111 → 111 matches, one text
   change on the whole corpus: `doff.png`) and is not applied — the deeper fix
   is a `DISCARD` screen class, which now exists but has no samples yet. See
   [`ML_PIPELINE.md`](ML_PIPELINE.md).
3. **Tightening the separator rule is not an option.** Requiring two real
   separators kills prefix-less names — 40+ of the 48 in the corpus, including
   every ground/character screen.

4. **A tier read too high cannot be corrected by anything but the user.**
   Raising is now allowed (see the decision below), lowering is not, because
   a shortfall in the measured rows is indistinguishable from slots the player
   left empty. A badge misread upwards therefore keeps its surplus rows until
   someone confirms the right tier in WARP CORE.
