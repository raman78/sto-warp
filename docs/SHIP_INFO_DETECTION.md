# Ship info detection

Production module: `warp/recognition/text_extractor.py`, entry point
`TextExtractor.extract_ship_info` (`warp/recognition/text_extractor.py:473`).
It reads three fields off the top band of a space screenshot — ship **name**,
ship **class** (called `ship_type` throughout the code) and **tier** — plus a
bounding box for each. `ShipDB.resolve` (`warp/warp_importer.py:1085`) then
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
`SHIP_INFO_ROI` (`:128`) exists but is only used by the re-OCR fallback in
`refine_ship_info` (`:1141`).

Instead the extractor finds one token it can trust (the **anchor**) and reads
the rest of the block relative to it. Everything downstream — which tokens may
join the class string, which row holds the name — is expressed as a constraint
against that anchor.

## Invariants

- **S1** — A `ship_*_bbox` is the union of exactly the tokens that went into
  the corresponding string. Nothing is added for padding, nothing is dropped.
  A bbox that spans emptiness therefore means the string is wrong too.
- **S2** — The bboxes are consumed, not merely displayed: `warp_importer.py:1951`
  emits `Ship Type` and `Ship Tier` as review items with those bboxes, the
  trainer uploads the corresponding image strips as text crops
  (`warp/trainer/sync.py:117`), and `refine_ship_info` re-OCRs from them. A
  wrong bbox is a wrong training sample, not just a cosmetic defect.
- **S3** — `Ship Name` is anchor-internal. It is never emitted as a slot
  (`warp_importer.py:1948`), because it identifies the player, not the build.
- **S4** — A non-empty `ShipResolution.type` does **not** mean the class was
  recognised. When nothing matched, the OCR string is echoed back;
  `ShipResolution.matched` (`warp_importer.py:1303`) is the only honest signal.

## 1. Anchor strategies

Two, tried in order, both inside `extract_ship_info`:

| Kind | Trigger | Anchor token | Set at |
|---|---|---|---|
| `tier` | A `[T5]`/`[T6-X2]`-style bracket found by `RE_TIER_LOOSE` | the token carrying the bracket | `:729` |
| `name` | A token that looks like a ship-name line, per `_is_name_prefix_token` (`:62`) | that token | `:747` |

With a tier anchor the class is assembled from the tier row and the row above
it; with a name anchor, from the row below the name. No anchor means no ship
info at all (`:971`) — the extractor emits `anchorless_candidates` instead and
`ShipDB.find_class_by_candidates_ex` gets a last-resort attempt.

`_is_name_prefix_token` is misleadingly named: it does not require a prefix.
`U.S.S. ENTERPRISE` matches, and so does a bare `Henrik Lindstrom`, which is
why ground/character screens produce a name at all. Measured on the 254-shot
corpus, 48 screenshots have a `ship_name` with no `U.S.S.`-style prefix.

## 2. The column window

The HUD stacks name / class / registry in one left-aligned column, so the
extractor only accepts tokens inside a horizontal band around the anchor
(`:769`):

```python
# warp/recognition/text_extractor.py
col_pad = max(80, int(min(anchor_w, _COL_PAD_ANCHOR_CAP) * 2.0))
col_lo  = anchor_x - col_pad
col_hi  = anchor_x + anchor_w + col_pad
```

`_COL_PAD_ANCHOR_CAP = 150` (`:135`) is the part worth explaining. The pad used
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
| Same row | `_adjacent_left_of` (`:833`) | tokens left of the tier token, contiguous in x; stops at a gap > `max(40, 4 × token height)` |
| Row above | `_row_to_type` (`:816`) | whole row, if it is in the column, is not a name row, and holds no ALL-CAPS proper noun |
| Token filter | `_valid_type_tok` (`:779`) | length > 2, not in `_HUD_BLACKLIST` (`:85`), not a section header, not a registry number, inside the column |

`_HUD_BLACKLIST` gates the class only. Name assembly does not consult it,
which is how `Kit Modules` and `Starship Selection Dry Dock` survive as
`ship_name` on some screenshots.

## 4. Resolution against the ship database

`ShipDB.resolve` runs a four-strategy lookup and promotes the canonical class
string from the DB over the OCR string whenever a real entry was found. The
strategy that won is reported in `last_match_strategy` — `exact-type`,
`word-subset`, `display-name`, `fuzzy-type`, `fuzzy-display`, `token-overlap`,
or `keyword-fallback` for "nothing matched" (`warp_importer.py:828`-`920`).

This layer is forgiving by design, which is why a polluted class string often
still resolves correctly, and why measuring the extractor by string equality
overstates its errors. It is also why it can resolve to the *wrong* ship: on
`Screenshot_96.png` the junk token `Decails` (OCR of the **Details** button)
pulled `Legendary Dreadnought Cruiser` towards
`Universe Temporal Heavy Dreadnought Cruiser`.

## Failure modes

| Symptom in the log | Cause | Where to look |
|---|---|---|
| `type='<junk> <real class>'` | a token from another panel entered the column | `_valid_type_tok` / column window, §2 |
| `ship_type_bbox` far wider than the class line | same cause — the bbox is the union of what got in (S1) | §2 |
| `[name anchor]` on a screen with no ship | `RE_NAME_PREFIX` (`:39`) matches a plain word: dots are optional and the separator class accepts a space, so `Die E` matches | §Open questions, item 2 |
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

**Change.** `_fuzzy_tier` (`warp/recognition/text_extractor.py:102`) picks the
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
(`warp/warp_importer.py:699`) grants +1 Universal Console, Device and Starship
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

**Not covered by this change.** A tier the OCR *did* read stays authoritative
against pixel evidence: `_infer_x_bonus` only runs when no tier was found at
all (`warp_importer.py:1917`). What a misread no longer survives is the user —
see the decision below.

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

4. **Pixel evidence still cannot contradict an OCR tier.** `_infer_x_bonus`
   runs only when no tier was read at all, so a misread nobody corrects keeps
   its surplus rows. Letting the measured counts argue with OCR needs care:
   they are lower bounds — a row is only as full as the player left it — which
   is why they are trusted today only in the absence of any other evidence.
