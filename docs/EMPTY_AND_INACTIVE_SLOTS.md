# Recognising that a slot holds nothing

A build screen is a grid of slots, and many of them are empty or not yet
unlocked. Recognition has to say so. This document describes what the feature
is for, how it works today, how well it works, and what follows from that for
the training data.

## Why this is a problem at all

An icon matcher always answers. Given a crop it picks the nearest thing it
knows, and if everything it knows is an item, then an empty slot gets the name
of an item. That was the original failure: blank bridge-officer cells came
back as `Charged Particle Burst`, at high confidence, and were written into
builds.

Two ways out. Filter the blanks out of the data and hope they never reach the
matcher — or give the matcher the option of answering "nothing". The project
took the second: `__empty__` and `__inactive__` are ordinary labels
end-to-end. Users confirm them in WARP CORE, they upload with everything else,
the mergers keep them, and the embedder carries them as gallery classes. Their
whole job is to be the answer *not an item*.

## How a cell is judged today

Two mechanisms, in this order:

```
     crop of a slot cell
             │
             ▼
   LayoutDetector._classify_cell        fixed rule: how bright, how even,
   "is this cell blank?"                how blue — no model, no training
             │
      ┌──────┴───────┐
   blank          occupied
      │               │
      ▼               ▼
  __empty__ /    SETSIconMatcher.match  the models: embedder, templates,
  __inactive__   "which item is it?"    session examples — and they can
  (matching                             still answer __empty__ /
   never runs)                          __inactive__
```

The rule reads the inner 60% of the cell in HSV and keys on what the game
actually draws. The property it keys on is **how much the cell varies**, not
how bright or how blue it is, and that is what lets one rule cover markings
that look nothing alike:

| What the game draws | What the cell is |
|---|---|
| nothing — dark and even, or the navy panel showing through | empty |
| a navy fill with an X across it | inactive |
| a `LOCK` word, a padlock, or a level requirement on near-black | inactive |
| a picture | active |

An unpainted cell is flat whatever its colour; every locked marking, X or
word or padlock, adds variance. So the rule splits twice on brightness
variance — once inside the blue-saturated navy window, once outside it — and
the hue and saturation tests only decide which of the two splits applies.

When it says blank, the cell is settled and no matching runs — which is also
why a screenshot resolves in seconds rather than minutes.

When it says occupied, the models decide, and "nothing" is still on the ballot
because it is a class like any other. The rules that stop a false "nothing"
from beating a real icon live in `SETSIconMatcher.match` and are mapped in
[`client_user_view_filter.md`](client_user_view_filter.md).

## How well it works

Measured 2026-09-03 against every user-confirmed crop in one maintainer's
training store — 732 blank cells and 5833 real icons (1500 of them sampled
for the model rows).

| Decider | blank cells called blank | real icons called blank |
|---|---|---|
| the rule | 717/732 — **98.0%** | 0/1500 — **0.00%** |
| the models, on their own | 732/732 — **100%** | 0/1500 — **0.00%** |

The model rows are measured with the session pool empty, so that a crop
cannot match itself. What remains is the shipped gallery, which is built from
community crops and may include this install's own uploads, so treat them as
an upper bound rather than a holdout.

| | cells |
|---|---|
| both agree the cell is blank | 717 |
| only the rule sees it | 0 |
| only the models see it | 15 |
| **neither sees it** | **0** |
| | **732/732 = 100%** |

The rule misses 15 cells that sit just outside its thresholds — locked cells
whose brightness varies a little more than expected, and empty device slots
drawn over a bright background — and the models catch every one. The pipeline
consults the rule first and the models second, so it takes the union either
way.

**These figures changed on the same day.** Before the mislabelled crops
described below were removed and the embedder retrained, the models missed 22
cells the rule caught, and the two looked complementary rather than one
strictly better than the other. The rule's numbers did not move, because the
rule does not learn.

### Telling the two blanks apart

The table above asks one question — was a blank cell called blank — and
counts `__empty__` and `__inactive__` together. It does not ask whether the
rule picked the right one of the two, and until 2026-09-04 nobody had
measured that.

Measured on a different corpus, so the two sets of figures are not directly
comparable: the 985 confirmed virtual crops in the published community
dataset, plus a 1200-crop sample of its real icons. Every figure comes from
running `LayoutDetector._classify_cell` itself, not a copy of it.

| | before | after | labels corrected |
|---|---|---|---|
| blank cells called blank | 96.6% | 98.3% | **98.4%** |
| real icons called blank | 0.00% | 0.00% | **0.00%** |
| of the blanks, `__empty__` called empty | 75.8% | 87.1% | **89.5%** |
| of the blanks, `__inactive__` called inactive | 96.2% | 98.2% | **98.4%** |

The first row is the same question the table above asks, on this corpus. The
last two are the new one.

The third column is the same code against the labels as they stand after the
review described under "The residual misses" — six crops were wrong, not the
rule. It is listed separately rather than merged into the second, because
moving a threshold and correcting a label are different acts and only the
first is a change to the program.

Getting the pair wrong is far cheaper than missing a blank altogether — both
labels stop matching and neither invents an item — which is why the coarse
figure was the one worth having first. The distinction still matters for
training: the embedder learns `__empty__` and `__inactive__` as separate
classes, so a flat navy cell filed under the wrong one teaches it that the
locked marking is optional.

Two thresholds moved, both chosen from the measured distributions rather than
tuned to the sample:

- Inside the navy window a flat fill now reads as empty rather than inactive.
  A locked cell carries the X and varies; an empty seat on the same panel is
  the background showing through and does not.
- The cut that separates a locked cell from a real icon was set below the
  dimmest real icon in the corpus, but not at it — the gap was left wider
  than the sample strictly requires, so an icon dimmer than any seen so far
  is still read as an icon.

The damaging direction — a blank cell called active, which then gets given an
item name — fell by half.

## The defect this measurement found, and what removing it did

Before the cleanup, all 22 model misses came back as `Charged Particle Burst`. That is not a
coincidence and it is not a limit of the model:

| an inactive bridge-officer cell, compared with | cosine similarity |
|---|---|
| genuine crops of Charged Particle Burst | 0.45 |
| the wiki artwork for that ability | 0.38 |
| crops **labelled** Charged Particle Burst that are in fact blank cells | 0.92 |

The model separates the ability from a blank cell easily. What it matched at
0.92 was a picture of a blank cell filed in the dataset under the ability's
name. Of the 29 crops that class has in the published dataset, **20 are
inactive bridge-officer cells**.

The loop is self-feeding: the recogniser guesses that name on a blank cell,
someone confirms the guess, the crop uploads under that name, the gallery
learns that a blank cell is that item, and the next guess is the same one.

It survived because every guard in the system looked one way. The seed filter
`_virtual_crop_looks_real`, the backend's review tool and its monthly audit
all search for a *colourful* crop under a blank label. Nothing searched for a
*blank* crop under an item's name, which is the more damaging direction —
across the whole dataset it is 25 crops in 9227 (0.27%), but concentrated
enough to take a class over.

Two guards now close it, both refusing to *learn* from such a crop rather than
deleting anything:

- `SETSIconMatcher.add_session_example` refuses a blank crop carrying a real
  item's name, logging the rejection. Every seeder and WARP CORE's accept
  path go through it.
- WARP CORE's auto-accept skips such a row, so the name needs a human. This
  mirrors the rule that already stopped auto-accept from confirming a blank
  label that came from a session match.

Both delegate the judgement to `_real_crop_looks_blank`, which asks
`LayoutDetector._classify_cell` — the same function the pipeline uses. One
definition, so the guard refuses exactly what the pipeline would call blank.
The cost is the rule's own error rate: 2 crops in 5833 do not become session
examples, and their classes have hundreds of others.

The 25 crops already in the dataset were a separate, deliberate cleanup. The
backend's review tool (`admin_reject_crops.py`) scans both directions —
`--direction real` surfaces them for keep / reject / relabel through the same
ledger and montage as before — and they were rejected on 2026-09-03.

**Retraining on the cleaned data closed it.** The 22 confusions are gone: the
embedder now names every one of the 732 blank cells correctly, including the
15 the fixed rule misses, and calls no real icon blank. That is the whole of
the change — the model was never the limitation.

## What this means for collecting more

Blank cells are the one label where more examples buy nothing measurable:

- The rule decides 98% of cases and does not read examples at all. No amount
  of data moves that number.
- The models are now correct on all 732, so there is no error left for more
  examples to remove. What fixed the ones that were wrong was deleting 20 bad
  labels, not adding good ones.
- The store holds ~1400 examples (773 confirmed locally, 645 in the community
  pool), which is far past what a nearest-neighbour reference needs.

So: keep accepting them — a user must be able to mark a slot empty, and that
is how the reference stays current when the game's art changes — but there is
nothing to gain from collecting them in bulk. What matters for these labels is
that they are *correct*, not that there are many: one wrong crop in a class of
29 was enough to take the class over.

Collection becomes worth it again when a **new appearance** shows up — a panel
type or a UI scale we have no example of. That is a gap in coverage, not in
volume.

## Where these labels are never accepted

One place refuses them by design: `knowledge.json`, the community table that
maps a crop's perceptual hash straight to a name. An entry there is a hard
override at full confidence, so a blank cell recorded under any name would
force that answer on every matching crop. The backend's merge drops every
`__*` name unconditionally, the client suppresses such an override on read,
and the client does not upload them in the first place.

That is not a contradiction of the idea above. Training data and the override
table answer different questions: one teaches a model what nothing looks like,
the other asserts an identity. See
[`client_user_view_filter.md`](client_user_view_filter.md) for the full map of
where these labels are stopped.

## What is left

**Does the fixed rule still need to answer first?** On this corpus the models
alone are now correct on all 732 blank cells and on all 1500 real icons, so
the rule adds no accuracy. It still earns its place on two other grounds, and
both would have to be measured before removing it:

- *Cost.* It settles roughly 700 of the 732 without an inference. Every cell
  it hands to the models is about a quarter of a second, so removing it turns
  a screenshot that resolves in seconds into one that takes minutes.
- *Generalisation.* It reads what the game draws — dark and even, or dim navy
  — rather than what it has been shown. A rule has nothing to forget when the
  next screenshot arrives at a resolution nobody has contributed, and the
  measurements above are an upper bound on the models precisely because the
  gallery may contain this install's own crops.

Neither is an argument against the models. They are reasons the question is
about cost and coverage now, not about accuracy.

**The residual misses.** The largest cluster described here until 2026-09-04
was locked cells whose brightness variation ran past the cut that separated
them from real icons. That cut has since been widened, on the measurement in
"Telling the two blanks apart" — the concern raised here, that widening
risks dim blue icons, was the thing measured: on that corpus the gap between
the most-varying locked cell and the dimmest real icon is real but narrow,
and the new cut sits inside it with room to spare on the icon side.

What remains is a different shape and is not a threshold problem. Every
disagreement between the rule and a stored label was reviewed by eye on
2026-09-04; these are the three kinds found:

- **An empty slot wearing the game's NEW banner.** A device slot the player
  has not filled still gets the yellow `NEW` tag drawn across its top third.
  The rule samples the inner 60% of the cell, which trims 20% from each edge
  — not enough to clear a banner that deep, so the strip that survives is
  bright, patterned and yellow, and the cell reads as occupied. The label is
  right and so is the picture; the rule simply cannot see past the banner.
  Four such crops, all in `Devices`.

  Sampling the lower 60% instead would clear it, since the banner is always
  at the top. That change touches every cell the program classifies, so it
  needs its own measurement over the whole corpus rather than a fix made in
  passing.

- **Crops that are cut off centre.** A navy cell whose X sits partly outside
  the frame, or whose frame catches a slice of the bright cell next door,
  reads as a picture because it contains one. Moving a threshold cannot fix
  a crop that shows the wrong thing.

- **Ground truth that was wrong.** Three crops confirmed as `__empty__`
  carried an unmistakable navy square with a diagonal X, and one `__empty__`
  had four votes behind it — the wrong convention repeated by four separate
  installs rather than slipped in once. Three further crops were not slot
  cells at all. All six were corrected or rejected on 2026-09-04, so the
  figures above already reflect the labels as fixed.

The models catch what the rule does not, so nothing is lost at recognition
time either way — the pipeline takes the union.
