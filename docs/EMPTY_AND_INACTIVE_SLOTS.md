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
actually draws: an empty slot is dark and even; a locked bridge-officer cell
is dim navy with a faint X; an unlocked one has a bright, detailed picture.
When it says blank, the cell is settled and no matching runs — which is also
why a screenshot resolves in seconds rather than minutes.

When it says occupied, the models decide, and "nothing" is still on the ballot
because it is a class like any other. The rules that stop a false "nothing"
from beating a real icon live in `SETSIconMatcher.match` and are mapped in
[`client_user_view_filter.md`](client_user_view_filter.md).

## How well it works

Measured 2026-09-03 against every user-confirmed crop in one maintainer's
training store — 732 blank cells and 5833 real icons.

| Decider | blank cells called blank | real icons called blank |
|---|---|---|
| the rule | 717/732 — **98.0%** | 2/5833 — **0.03%** |
| the models, on their own | 710/732 — 97.0% | 1/1500 — 0.07% |

The second row is measured with the session pool empty, so that a crop cannot
match itself; what remains is the shipped gallery, which may still hold this
install's own uploads, so it is an upper bound.

Neither number is the interesting one. This is:

| | cells |
|---|---|
| both agree the cell is blank | 695 |
| only the rule sees it | 22 |
| only the models see it | 15 |
| **neither sees it** | **0** |
| | **732/732 = 100%** |

The two fail on disjoint sets. The rule misses cells that sit just outside its
thresholds — locked cells whose brightness varies a little more than expected,
and empty device slots drawn over a bright background. The models missed 22
cells, and every one of them for the same reason, described next.

Since the pipeline consults the rule first and the models second, it already
takes the union of the two. That is why the arrangement exists, and why
neither half is redundant.

## The defect this measurement found

All 22 model misses came back as `Charged Particle Burst`. That is not a
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

The 25 crops already in the dataset are a separate, deliberate cleanup. The
backend's review tool (`admin_reject_crops.py`) now scans both directions —
`--direction real` surfaces them for keep / reject / relabel through the same
ledger and montage as before.

## What this means for collecting more

Blank cells are the one label where more examples buy nothing measurable:

- The rule decides 98% of cases and does not read examples at all. No amount
  of data moves that number.
- The models handle the remaining 2%, and their class for "nothing" is
  already strong — it matches the very cells that were failing at 0.92. The
  failures were mislabelled data, not missing data.
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

- The rule's 15 residual misses are two tight clusters against its thresholds:
  locked cells whose brightness variation runs to 28 against a cut at 20, two
  whose hue sits a point below the navy band, and four empty device slots on a
  bright background. Widening the cuts risks dim blue icons, so it needs its
  own measurement before anything changes.
- The 25 mislabelled crops in the published dataset are waiting on a review
  pass with the tool above.
- Once they are cleaned, the models-alone figure is worth re-measuring: the
  22 failures should disappear, and only then is it meaningful to ask whether
  the fixed rule still needs to answer first.
