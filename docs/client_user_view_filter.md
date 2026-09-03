# Virtual labels in the client — where they are stopped

Scope: the client side of invariant **Z5** from
[`data_source_audit.md`](data_source_audit.md) (in Polish). Z5 says that the
virtual classes — `__empty__`, `__inactive__` and the `__boff_*` row hints —
are legitimate labels for the ML pipeline but must never reach the user: not
as an item name in an exported build, not as a thumbnail, not as text in the
review panel.

After decision D-A.1 the training set is one shared pool, so `__*` is allowed
*into* the data. Z5 therefore has to be satisfied on the way **out**: at each
point where recognition output is presented to a user or written to a build.
This document is the map of those points, the reasoning behind each, and the
decisions taken about them. It is the closure of post-audit TODO #1 (D-H.7).

`dev/` is the maintainer's local working set — it is gitignored in its
entirety and is not in a checkout. Paths under `dev/` here are pointers for
reproducing a measurement, not scripts to run from the repo.

---

## 1. Which labels this is about

| Label | Produced by | Means |
|---|---|---|
| `__empty__` | cell pre-classifier, embedder gallery, user confirmation | the slot exists and holds nothing |
| `__inactive__` | same | the slot is not unlocked at this rank / tier |
| `__boff_<profession>` | `LayoutDetector` only | "a Tac/Eng/Sci ability could sit in this row" |

The first two are real classes: the models are trained on them, users confirm
them in WARP CORE, and they travel to the community dataset. The third never
becomes an item name — see §2.5.

---

## 2. The map

### 2.1 Read path — the community knowledge override

`SETSIconMatcher.match` consults the community pHash table first, and that
table is the one input the client does not control. Two guards sit there:

| Guard | Behaviour |
|---|---|
| virtual / test-entry suppression | an override naming `__*` or `Test Item Name` is skipped, and matching falls through to the ML and template stages |
| embedder cross-check | when the override claims a real item but the embedder answers `__*` at conf ≥ `VIRTUAL_OVERRIDE_CONF`, the override is suppressed |

This is defense-in-depth at the model level, independent of what the backend
accepts. It exists because a poisoned entry used to turn a real icon into an
empty slot at confidence 1.00.

### 2.2 Read path — the anti-virtual rules in the combine stage

This is the core of Z5 for the user. After the knowledge, ML, template and
session stages have each produced a candidate, `SETSIconMatcher.match`
decides whether a virtual answer is allowed to win. Four independent rules
suppress it; any one firing is enough:

| Rule | Fires when | Constant |
|---|---|---|
| ML is confident and real | the embedder names a real item at conf ≥ 0.40 | `VIRTUAL_OVERRIDE_CONF` |
| poison guard | session returns a virtual at ≥ 0.95 (a pixel-perfect self-match) while the embedder names any real item at ≥ 0.15 | `SESSION_PIXEL_PERFECT`, `POISON_GUARD_ML_MIN` |
| embedder margin | the best real gallery row beats the best virtual row by ≥ 0.05, whatever the absolute confidence | `EMBED_REAL_VS_VIRTUAL_MARGIN` |
| query sanity | the crop is itself bright and colour-rich (> 0.15 on both) while session or template answered virtual | `VIRTUAL_SEED_BRIGHT_RATIO`, `VIRTUAL_SEED_RICH_RATIO` |

The last three announce themselves in `warp_detection.log` as
`poison-guard fired`, `embed-margin guard fired` and `query-sanity guard
fired`, each naming the scores that triggered it. That is how the rules can be
checked against a running program rather than against this table.

If no rule fires, a virtual answer *may* win and is returned to
`WarpImporter` — which is correct, because an empty slot really is empty most
of the time. Everything downstream of §2.3 assumes this and filters at the
point of use.

`SETSIconMatcher._thumb_for_name` returns nothing for a virtual name, so no
reference picture is ever shown for one.

### 2.3 Seed path — what may become a session example

How well the blank/occupied judgement itself works, and what these labels are
for, is [`EMPTY_AND_INACTIVE_SLOTS.md`](EMPTY_AND_INACTIVE_SLOTS.md); this
section covers only what is allowed into the pool.

Confirmed crops are seeded into the in-memory k-NN pool by
`SETSIconMatcher.seed_from_training_data` and `seed_from_community_crops`.
Two filters apply, and they answer different questions:

| Filter | Rejects | Why |
|---|---|---|
| `_virtual_crop_looks_real` | a crop labelled `__*` that is bright and colour-rich | a real icon mislabelled as empty; seeding it makes it self-match forever |
| `_real_crop_looks_blank` | a blank cell carrying a real item's name | the mirror, and the more damaging one: it teaches the gallery that the item is what nothing looks like. Added 2026-09-03 after 20 of the 29 community crops of `Charged Particle Burst` turned out to be inactive BOFF cells |
| `SETSIconMatcher._template_is_degenerate` | a crop of one flat colour, any label | `TM_CCOEFF_NORMED` divides by the template's standard deviation, so a constant template scores exactly 1.00 against every query |

Neither filters virtual as a *class*. A genuine `__empty__` crop — dim,
uniform but not perfectly flat — is seeded, and has to be: the anti-virtual
rules above compare a real candidate against a virtual one, and they need the
virtual side to exist.

The flat-colour rejection was added 2026-08-31 after two pure-black
`__empty__` crops in the community pool were found to offer every real icon
`__empty__` at ≈ 0.80–0.85. Full measurement in
[`ML_PIPELINE.md`](ML_PIPELINE.md) §"A session example must have structure".

### 2.4 Write path — the build

Virtual names must not reach a build planner. `warp/build_writer.py` defines
`VIRTUAL_ITEM_NAMES` and gates on it at every write:

| Construct | Role |
|---|---|
| `_write_equipment_and_traits` | skips a slot whose name is empty or virtual — the gate that keeps `__empty__` out of the exported build |
| `_match_clusters_to_seats` | counts only non-virtual items when deciding how full a BOFF seat is |
| `_write_abilities` | treats a virtual name as an unfilled ability slot for the rank check |

`warp_importer._recog_score` is not a filter but is worth knowing about: a
virtual match below `IMPORTER_CONFIDENT_VIRTUAL_THRESHOLD` counts at half
weight in the reported recognition score, so an uncertain "nothing here" does
not inflate the number.

`boff_keys._seat_label_from_items` skips virtual names (and the empty string)
via `_VIRTUAL_NAMES` when it collects the abilities that describe a seat, so a
seat is never labelled from an empty cell.

### 2.5 `__boff_*` never leaves the layout detector

`LayoutDetector._detect_via_full_scan` tags a detection with
`__boff_<profession>` as its *item type*, and `_score_row_for_slot` reads that
tag when scoring a row against a candidate slot. `_get_item_type` returns the
same form. Nothing outside `layout_detector.py` reads these names — they are
row hints on a detection tuple, never an item name — so no downstream filter
is needed for them, and none exists.

### 2.6 Trainer UI — display, and deliberate availability

WARP CORE shows recognition output to the user, so a raw `__empty__` would be
both ugly and ambiguous. Two behaviours, and only the first is a filter:

- `_review_row_visuals` renders `[empty slot]` / `[inactive slot]` instead of
  the literal name.
- The virtual names are deliberately **offered** as labels: they appear in the
  rematch candidate lists and in the item selector, because a user must be
  able to say "yes, this slot is empty". Without that there would be no
  ground truth for these classes at all.

One protection sits between the two: `_finish_bbox_drawn` refuses to
auto-accept a virtual name that came from a session match. That is the
self-poisoning path — a virtual is written, becomes a session example, and
self-matches from then on — so it requires a human.

### 2.7 Upload path — the only input-side filter

`WARPSyncClient` carries `_is_poison_label`, gated by the module-level flag
`_POISON_FILTER_ENABLED`. When enabled, a contribution whose label starts with
`__` or equals `Test Item Name` is dropped before the POST.

This filter does not protect the user view. It protects nothing on this side
at all: it exists to mirror the backend's own policy so the client does not
spend its daily contribution budget on requests the backend will refuse. Its
current state is the subject of §4.

### 2.8 Maintainer tools

`warp/tools/scrub_training_data.py` (offline poison scrub, with a visual
review mode) and `warp/tools/conflict_reviewer.py` operate on the training
store. They are maintainer-only and do not run in the user's session.

---

## 3. Classification

**Protecting the user view** — these are what close Z5 on the output side:

1. Knowledge-override suppression and the embedder cross-check (§2.1).
2. The four anti-virtual rules in the combine stage (§2.2).
3. The two seed-time crop filters (§2.3).
4. `_thumb_for_name` returning nothing for a virtual name (§2.2).
5. The three write gates in `build_writer` (§2.4).
6. `_seat_label_from_items` in `boff_keys` (§2.4).
7. `_review_row_visuals` in the trainer (§2.6).

**Not user-view filters** — legitimate internals, or subject to §4:

- `_is_poison_label` in `sync_client` — upload eligibility, not presentation.
- `training_data.py` and `embedder_trainer.py` — ML training data, where `__*`
  is legal end-to-end per D-A.1.
- `__boff_*` production in `layout_detector` — internal row hints (§2.5).
- `scrub_training_data.py`, `conflict_reviewer.py` — maintainer-only.

---

## 4. Decisions

Decision IDs are referenced from `data_source_audit.md` and from
`sync_client.py`; they are kept stable even where the finding has moved on.

| # | Decision | Status |
|---|---|---|
| Z5-C.1 | The output-side defence is layered and sufficient: no `__*` reaches a build, a thumbnail, a BOFF seat label or a recognition result on a real-looking icon. | Holds. Re-verified 2026-08-31; the combine stage now has four rules rather than the three mapped in June, and a fifth filter at seed time. |
| Z5-C.2 | The client's upload filter is redundant *for user safety* — the knowledge-override suppression catches `__*` on the read path regardless of what the backend stores. | Holds. It is not redundant for budget: see Z5-C.7. |
| Z5-C.3 | The client filter and the backend filter must be flipped together, never one alone. | **Currently violated** — see Z5-C.7. |
| Z5-C.4 | The anti-virtual thresholds are calibrated and must not be changed without evidence. `VIRTUAL_OVERRIDE_CONF` 0.40, `POISON_GUARD_ML_MIN` 0.15, `SESSION_PIXEL_PERFECT` 0.95, `VIRTUAL_SEED_BRIGHT_RATIO` / `VIRTUAL_SEED_RICH_RATIO` 0.15 each, `EMBED_REAL_VS_VIRTUAL_MARGIN` 0.05. The seed ratios were raised from 0.07 on 2026-07-17 after a visual review of community crops (`dev/diag_view_community_poison.py`): genuine empty and inactive BOFF slots reach ~12% bright/rich, real mislabelled icons ≥ 19%. | Holds. `tests/test_virtual_crop_guard.py` pins the seed ratios. |
| Z5-C.5 | The `[empty slot]` / `[inactive slot]` rendering and the availability of virtual names as manual labels both stay. Removing either would leave the classes with no ground truth. | Holds. |
| Z5-C.6 | `__boff_*` generation in the layout detector stays. | Holds, and on firmer ground than in June: these labels never leave `layout_detector.py`, so they do not depend on a downstream filter (§2.5). |
| Z5-C.7 | **D-B.3 was implemented and rolled back.** The client filter was disabled on 2026-06-09 and re-enabled the next day (`76bff9f`) because the deployed backend answered HTTP 400 to every virtual label, burning the contribution budget. | Open — see §5. |

---

## 5. Current state and what is unresolved

The two flags are out of step:

| Side | Flag | Value on disk |
|---|---|---|
| client | `_POISON_FILTER_ENABLED` in `warp/knowledge/sync_client.py` | `True` — filtering |
| backend | `_POISON_FILTER_ENABLED` in `sets-warp-backend` `main.py` | `False` — accepting |

Z5-C.3 requires these to match. The asymmetry is in the safe direction — the
client is the stricter of the two, so nothing extra reaches a user — but it
means user confirmations of `__empty__` and `__inactive__` are still not
reaching community knowledge, which is what D-A.1 and D-B.3 were for.

The rejections that caused the June rollback came from a Space running a build
older than the backend change; pushing to the backend's `main` did not deploy
it. That was fixed on 2026-07-17 by an auto-deploy workflow, so the reason for
the rollback is very likely gone.

**Open question.** Does the deployed backend now accept a virtual label? It
cannot be answered read-only — `/health` does not report the flag, and the
only way to know is one deliberate contribution with a virtual label,
observing whether the response is 200 or 400. Until that is done, the client
flag stays `True`: flipping it on the strength of a source file in another
repository is what produced the June rollback. When it is answered, flip both
flags in one change per Z5-C.3, and record the result here.

---

## 6. Out of scope

- Re-tuning the anti-virtual thresholds — frozen per Z5-C.4.
- The recognition stage architecture itself.
- The parent audit `data_source_audit.md`, which is in Polish and covers the
  whole data flow rather than this one boundary.

---

## 7. Log

- 2026-06-09: document created. Eight user-view filters mapped, decisions
  Z5-C.1..Z5-C.6, D-B.3 unblocked.
- 2026-06-10: D-B.3 rolled back on the client (`76bff9f`) — the deployed
  backend rejected virtual labels. Not recorded here at the time.
- 2026-08-31: rewritten in English and re-verified against the code. Three
  claims were wrong: the build gate is `_write_equipment_and_traits`, not
  `_apply_alien_species`; the combine stage has four suppression rules, not
  three; `__boff_*` labels never leave the layout detector, so they do not
  rely on a downstream filter. Added the flat-template seed filter, the
  current flag divergence as Z5-C.7, and §5.
