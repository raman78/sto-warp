# Data Lifecycle — user confirmation to delivered model

What happens to a piece of training data from the moment you click **Accept**
in WARP CORE until it comes back to every install as part of an updated
recognition model. Covers the four upload channels, the staging/data split
on HuggingFace, the four democratic mergers, the staging drain, and the
audit safety net.

This is the **client-side view** of an architecture whose write side lives
in `sets-warp-backend`. For backend-internal details (merger source code,
admin scripts, HF-token handling) see the backend's
[`docs/technical_overview.md`](../../sets-warp-backend/docs/technical_overview.md).

---

## 1. The full picture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                              USER MACHINE                                │
│                                                                          │
│   WARP CORE confirmation (Enter / autocomplete pick / Accept / Auto≥)    │
│             │                                                            │
│             ▼                                                            │
│   ~/.local/share/warp/training_data/                                     │
│       annotations.json + crops/<sha>.png  +  screen_types/<TYPE>/<sha>.png│
│             │                                                            │
│             │  SyncWorker: every 10 min, ≤1000 uploads/day per install   │
│             ▼                                                            │
└─────────────┼────────────────────────────────────────────────────────────┘
              │
              │   HTTPS POST  (server-side HF_TOKEN, never on client)
              ▼
┌──────────────────────────────────────────────────────────────────────────┐
│       sets-sto-warp-backend.hf.space  (Render-hosted FastAPI Space)      │
│                                                                          │
│   /contribute              ─►  contributions/YYYY-MM-DD/<uuid>.{png,json} │
│   /contribute/bulk-crops   ─►  staging/<install_id>/crops/<sha>.png       │
│                                staging/<install_id>/annotations.jsonl    │
│   /upload/screen-types     ─►  staging/<install_id>/screen_types/<T>/    │
│   /upload/anchors          ─►  staging/<install_id>/anchors_grid_*.json  │
│   /upload/sets-gaps        ─►  sets_gaps/<install_id>.json               │
└─────────────┼────────────────────────────────────────────────────────────┘
              │
              │   per-install HF dataset writes (one commit per batch)
              ▼
┌──────────────────────────────────────────────────────────────────────────┐
│          HuggingFace — sets-sto/sto-icon-dataset (RAW / staging)         │
│                                                                          │
│   staging/<iid_1>/crops/<sha>.png       ┐                                │
│   staging/<iid_2>/crops/<sha>.png       │ everybody writes here          │
│   staging/<iid_N>/crops/<sha>.png       │ raw votes accumulate           │
│   staging/<iid_*>/annotations.jsonl     │ until a merger runs            │
│   staging/<iid_*>/screen_types/…        │                                │
│   staging/<iid_*>/anchors_grid_*.json   ┘                                │
│                                                                          │
│   contributions/<date>/<uuid>.{png,json}  ← phash knowledge overrides    │
│                                                                          │
│   sets_gaps/<iid>.json   ← NOT staging: no vote, nothing to merge.       │
│                            One snapshot per install, replaced on each    │
│                            upload, read only by the maintainer's report. │
└─────────────┼────────────────────────────────────────────────────────────┘
              │
              │  GitHub Actions — merge_staging.yml, cron `22 */2 * * *`
              │  one CI job runs all four mergers sequentially
              ▼
┌──────────────────────────────────────────────────────────────────────────┐
│   DEMOCRATIC MERGE  (every 2 h — tallying settles every entry)           │
│                                                                          │
│   democratic_merge_crops.py    staging/*/crops + annotations.jsonl       │
│                                  ─► data/crops/<ab>/<sha>.png            │
│                                  ─► data/annotations.jsonl               │
│   democratic_merge_anchors.py  staging/*/anchors_grid_*.json             │
│                                  ─► data/anchors/<bt>_<bucket>.json      │
│   democratic_merge_screens.py  staging/*/screen_types + text crops       │
│                                  ─► data/screen_types/<T>/<sha>.png      │
│                                  ─► data/screen_types/metadata.jsonl     │
│                                  ─► data/text_corrections.jsonl          │
│   admin_merge.py               contributions/*.json                      │
│                                  ─► knowledge.json (phash → name)        │
│                                  on sets-sto/warp-knowledge              │
│                                                                          │
│   Every tallied entry is settled and its staging copy DELETED, plus     │
│   any staging file no row refers to. Commits are chunked, additions      │
│   before deletions. Poison filter strips `__virtual__` /                 │
│   `Test Item Name` before they reach data/.                             │
└─────────────┼────────────────────────────────────────────────────────────┘
              │
              │  GitHub Actions — train_central_model.yml, cron `0 * * * *`
              ▼
┌──────────────────────────────────────────────────────────────────────────┐
│   CENTRAL TRAINING  (every hour, 60 min hard cap, CPU runner)            │
│                                                                          │
│   admin_train.py reads ONLY data/  (never staging/)                      │
│       1. Stratified train/val split on promoted crops                    │
│       2. Fine-tune EfficientNet-B0 (icon_classifier.pt)                  │
│       3. Fine-tune MobileNetV3-Small (screen_classifier.pt)              │
│       4. Build ship_type_corrections.json from text_corrections.jsonl    │
│       5. Skip-if-unchanged: exit ~60 s if no new shas since last run     │
│       6. Upload all artefacts in one HF commit                           │
└─────────────┼────────────────────────────────────────────────────────────┘
              │
              ▼
┌──────────────────────────────────────────────────────────────────────────┐
│         HuggingFace — sets-sto/warp-knowledge  (DELIVERY)                │
│                                                                          │
│   models/icon_classifier.pt        models/icon_embedder.pt               │
│   models/screen_classifier.pt      models/embedding_index.npz            │
│   models/label_map.json            models/embedder_label_map.json        │
│   models/model_version.json        models/ship_type_corrections.json     │
│   knowledge.json                   models/community_anchors.json         │
└─────────────┼────────────────────────────────────────────────────────────┘
              │
              │   ModelUpdater (15 min check cadence, only install if
              │   remote trained_at strictly newer than local)
              ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                              USER MACHINE                                │
│                                                                          │
│   warp/models/icon_classifier.pt   ◄── new community model in place      │
│   warp/models/screen_classifier.pt                                       │
│   warp/models/icon_embedder.pt                                           │
│   …                                                                      │
│                                                                          │
│   SETSIconMatcher.reset_ml_session()  ─►  next Auto-Detect uses it       │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Two HF repos, one role each

| Repo | Role | What lives here |
|---|---|---|
| `sets-sto/sto-icon-dataset` | **Raw + curated data** | `staging/<iid>/…` (per-install raw votes), `contributions/…` (raw pHash candidates), and `data/…` (promoted, de-duplicated, voted-in artefacts the training pipeline consumes). |
| `sets-sto/warp-knowledge` | **Delivery** | `models/*.pt`, label maps, `knowledge.json` (phash → name overrides), `community_anchors.json`, `ship_type_corrections.json`. Everything `ModelUpdater` downloads. |

Splitting raw votes from delivered artefacts keeps the model repo small and
its history clean — a clone of `warp-knowledge` is the entire delivery
surface, no rummaging through staging history.

---

## 2a. What may be confirmed at all

Everything downstream — the vote, the promotion, the training run — treats a
confirmed label as ground truth. So the narrowest place to keep a bad name
out is the moment of confirmation, before anything is uploaded.

`WarpCoreWindow._name_is_acceptable` is that gate. A name must match the
slot's own candidate list exactly, or be one of the virtual classes; an empty
name is allowed, because that is how a slot is recorded as Unknown. Ship Type
and Ship Tier bypass it — they are OCR reads edited through their own combos,
not names picked from an item list, and their vocabulary is the ship roster
rather than the item cargo.

It backs three paths, and the third is the one that was missing:

| Path | What confirms | Gated since |
|---|---|---|
| `_on_accept` | a name the user typed or picked | 2026-05-18 |
| `_apply_auto_accept` | a recogniser result above the confidence threshold | 2026-09-04 |
| `_finish_bbox_drawn` | a recogniser result on a freshly drawn box | 2026-09-04 |

Confidence says "this looks like X". It says nothing about whether X is a
name the exporter can write or cargo can resolve, so the two auto-confirm
paths leave an unrecognised name pending for a human instead of writing it.
That was not hypothetical: `Fire on my Mark (Ground)` and `Liberated Borg
Kingdom Nanoprobes (space)` reached `data/` this way — wiki *art* names that
no user typed (see [`ML_PIPELINE.md`](ML_PIPELINE.md), "Reaching items nobody
has confirmed") — and the second beat the correct cargo name in a merge vote,
which its `losers` record still shows.

An empty candidate list means cargo could not be consulted, and the gate
falls open rather than blocking the trainer outright. The virtual classes are
added unconditionally, so they are excluded from that emptiness test — with
them included it could never be empty and the fallback would be unreachable.

The backend does not repeat this check: it holds no item vocabulary at all
(`config/labels.json` carries screen types and slot names, no item names), so
the gate is the client's and the merge-side vocabulary check in
`admin_reject_crops` is the review backstop.

---

## 3. `staging/` vs `data/` — the contract

Every artefact the user can vote on has two homes inside
`sets-sto/sto-icon-dataset`:

| Where | Who writes here | Who reads here | Lifetime |
|---|---|---|---|
| `staging/<install_id>/…` | Backend, on behalf of one client at a time | The four democratic mergers (every 2 h) | Deleted on promotion to `data/` — drain-on-promote |
| `data/…` | The four mergers only — atomic per run | `admin_train.py` (the only training input), `admin_audit_staging.py` (for orphan detection) | Permanent unless a future merger overwrites the entry with a fresh majority |

**Why the split:** training reads `data/`, never `staging/`. That's the
invariant. Without it, every per-install upload would directly influence
the next training run — a single user could outvote the community by
flooding their own staging. With it, the only path from a vote to a model
is through majority promotion.

---

## 4. What a tally decides

Staging means one thing: an entry has arrived from a client, has not been
tallied, and is not in the models. Tallying settles it either way, so every
entry is applied and staging empties. The vote count then expresses
confidence *in* the record rather than gating entry to it.

`merge_staging.yml` still passes `--min 2` to every merger, but only
`admin_merge` enforces it, and there asymmetrically: a new pHash needs one
vote, changing an existing one needs the threshold. The crops merger accepts
every tallied entry and ignores the flag, which its `--min` help text says
outright — the parameter was removed from its merge function on 2026-09-04
because a dead knob that looks live is worse than no knob.

| What the tally finds | What happens |
|---|---|
| Key is not yet in `data/` | Promoted on first sighting. |
| The vote agrees with `data/` | Counted: `votes` accumulates, and the record is otherwise left alone. |
| The vote disagrees | Applied. `votes` restarts from this batch's count and the superseded verdict is kept in `losers` with its own strength. |

**Why this replaced a two-vote ratchet on updates.** That bar is sound for a
crowd and equal to "never" for this project. Measured 2026-09-03: two
contributors with annotations, 4003 entries in staging of which 3897 merely
confirmed what `data/` already said and could never drain, and 102
corrections waiting indefinitely — among them a crop stored as
`Attack Pattern Beta'`, a name no cargo row has ever matched, which a human
had already corrected while the models kept training on the typo.

Nothing is lost when a verdict is overturned: `losers` carries the previous
name and how many votes it had, so a contested entry is distinguishable from
a settled one and the change is reversible. A weak entry is visibly weak —
one vote, no dissent — which is what a review of the tail sorts on.

`--min` remains on the command line for compatibility with the shared merge
workflow and is **not** a gate for crops; a non-default value is reported at
the top of the run so it cannot look effective when it is not.

---

## 5. Drain on promote, and the sweep behind it

Each of the four mergers ends its run by emitting the rows it promoted to
`data/` and deleting the corresponding `staging/` paths. Since the promotion
grew past what one commit can carry, that is a short ordered sequence rather
than a single atomic write — additions first, then the index that references
them, then the deletions — so an interrupted run leaves duplicates to redo
and never a reference to something missing (`hf_commit`).

Drain-on-promote alone is not enough to keep staging clean, because it only
removes what was promoted. Anything the tally *refuses to consider* is
invisible to that drain by construction, so it stays for good. Five such
classes have been found and closed:

| Residue | Why it could not drain | What happens now |
|---|---|---|
| A crop PNG no annotation row refers to | it is only tallyable through its row, so it produced no vote to promote | the crop merge sweeps it, including whole install directories that have no `annotations.jsonl` at all — uploads write PNGs and rows in one commit, so that state means something other than the upload path wrote them |
| A row whose crop exists nowhere | tallying reads the PNG's bytes; with no PNG in staging and no entry in `data/`, the row can never be promoted | `_surviving_rows` drops it — the mirror of the case above, and the direction that had no sweep |
| A crop barred by the review ledger | the tally skips rejected shas, so nothing promoted them and nothing drained them; the rejection was re-litigated on every run | `_surviving_rows` drops the row and the sweep drops the PNG |
| A screen typed outside the whitelist | `democratic_merge_screens` skips a type it cannot merge, so it never promotes | swept by `_sweep_unpromotable`; `UNKNOWN` is the live case, the client's not-yet-classified sentinel |
| A contribution naming a virtual class | `admin_merge` refuses `__*` unconditionally, so its pHash is never promoted | drained as refused rather than left pending |

Both crop and screen mergers had the same trap: they return early when there
are no votes, which is **exactly the state these entries leave staging in**.
A sweep placed after that point is unreachable in the only situation it
exists for, so each merger now tests for sweepable residue before deciding
there is nothing to do.

The counts when this was completed on 2026-09-04: 29 `UNKNOWN` screens, 4
rows whose crop existed nowhere, 2 crops barred by the ledger. Before that,
ten orphaned crops had been sitting in `staging/migration-sister/`, left by a
one-off migration; the only thing that had ever removed any of these was
`admin_drain_stale_staging.py`, run by hand, last on 2026-07-17.

`_surviving_rows` keeps an ordinary single vote waiting for company. Dropping
one of those would discard a contribution, which is worse than any residue
swept here.

`admin_audit_pipeline_movement.py` asserts the result daily: staging holding
anything no run can settle is a breach, reported with its origin. See §6. The grep target in `merge_staging.yml`
("Drain summary") catches lines like:

```
DRAIN: domain=crops promoted=42 staging_files_removed=42 commit=<sha>
```

so the workflow summary tells you per-domain how much was drained.

Two consequences:

1. **Steady-state orphan count is zero.** A staging entry only persists if
   the merger ran but didn't promote it — i.e. it didn't meet the
   threshold yet. Anything else is a leak.
2. **Single-commit atomicity.** The promote write *and* the drain delete
   land together. There is no observable moment where the same sha
   exists in both `staging/` and `data/`.

---

## 6. The audit safety net

Two audits, and they answer different questions.

`audit_pipeline_movement.yml` runs `admin_audit_pipeline_movement.py` every
morning. It asks whether anything is **flowing**: uploads arriving while
`data/` has not been written for `--max-age-days` is a breach, and so is
staging holding a file no run can settle. It is deliberately blind to the
cause — a bad token, a raised threshold, a crashed merger and an opaque HTTP
400 all look the same from there, and all deserve an email the next morning.

It exists because the crop merge failed on every scheduled run from
2026-07-16 to 2026-09-03 and the workflow reported success throughout: the
step piped its output through `tee`, and bash returns the exit status of the
last command in a pipeline. Seven weeks of promotions were lost behind a
green tick, and it was found by hand while tracing an unrelated crop. Every
`run:` block in this repo now sets `pipefail`, and a test holds that line.

The state audits below could not have caught it: a pipeline that has stopped
entirely looks healthy to a threshold on pile size, right up until the pile
is enormous.

`audit_staging_health.yml` runs `admin_audit_staging.py` on the 1st of
every month (and on demand). It is **read-only**:

- Counts entries in `staging/` whose semantic key (sha, pHash, anchor
  bucket) is already present in `data/`.
- Compares against per-domain thresholds (default: 100 crops, 50 screens,
  50 contributions).
- Exits non-zero — and emails the repo owner — if any threshold is
  breached.

The audit deliberately **does not auto-fix**. If `merge_staging.yml`
starts leaking orphans, a scheduled cleanup would paper over the bug. The
intended response chain is: audit fails → owner investigates → owner runs
`drain_stale_staging.yml` manually after the root cause is fixed.

Anchors are excluded from the default audit because their staging files
aggregate multiple contributors' votes — a "stale" anchor may still be
in-flight rather than orphaned.

---

## 7. One-shot drain — `admin_drain_stale_staging.py`

Kept for historic backlogs, and no longer part of normal operation: the
mergers settle and drain everything they tally, and the crop merge sweeps
what no row refers to (§5), so absent an actual failure there is nothing for
this to do. Its rule — drop a staging copy whose sha is already in `data/` —
also no longer matches the residue that occurs in practice, which is files
`data/` never had.

The drain-on-promote logic in the mergers is recent. Before it shipped,
promoted entries accumulated in `staging/` indefinitely.
`admin_drain_stale_staging.py` is the one-time catch-up: it walks every
`staging/<iid>/…` path, checks whether the same sha / pHash already
exists in `data/`, and deletes the staging copy.

```
crops          staging/<iid>/crops/<sha>.png       DROP if sha in data/annotations.jsonl
                staging/<iid>/annotations.jsonl     TRIM promoted lines
screens        staging/<iid>/screen_types/<T>/…    DROP if sha in data/screen_types/metadata.jsonl
contributions  contributions/<date>/<id>.json      DROP if id in knowledge.json::processed_contributions
                contributions/<date>/<id>.png      DROP companion crop
anchors        staging/<iid>/anchors_grid_*.json   OPT-IN (--include-anchors only)
```

Two atomic commits — one per HF repo. The script is content-addressed
and idempotent; running it twice on the same state is a no-op.

It is not on a schedule. Triggered manually via `drain_stale_staging.yml`
(workflow_dispatch only) after the monthly audit flags a problem and the
root cause is fixed.

---

## 8. The poison filter

Certain names must **never** reach `data/`:

- `__boff_*` and other internal test markers — dev-time placeholders
  that would corrupt the training set.
- `Test Item Name` — dev-time placeholder, never a real item.

`__empty__` and `__inactive__` are **allowed** through the crop merger
(`democratic_merge_crops.py`). The ArcFace embedder needs them as gallery
classes so inactive/empty slots map to their own class instead of
nearest-neighbour-snapping to a real ability (see
[`ML_PIPELINE.md` §3 — Virtual gallery classes](ML_PIPELINE.md#virtual-gallery-classes-__inactive__-__empty__)).

The pHash override path in `icon_matcher.py` independently suppresses
virtual names (`name.startswith('__') → suppress=True`), so they never
enter `knowledge.json` as hard overrides — the risk that originally
motivated blocking all `__*` names in the merger.

The filter is enforced in **two** places (defence in depth):

1. **Client upload guard** (`warp/knowledge/sync_client.py`,
   `_poison_filter_enabled` flag) — blocks poison names from leaving
   the client at all.
2. **Merger guard** (`_is_poison_name` in each `democratic_merge_*`) —
   blocks them again when promoting to `data/`, in case the client
   filter is bypassed by an older release. The crop merger exempts
   `__empty__` / `__inactive__` (see above); other mergers block all
   `__*` names.

The mapping of every output-side filter on the client is documented
in [`client_user_view_filter.md`](client_user_view_filter.md).

---

## 9. End-to-end timing — best case

| Step | Time after Accept |
|---|---|
| Local crop written to `~/.local/share/warp/training_data/` | < 100 ms |
| `SyncWorker` upload batch posted to backend | up to 10 min (next sync tick) |
| Backend writes to `staging/<install_id>/…` | seconds |
| `merge_staging.yml` promotes the new sha to `data/` | up to 2 h (next merge cycle) |
| `train_central_model.yml` includes it in next training run | up to 1 h after promotion |
| Model published to `sets-sto/warp-knowledge` | ~10 – 50 min training + upload |
| Client `ModelUpdater` downloads + installs | up to 15 min after publish |

**Best case: ~3.5 h from confirmation to delivered model.**
**Worst case (training skipped multiple cycles, low vote count needs to
hit 2):** several days — but the same correction is already useful
**immediately** in the local session via the in-session matcher (see
[`WARP_GUIDE.md` §6 — Manual accept](WARP_GUIDE.md#manual-accept)).

---

## 10. Where to look when something goes wrong

| Symptom | Where to look first |
|---|---|
| "I confirmed an icon, but the next Auto-Detect still gets it wrong on a different machine" | Expected — the journey takes hours; see §9. |
| Upload counter sits at zero in System logs | `SyncWorker` cap hit, backend unreachable, or pending batch zero. See [`REMOTE_SYNC_AUDIT.md`](REMOTE_SYNC_AUDIT.md) §2.1. |
| `data/annotations.jsonl` line count not growing despite many uploads | Threshold not met (need a second voter for an update) or all uploads are dupes of existing shas. |
| Monthly audit fails with `orphans > threshold` | A merger regression. Investigate the merger's last run; do **not** run drain blindly. |
| Staging cleanup leaves a per-install folder behind | The merger drained the contents but not the parent dir. Cosmetic — does not affect training. |
