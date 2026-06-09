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
└─────────────┼────────────────────────────────────────────────────────────┘
              │
              │  GitHub Actions — merge_staging.yml, cron `22 */2 * * *`
              │  one CI job runs all four mergers sequentially
              ▼
┌──────────────────────────────────────────────────────────────────────────┐
│   DEMOCRATIC MERGE  (every 2 h, Z3 thresholds: NEW=1, UPDATE≥2)          │
│                                                                          │
│   democratic_merge_crops.py    staging/*/crops + annotations.jsonl       │
│                                  ─► data/crops/<sha>.png                 │
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
│   On promotion: source staging entry is DELETED in the SAME commit       │
│   (drain-on-promote). One commit per merger. Poison filter strips        │
│   `__virtual__` / `Test Item Name` before they reach data/.              │
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

## 4. Z3 asymmetric thresholds

`merge_staging.yml` passes `--min 2` to every merger by default. Each
merger applies the threshold asymmetrically:

| Case | Votes needed |
|---|---|
| Key is **not yet** in `data/` (new sha, new pHash, new anchor bucket) | **1** is enough — promote on first sighting. |
| Key **already exists** in `data/` and the new majority disagrees | **≥ 2** votes for the new label before the existing entry is overwritten. |

The two-vote ratchet on updates prevents a single dissenter from flipping
a long-standing community label. The one-vote acceptance on new keys
keeps unusual icons reachable — waiting for a second vote on a
once-a-year ship would mean it never lands in `data/` at all.

The `merge_staging.yml` workflow_dispatch lets admins override `min_votes`
on demand if a campaign of corrections needs to land in one cycle.

---

## 5. Drain on promote

Each of the four mergers ends its run by emitting the rows it promoted to
`data/` and **deleting the corresponding `staging/` paths in the same
HF commit** that wrote them. The grep target in `merge_staging.yml`
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

Two name patterns must **never** reach `data/`:

- `__virtual__` style markers (`__empty__`, `__inactive__`,
  `__boff_*`) — they are legitimate training labels (the model needs
  them to recognise empty slots), but if they leaked into the lookup
  table the matcher would override every real crop with conf 1.0 and
  the slot would always read as empty.
- `Test Item Name` — dev-time placeholder, never a real item.

The filter is enforced in **two** places (defence in depth):

1. **Client upload guard** (`warp/knowledge/sync_client.py`,
   `_poison_filter_enabled` flag) — blocks the names from leaving the
   client at all.
2. **Merger guard** (`_is_poison_name` in each `democratic_merge_*`) —
   blocks them again when promoting to `data/`, in case the client
   filter is bypassed by an older release.

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
