# Remote Sync — Capacity Audit

> Snapshot before opening the program to a wider user base. Verifies the
> upload and download paths and estimates how many concurrent users the
> current infrastructure can carry. No code changes were made as part of
> this audit — findings and recommendations are listed at the bottom.

---

## 1. Channels in use

WARP talks to **a single backend** for all writes; HF Hub is read-only
from the client.

| Channel | Target | Used for | Auth |
|---------|--------|----------|------|
| HF Spaces backend | `sets-sto-warp-backend.hf.space` | `POST /contribute` (pHash), `POST /contribute/bulk-crops`, `POST /upload/screen-types`, `POST /upload/anchors`, `GET /knowledge`, `GET /model/version` | Server-side `HF_TOKEN` (never on client) |
| HF Hub (read-only) | `sets-sto/warp-knowledge` | Model + label-map + community_anchors + ship_type_corrections download | Anonymous (public dataset) |

**Since v1.0.5** the client holds **no HF credentials**. All writes to the
icon dataset go through the backend, which holds the write token as a
Space secret.

---

## 2. Upload path

### 2.1 Crops + annotations + screen-types + anchors (backend proxy)

```
WARP CORE confirm
  └─► TrainingDataManager.add_annotation()
         • annotations.json + crop PNG written under warp/training_data/
         • per-sha label cache (.sync_uploaded_labels.json) tracks last-sent
           (slot|name) to skip no-op rewrites
  ▼
SyncManager.check_and_upload()         (registered in warp_button.py)
  • interval:        every 10 min
  • startup delay:   15 s after app launch
  • daily rate cap:  1000 new files per install_id (corrections are free)
  ▼
SyncWorker.run() — batched POSTs to the HF Spaces backend:
  • POST /contribute/bulk-crops      (≤50 items per request)
      → staging/<install_id>/crops/<sha>.png (only when sha unseen)
      → annotations.jsonl entry merged server-side (last-wins per sha)
  • POST /upload/screen-types        (≤20 items per request)
      → staging/<install_id>/screen_types/<stype>/<sha>.png
  • POST /upload/anchors             (≤20 grids per request)
      → staging/<install_id>/anchors_grid_<sha8>.json
```

**Validated invariants** (`warp/trainer/sync.py`, `main.py` on the backend):

- Crops are stored under `staging/<install_id>/` — no two users can collide.
- `.sync_uploaded_hashes.json` and `.sync_uploaded_labels.json` cache the
  sha→sent and sha→label state locally so anonymous `list_repo_files` is
  only called once per install (bootstrap), not on every cycle.
- Per-sha last-wins dedup for `annotations.jsonl` runs **server-side** —
  the backend reads the current jsonl, merges new entries, and rewrites
  in one commit.
- One commit per batch on the backend — N crops cost ceil(N/50) HF
  commits, not N.

### 2.2 pHash knowledge (backend proxy)

```
WARP user confirms an unusual icon
  └─► WARPSyncClient.contribute(crop_bgr, item_name, …)
          • non-blocking thread
          • local rate-limit:  200 contributions / install_id / day
          • circuit breaker:   on 503/network error, back off 5 min
  ▼
POST https://sets-sto-warp-backend.hf.space/contribute
  • 60 s read timeout (covers Space cold-start after ~48 h idle)
  • backend appends to contributions/YYYY-MM-DD/<uuid>.{png,json}
  • admin merge (manual / scheduled) folds approved contributions into
    knowledge.json
```

---

## 3. Download path

### 3.1 Models (`ModelUpdater`)

```
ModelUpdater().check_and_update()        (called at app launch + WARP CORE open)
  • polling cadence: 15 min  (_CHECK_INTERVAL_HOURS = 0.25)
  • rate-limit cache: warp/models/model_version_remote_cache.json
  • requests with (connect=5 s, read=60 s) timeouts
  ▼
GET https://sets-sto-warp-backend.hf.space/model/version
  └─► remote = {trained_at, n_classes, val_acc, available}
  ▼
If remote.trained_at > local.trained_at OR embedder stale:
  hf_hub_download from sets-sto/warp-knowledge for each of:
    icon_classifier.pt              (EfficientNet-B0 softmax — required)
    label_map.json                  (required)
    icon_classifier_meta.json
    model_version.json
    screen_classifier.pt            (MobileNetV3-Small)
    screen_classifier_labels.json
    community_anchors.json          (optional, P11)
    ship_type_corrections.json      (optional)
    icon_embedder.pt                (optional, ArcFace)
    embedder_label_map.json         (optional)
    icon_embedder_meta.json         (optional)
    embedding_index.npz             (optional)
  ▼
shutil.copy2 to warp/models/ (only after ALL required files OK)
  ▼
SETSIconMatcher.reset_ml_session()
LayoutDetector.reset_community_anchors_cache()
TextExtractor.load_corrections(...)
```

**Network resilience:**

- Retry schedule: 1 min → 5 min → 15 min → 60 min on network failure.
- Failure does **not** save the rate-limit timestamp → next app launch retries
  immediately instead of waiting 15 min.
- Atomic-ish file install: all temp paths gathered first, only copied to
  `warp/models/` once every required file downloaded successfully.

### 3.2 pHash knowledge (`WARPSyncClient`)

- Polling cadence: re-download after `KNOWLEDGE_MAX_AGE_HOURS = 24`.
- 5 s connect / 15 s read timeout (cache fallback exists).
- Used as override layer in `icon_matcher` before template matching.

---

## 4. Capacity envelope

### 4.1 HF Hub (sets-sto/sto-icon-dataset) — backend-side writes

| Constraint | Value (source) | Headroom at 100 users | Headroom at 1000 users |
|------------|----------------|----------------------|------------------------|
| Commits per token (backend) | ~1000/day (HF docs) | Batched: 100 users × ~2 batches/cycle × 144 cycles/day ≈ 28 800/day in worst case, but the cache + 10-min de-dup typically reduces to <500/day | Same shared backend token; batching of 50 crops / 20 screens / 20 anchors keeps commit count tractable up to ~5 k users |
| Concurrent commits | Serialized on the backend side (single Space worker) | Fine — requests queue at backend, not at HF | Need to scale Space replicas or move to paid tier before commit serialization becomes the bottleneck |
| `list_repo_files` (anonymous, client) | Grows linearly with crops | 100 × 50 = 5 000 files → list ≈ 1–2 s | 1000 × 50 = 50 000 files → list ≈ 10–20 s |
| Storage | Free public datasets, no published cap | Negligible (~64×64 PNG ≈ 4 KB) | ~200 MB at 1000 × 50 crops |

**Single shared backend token, never on clients.** The HF write token
lives only as a Space secret. Compromising a client cannot leak the token;
abuse must come through the backend's rate-limited, validating endpoints.

### 4.2 HF Spaces backend (`sets-sto/warp-backend`)

| Constraint | Free tier | At 100 active users | At 1000 active users |
|------------|-----------|--------------------|----------------------|
| Cold start | ~30–60 s after ~48 h idle | Handled by 60 s read timeout + client circuit breaker | Same — once warm, stays warm |
| Concurrent requests | Single uvicorn worker (default Docker SDK) | OK — most requests are I/O-bound on HF API | Bottleneck — request queue grows; upgrade Space or shard by install_id |
| CPU/RAM | 2 vCPU / 16 GB (free CPU Basic) | Plenty | Fine for proxying; HF Hub is the real bottleneck |
| Bandwidth | Generous but not unlimited | OK | Monitor — `/knowledge` is the heaviest endpoint |

**Free Spaces comfortably supports ~100 concurrent users.** Beyond that,
the single-worker queue becomes the constraint; paid Space tiers add
replicas without code changes.

### 4.3 GitHub Actions training pipeline

- `admin_train.py` runs hourly. Each run reads `staging/*/` from HF and
  trains EfficientNet-B0. Step timeout: 6 h (GH default).
- At 1000 users × 50 crops, training set ≈ 50 000 images — still fits in
  a single Actions run on a `ubuntu-latest` runner.

---

## 5. Risk register

| Risk | Severity | Mitigation today | Recommendation |
|------|----------|------------------|----------------|
| ~~HF token shared across all users — leak compromises every install~~ | **RESOLVED v1.0.5** | Token removed from all clients; lives only as Space secret; legacy `~/.config/warp/hub_token.txt` purged on first run after upgrade | Rotate the old shared token in HF UI (old one must be revoked since it leaked to every install). |
| HF rate limit on backend token at >5000 active users | Medium | Backend batches 50 crops / 20 screens / 20 anchors per commit; per-sha + per-label cache keeps cycles ~no-op when nothing changed | Shard backend writes by install_id prefix, or rotate among multiple backend tokens. |
| Space cold-start makes the first user of a session wait | Low | 60 s read timeout absorbs it; client circuit breaker holds outbound queue | Upgrade to a paid Space tier — no idle sleep, faster restart. |
| All clients start at the same time (e.g. weekend evening) → thundering herd at second 16 | Medium | Fixed 15 s startup delay; per-cycle de-duplication keeps payload small | Add jitter to startup delay (e.g. `15 s + uniform(0, 30 s)`) to spread commits over 30 s window. |
| `list_repo_files` cost grows with crops | Low | Cached locally after first call; subsequent syncs O(1) | None now. Revisit if user base passes 5 k installs. |
| Bad actor floods their own staging/ folder | Low | Per-install staging; client rate limit (1000/day); admin merge step is the gate | None — staging is the right design. |
| User edits a label N times → N HF commits | Low | Label cache (`.sync_uploaded_labels.json`) skips no-op rewrites | Already mitigated. |
| Free public dataset eventually hits HF size limits | Low | None published; HF informally hosts datasets up to ~100 GB | Monitor; archive older `staging/*` after admin merge to keep active size low. |

---

## 6. Bottom line

**v1.0.5 closed the high-severity shared-token risk.** All client writes
now flow through the HF Spaces backend, which holds the write token as a
Space secret. Existing installs purge their legacy `hub_token.txt` on
first run after upgrade.

**For wider testing (a few dozen to ~100 concurrent users), the setup is
fit for purpose.** Both upload and download paths are non-blocking,
idempotent, and cached. The retry/back-off + circuit breaker covers
Space cold-starts.

**Remaining follow-ups before scaling past ~5 000 active contributors:**

1. Rotate the old shared HF write token (still valid until revoked in
   the HF UI — manual user action).
2. Add startup-delay jitter to spread per-cycle uploads over a 30 s
   window.
3. Consider sharding backend writes by install_id prefix, or a paid
   Space tier for replica concurrency.
