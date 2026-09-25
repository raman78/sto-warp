# sto-warp ML Pipeline — Technical Reference

How machine learning works end-to-end in sto-warp: what the client captures
locally, how that data reaches the community, where the central model is
trained, and how updates flow back to every install.

---

## Overview

sto-warp uses two production classifiers plus one embedder:

| Model | Architecture | Purpose |
|-------|-------------|---------|
| `icon_classifier.pt` | EfficientNet-B0 | Matches item icon crops to item names |
| `screen_classifier.pt` | MobileNetV3-Small | Classifies screenshot type (SPACE_EQ, BOFFS, TRAITS, …, SKILLS, DISCARD) — 9-class softmax |
| `icon_embedder.pt` | EfficientNet-B0 + ArcFace head | k-NN gallery lookup — primary matcher for BOFF abilities; includes `__inactive__`/`__empty__` as gallery classes |

**The client does not train the production models.** All three files are
downloaded from `sets-sto/warp-knowledge` on HuggingFace (see §4). The
client's contribution is the *data* — confirmed icon crops, confirmed screen
types, and OCR corrections — uploaded to staging and folded into the
community model by the central pipeline (§3).

A separate one-shot admin path for bootstrapping the embedder on synthetic
crops lives at the bottom of this document (§9). End users do not run it.

### Multi-language OCR (1.0.18)

The recognition pipeline accepts screenshots from non-English STO clients.
A translation layer sits between EasyOCR output and the reference-data
lookup tables:

```
OCR text (any lang)  ──►  ui_translations.py  ──►  English canonical
                               │
                    reads warp/data/ui_translations.csv
```

| Component | File | Role |
|---|---|---|
| Translation CSV | `warp/data/ui_translations.csv` | Admin-editable table. Each row maps `(category, canonical_en, language, translation)`. Categories: `space_slot`, `ground_slot`, `eq_word_single`, `eq_word_first`, `eq_word_second`, `screen_header`, `spec_name`, `ship_type_word`. |
| Translation loader | `warp/recognition/ui_translations.py` | Parses the CSV once on first access. Exposes `normalize_map()`, `synonyms()`, `augment_substring_phrases()`, `translate_ship_type()`, `ocr_languages()`. |
| EasyOCR language list | `ui_translations.ocr_languages()` | Derived from CSV contents — language codes in the CSV automatically add to the EasyOCR reader's language list (e.g. `['en', 'de']`). |
| Ship-type translation | `TextExtractor.extract_ship_info` | After OCR reads a ship type, `translate_ship_type()` splits the string on hyphens/spaces and replaces each token using the `ship_type_word` category (longest match first, up to 3-token phrases). |
| Screen-type detection | `_fuzzy_tier` | `_TRAIT_SPACE_HEADERS`, `_SPACE_EQ_LABELS` and similar tuples are expanded via `augment_substring_phrases()` so localized header text triggers the same screen-type classification. |

To add a language: append rows to the CSV with the appropriate ISO 639-1
code. No Python edits needed. German (`de`) is the first supported
non-English language; it covers space/ground slot labels, equipment
categories, screen headers, and ship-type compound words.

---

## 1. Local data capture

### Trigger

The user accepts a bounding box in WARP CORE (Enter, autocomplete pick,
**Accept** button, or auto-accept ≥ threshold).

### Input data

```
~/.local/share/warp/training_data/
├── annotations.json          ← bbox + label records, keyed by screenshot hash
├── crops/
│   ├── crop_index.json       ← crop file → slot, name, state, source
│   ├── <slot>__<name>__<image_key>-<ann_id>.png   ← the box, cut at its own size
│   └── ...
└── screen_types/
    ├── SPACE_EQ/
    │   └── <filename>.png    ← confirmed full screenshot per type
    ├── SKILLS/               ← captain skill tree screens (space or ground)
    ├── DISCARD/              ← non-build images (console screenshots, irrelevant files)
    └── ...
```

Every confirmation:
1. Records the annotation in `annotations.json` under the screenshot's key,
   the first 16 hex digits of its SHA-256 (`bbox`, `slot`, `name`, `state`,
   `ann_id`, `ml_name`, `ml_conf`, `auto_confirmed`, `crop_name`, …).
2. Cuts the box from the screenshot and saves it under `crops/`
   (`TrainingDataManager._export_crop`), indexed in `crop_index.json`.
   That index is what the uploader sends (`get_confirmed_crops`).

**A crop's name carries its screenshot.** `ann_id` is a hash of bbox and slot
alone, so the same slot box on two screenshots has the same `ann_id`. The
game UI does not move, so this is common: 125 ids were shared on the
maintainer's store. Crops used to be named `<slot>__<name>__<ann_id>.png`
and found by `ann_id`, and the startup sweep (`cleanup_orphaned_crops`)
renamed one screenshot's crop to the other's label on every start. Measured
2026-09-25: 98 of 7314 confirmed crops showed a picture other than their
box, and about 100 had reached the community under another item's name. One
of them was a Fragment of AI Tech icon filed as Unconventional Systems.
Names now include `<image_key>-`, and every lookup, rename and delete is
scoped to the screenshot (`_crop_for`, `_cleanup_crops_for_ann`).

`migrate_crop_names` runs first at startup and converts a store that still
has old names. It re-cuts each crop from its screenshot, found by hash
under `screen_types/`. Where the screenshot is absent it keeps the old file
only if no other annotation shares its `ann_id`, and otherwise leaves the
annotation without a crop and logs it, because the old file could show
either picture. On the maintainer's store: 7440 re-cut, 704 kept, 16 left
without a crop, and afterwards all 7440 checked crops matched their boxes.
Re-cut crops with unchanged pixels keep their bytes and are not re-sent.
Those whose label was wrong are re-sent under the right one, which the crop
merge applies as a correction.

**The name is read from the right.** Item names are cut at 40 characters,
and a cut that ends in `_` (`console_-_universal_-_flagship_tactical_`)
leaves `___` before the id. The startup sweep split names from the left, read
the id as `_<id>`, found no annotation for it, and deleted the crop as an
orphan on every start. That is why 188 console and weapon annotations with
long names had no crop at all. `_parse_crop_fname` splits from the right,
and those crops were cut again from their screenshots.

**Readers do not rely on `crop_name`.** The field in `annotations.json` is
not kept current: saving a row rewrites the record from a fresh
`Annotation`, which carries none. The session seed
(`SETSIconMatcher.seed_from_training_data`) and `scrub_training_data` fall
back to the file name, built by the same `TrainingDataManager._crop_fname`
the writer uses. They try the name with the screenshot key first, then the
bare-`ann_id` form older stores still have.

Screen type labels are saved separately when you tick / change the screen
type for a file (stored in `screen_types/<TYPE>/<filename>.png`). The
crop is also fed to the in-session matcher immediately, so the next
Auto-Detect on a different screenshot can already match against the
just-confirmed icon — see [`docs/WARP_GUIDE.md` §6](WARP_GUIDE.md#6-confirming-items-and-accepting-results)
for the user-side view.

### No local production training

Earlier sets-warp releases shipped a "Train Model" button that produced a
local icon classifier. That path was removed before sto-warp 1.0.0. The
client today only **captures** training data; production training is
performed centrally (§3). The only training code still callable on a user
machine is the embedder bootstrap (§9), which targets the k-NN gallery, not
the softmax classifier, and is gated behind a CLI flag.

---

## 2. Community upload (local → HuggingFace staging)

### When it happens

WARP CORE runs a background sync timer every 10 minutes. On each tick it
checks for unsynced confirmed crops and POSTs them in batches to the HF
Spaces backend (`sets-sto-warp-backend.hf.space`), which holds the HF
write token as a server-side secret. Since v1.0.5 the client holds no HF
credentials.

File: `warp/trainer/sync.py` — `SyncWorker`

### Rate limiting

Two independent caps, enforced client-side per install_id per UTC day:

| Channel | Cap | Constant | Notes |
|---|---|---|---|
| Crops + screen types (`SyncWorker`) | **1000 / day** | `MAX_DAILY_UPLOADS` in `warp/trainer/sync.py` | Corrections to a previously-uploaded crop do not count against the cap. |
| pHash knowledge contributions (`WARPSyncClient.contribute`) | **200 / day** | `MAX_CONTRIBUTIONS_PER_DAY` in `warp/knowledge/sync_client.py` | Per-icon pHash overrides only — does not gate the main crop upload path. |

The counters are persisted in the per-channel state files under
`~/.config/warp/` so they survive restarts within the same UTC day.

### What is uploaded

**Icon crops** → `sets-sto/sto-icon-dataset` (HF Dataset):
```
staging/<install_id>/crops/<sha256>.png   ← icon crop image (64×64 px)
staging/<install_id>/annotations.jsonl    ← one JSON line per crop:
    {"crop_sha256": "...", "name": "Ablative Shell", "slot": "Science Console", ...}
```

**Screen type screenshots** → same repo:
```
staging/<install_id>/screen_types/<TYPE>/<sha256>.png
```

**Ship Type / Ship Tier text crops** → same staging repo:
```
staging/<install_id>/crops/<sha256>.png       ← text region crop (same path as icon crops)
staging/<install_id>/annotations.jsonl        ← entry per text crop:
    {"crop_sha256": "...", "name": "Fleet Support Cruiser",
     "slot": "Ship Type", "ml_name": "F1eet Support Cruiser", "date": "..."}
```

- `name`: user-confirmed ship type / tier string
- `ml_name`: raw OCR output before user correction (empty when OCR was already correct)
- These entries are filtered out of icon classifier training and processed separately
  to build `ship_type_corrections.json` (see §3 below)

### What is NOT uploaded

- Full screenshots (only the 64×64 icon crop or text region crop)
- Ship name or character name (Ship Name bbox is position-only — never crops, never text)
- The local `.pt` model files
- Anything outside `warp/training_data/`

### Install ID

A random UUID generated at first launch, stored in
`warp/knowledge/install_id.txt`. It is anonymous — not linked to any account.
Its only purpose is democratic voting (1 install = 1 vote per crop hash).

### Deduplication

Each crop is identified by `sha256(crop_bytes)`, truncated to 32 hex chars —
the same key the dataset is stored under, so anything comparing the two must
truncate as well. The same icon appearing in several screenshots is therefore
uploaded once per install.

The hash alone is not the whole test, and treating it as such was a fault
rather than a design. A label can change while the file does not: correcting
an item name, or re-typing a screenshot in WARP CORE, leaves the bytes
untouched. So the upload cache records the **label** each item was last sent
under, and a change to it re-queues the item:

| Channel | Cache | Compares |
|---|---|---|
| crops | `.sync_uploaded_labels.json` | `slot\|name` |
| screen types | `.sync_uploaded_screen_hashes.json` | the screen type |

Crops have compared the label since they were written. Screen types did not
until 2026-09-05, and the consequence was that a correction never left the
machine at all — see [`DATA_LIFECYCLE.md`](DATA_LIFECYCLE.md) §5b for what
that cost and how the old cache format migrates.

---

## 3. Central training (HuggingFace staging → trained model)

### Trigger

A GitHub Actions workflow runs `admin_train.py` on a schedule:

```yaml
# .github/workflows/train_central_model.yml  (sets-warp-backend repo)
on:
  schedule:
    - cron: '0 * * * *'   # every hour
  workflow_dispatch:       # manual trigger
```

Training is skipped (fast exit) when:
- No new crops since the last training run (`--skip-if-unchanged`)
- Fewer than 10 new crops arrived (`MIN_NEW_CROPS = 10`)

### Democratic voting

```
For each crop sha256:
    votes = {install_id: label for each user who uploaded that crop}
    winner = majority vote (most common label)
    if tie → first uploader's label wins
```

A single user cannot override the community. If User A labels a crop
"Ablative Shell" and three others label it "Ablative Field Projector",
the community label wins.

### Screen type dataset capping

To prevent dataset bloat for stable UI screens:

```
For each screen type class:
    if n_samples >= 30 and n_samples > 150:
        randomly keep 150 samples
        (avoids storing thousands of near-identical UI screenshots)
```

### Central training process (icon classifier)

File: `sets-warp-backend/admin_train.py` — `train()`

```
1. Download previous icon_classifier.pt from HF warp-knowledge (for fine-tuning)
2. Download all winning crops from staging via snapshot_download per install_id
   (bulk folder download — far fewer HTTP round-trips than per-file)
3. Stratified train/val split
4. Build EfficientNet-B0, replace head for n_classes
5. Load previous backbone weights (classifier.* keys stripped)
   LR = 3e-4 × 0.3 when fine-tuning, 3e-4 from scratch
6. Train with focal loss + cosine annealing + early stopping
7. Save: icon_classifier.pt, label_map.json, icon_classifier_meta.json,
         model_version.json, training_manifest.json
8. Upload all files to sets-sto/warp-knowledge/models/
```

`training_manifest.json` records the set of crop SHAs used in this run —
the next run's skip-if-unchanged check compares against this.

### Central training process (screen classifier)

File: `sets-warp-backend/admin_train.py` — `train_screen_classifier()`

Same flow with MobileNetV3-Small, fine-tuning from the previous
`screen_classifier.pt` backbone.

**The class count is a property of each training run, not a constant.** Two
things vary it, and nothing downstream may assume either:

- `admin_train.SCREEN_TYPES` filters which screen types are admitted at all
  (11 entries). It does **not** set the head order.
- The head is built over `sorted(set(labels))` — alphabetically, across only
  the classes that survived `SC_MIN_CLASS_SAMPLES`. A type nobody has
  contributed enough examples of is dropped from the run entirely, so adding
  or filling a class changes both the size *and* the index of every class
  after it alphabetically.

That is why the committed `warp/models/screen_classifier.pt` has a 7-class
head with labels `BOFFS, GROUND_EQ, GROUND_MIXED, SPACE_EQ, SPACE_MIXED,
SPECIALIZATIONS, TRAITS` — alphabetical, and four short of the admitted list.

`ScreenTypeClassifier` therefore reads `n_classes` from the checkpoint's
`classifier.3.weight` rather than from a constant or a metadata file. Before
that it assumed 7, a central run published an 8-class head, and the classifier
stopped loading altogether:

```
ScreenClassifier: model load failed: size mismatch for classifier.3.weight:
copying a param with shape torch.Size([8, 1024]) from checkpoint, the shape
in current model is torch.Size([7, 1024])
```

Every screenshot then took the non-ML path, and nothing in the UI said so.

Names come from `screen_classifier_labels.json`, published beside the weights
in the same `create_commit`, so the two are never separable on HuggingFace.

**The label map is part of the weights, and both sides now enforce that.**

| Rule | Where | What it prevents |
|---|---|---|
| Weights are not installed unless their label map arrived in the same download | `ModelUpdater._drop_unpaired`, driven by `_PAIRED_FILES` | New weights read through the previous run's names |
| The "download if missing" check tests both files, not just the `.pt` | `ModelUpdater._ensure_screen_classifier` | A partial download becoming permanent, since the old guard saw the `.pt` and returned |
| No label map means the model is refused, not guessed | `ScreenTypeClassifier._load` | Confidently wrong screen types |
| A label count that disagrees with the head size refuses the model | `ScreenTypeClassifier._load` | An answer the client cannot name |

The client's `SCREEN_TYPES` list is **not** a fallback for the label map and is
not used as one. It is in declaration order while the head is alphabetical, so
reading index *n* through it does not give an approximate name, it gives a
different class — against the shipped 7-class model, index 0 would report
`SPACE_EQ` and mean `BOFFS`. Worse, the list has 9 entries, and the count
climbs through 9 as classes reach the minimum sample threshold, so the count
check would not have caught it. The list survives only as the vocabulary
`add_session_example` validates against.

When the model is refused, `classify` returns `('', 0.0)`, the k-NN session
path still applies, and the user can set the type in WARP CORE. Refusing is
therefore strictly better than guessing: a wrong screen type sends the whole
screenshot down the wrong recognition path and is indistinguishable from a
right one in the UI.

`SKILLS` covers the captain skill tree (space/ground tabs). `SPACE_SKILLS` and
`GROUND_SKILLS` are post-hoc environment refinements of it — they are not
trained classes. `SPACE_BOFFS` and `GROUND_BOFFS`, despite following the same
naming pattern, *are* trained classes in the backend list. Both `SKILLS` and
`DISCARD` skip all recognition: WARP returns an empty `ImportResult` for images
classified into either (conf ≥ 0.50).

### Central training process (ArcFace embedder)

File: `sets-warp-backend/admin_train_metric.py`

The embedder follows the same community data path as the classifiers above.
It reads curated crops from `data/annotations.jsonl` (built by
`democratic_merge_crops.py`), downloads the crop images from HF by SHA,
trains an EfficientNet-B0 backbone with an ArcFace projection head, builds
a k-NN gallery from the full training set, and uploads the result to
`sets-sto/warp-knowledge/models/`.

```
1. Read data/annotations.jsonl → (sha, name, slot) tuples
2. Download crop PNGs by SHA from sets-sto/sto-icon-dataset
3. Stratified train/val split (80/20)
4. Train EfficientNet-B0 + ArcFace (margin=0.5, scale=30, 256-d embed)
5. Build gallery: embed every training crop, store per-class centroid
6. Save: icon_embedder.pt, embedding_index.npz, embedder_label_map.json
7. Upload to sets-sto/warp-knowledge/models/
```

#### Virtual gallery classes (`__inactive__`, `__empty__`)

The embedder gallery **must** include `__inactive__` and `__empty__` as
classes. Without them, the k-NN lookup has no "none of the above" option —
an inactive cell (uniform navy-blue X) snaps to the nearest real ability
(e.g. "Charged Particle Burst" at 0.93 conf) because the embedding space
has no closer alternative.

These virtual labels enter the pipeline through the same democratic voting
path as real abilities. Client `sync.py` uploads crops labelled
`__inactive__` / `__empty__` to staging; `democratic_merge_crops.py`
allows them through its poison filter (see
[`DATA_LIFECYCLE.md` §8](DATA_LIFECYCLE.md#8-the-poison-filter) for the
filter's current policy); the central trainer treats them as ordinary
classes.

The pHash override path in `VIRTUAL_OVERRIDE_CONF` in `icon_matcher.py` independently suppresses
virtual names (`name.startswith('__') → suppress=True`), so even if the
k-NN returns `__inactive__`, it is never written to `knowledge.json` as a
hard override. The gallery classes are defensive — they exist so the
embedding space has somewhere to map empty/inactive crops *instead of*
mapping them to real abilities.

#### Reaching items nobody has confirmed (wiki-art enrolment)

The gallery ships pre-built from confirmed crops, so an item no contributor
has ever confirmed has no entry and cannot be matched at all. The wiki PNG
for that item is already on disk — it is the same icon library the template
matcher uses — and the embedder is a function, so a usable reference costs
one forward pass and no training. `SETSIconMatcher._enroll_wiki_art` adds one
for every item the gallery lacks, marked as art so the rest of the pipeline
can tell an art reference from a confirmed crop.

The naming problem this creates, and the rule that answers it, matter more
than the mechanism. The wiki files its art under names that describe the
*picture*, not the item: `Impulse Engines (23c)` for the 23rd-century
artwork, `Adaptive Defense (ground)` and `Adaptive Defense (space)` for two
traits cargo stores as separate rows under one `name`, and further
qualifiers for colour, faction and reputation. Cargo carries none of those
distinctions in the item name, so art enrolled under the filename verbatim
enters the gallery under a label nothing downstream can resolve.

`_base_item_name` folds them, driven by the item names cargo currently
carries rather than by the spelling of the tag — because the tag is not
reliably a variant marker. `Modified Phaser Pistol (23c.)` is a real item
name, parenthesis and all. The order of the checks is what keeps both
readings working:

| Case | Result |
|---|---|
| the name is already an item cargo carries | leave it — a tagged item name |
| else the base name is an item | fold onto the base |
| neither | leave it — nothing to fold onto |

Folding a space and a ground variant onto one name is correct rather than
lossy. Cargo stores them as two rows under a single `name`, so one name is
all this program can emit for either, and the gallery holds many vectors per
class by design. The pair becomes two references for one label, and whichever
wins the k-NN returns the same, correct name.

Measured 2026-09-04 over the 4406-icon library: 155 icons carry a
parenthesised tag, 109 are cargo names outright, 46 fold onto a base, and
none falls outside those two cases — which is what makes the rule safe to
apply to any tag rather than to an enumerated list. The rule previously knew
only the era tag, so 12 icons entered the gallery unresolvable; two of them
reached confirmed training data, and one beat the correct cargo name in a
merge vote. See [`DATA_LIFECYCLE.md`](DATA_LIFECYCLE.md) for the gate that
now stops a name like that being confirmed.

### Ship type / tier OCR correction map

File: `sets-warp-backend/admin_train.py` — `collect_text_corrections()`

```
1. Filter staging annotations where slot in {'Ship Type', 'Ship Tier'}
2. For each (ml_name, name) pair where ml_name != '' and ml_name != name:
     votes[ml_name][install_id] = name   ← 1 vote per install_id
3. Democratic vote per ml_name key → majority corrected_name wins
4. Build ship_type_corrections.json:
     {"F1eet Support Cruiser": "Fleet Support Cruiser", ...}
5. Upload to sets-sto/warp-knowledge/models/ship_type_corrections.json
```

Applied in `warp/recognition/text_extractor.py` immediately after OCR reads
the ship type/tier text, before fuzzy ShipDB lookup. Ships with OCR errors
that multiple users have corrected are instantly fixed for all clients.

### model_version.json

Published after each successful central training run — unless the run is
refused. Both trainers compare their result against the version currently
served and stop if the class count fell below 90% of it, or accuracy by more
than ten points; the previously published model then stays in place and the
workflow fails. See [`DATA_LIFECYCLE.md`](DATA_LIFECYCLE.md) §5b for why the
thresholds are loose.

The values below show the shape, not a current release:

```json
{
  "version": "<git sha>",
  "trained_at": "2026-03-23T14:00:00Z",
  "n_classes": 2933,
  "val_acc": 0.87,
  "n_samples": 12450,
  "n_users": 18,
  "screen_trained_at": "2026-03-23T14:00:00Z",
  "screen_val_acc": 0.95,
  "screen_n_samples": 340
}
```

---

## 4. Community model delivery (HuggingFace → local)

### On first install (cold-start splash)

The cold-start splash described in
[`SYNC_ARCHITECTURE.md`](SYNC_ARCHITECTURE.md) §3 owns the first-install
model download. Phase `model` of the splash runs the same `ModelUpdater`
that powers the periodic refresh, with `force=True` so the 15 min skip
guard is ignored — the very first launch always pulls every required file:

```
models/icon_classifier.pt          (required)
models/label_map.json              (required)
models/icon_classifier_meta.json
models/model_version.json
models/screen_classifier.pt
models/screen_classifier_labels.json
models/icon_embedder.pt            (optional, ArcFace)
models/embedder_label_map.json     (optional)
models/embedding_index.npz         (optional)
```

After the splash completes, `~/.config/warp/startup_sync_done` is written
and subsequent launches go through the periodic refresh path described
below.

### Ongoing updates (ModelUpdater)

File: `warp/trainer/model_updater.py`

Fired by `SyncCoordinator` as the `model` step of every refresh cycle
(launch + every 60 minutes, see
[`SYNC_ARCHITECTURE.md`](SYNC_ARCHITECTURE.md)). Internally rate-limited.

```
1. Check rate limit: skip if last check was < 15 min ago
   (_CHECK_INTERVAL_HOURS = 0.25 in model_updater.py)
2. GET https://sets-sto-warp-backend.hf.space/model/version
   → returns {available, trained_at, n_classes, val_acc,
              embedder_trained_at, embedder_n_classes, embedder_recall, ...}
3. Compare remote trained_at vs local model_version.json trained_at:
     remote > local  → download and install new model (_MODEL_FILES)
     remote ≤ local  → compare embedder timestamps (step 3b)
3b. Compare remote embedder_trained_at vs local icon_embedder_meta.json
    trained_at (_embedder_is_outdated):
     remote > local  → download _EMBEDDER_FILES only
     otherwise       → skip (local is current)
4. Download the selected file list from HF via hf_hub_download
5. Copy files to the models dir atomically
6. Call SETSIconMatcher.reset_ml_session() to reload immediately
7. Save check timestamp to model_version_remote_cache.json
```

**Demotion guard.** The download is only installed if the remote
`trained_at` is **strictly later** than the local one. A tier-down or
class-count regression that would silently downgrade the model is rejected
in the same check — see the `1.0.10` Changelog entry on tier corrections
for the user-visible symptom this prevents.

**Two clocks, not one.** The softmax classifier
(`train_central_model.yml`, hourly, only retrains once ≥ 10 new crops have
been merged) and the ArcFace embedder (`train_metric_model.yml`, daily)
are published by independent workflows, so their `trained_at` stamps drift
apart. The embedder is the primary matcher (priority 0 in
`icon_matcher.py`), so gating it on the classifier's timestamp would freeze
it on every install whenever crop intake stalls. Step 3b is what keeps the
two independent. Backends that do not yet report `embedder_trained_at`
simply fall through to the old behaviour.

Independently of both, `_embedder_needs_refresh()` still forces a **full**
redownload when the embedder files are missing or carry pre-2026-05-16
snake_case labels.

### screen_classifier fallback

If `screen_classifier.pt` is missing (e.g., first install before bootstrap
completes, or manual deletion), `_ensure_screen_classifier()` downloads it
immediately, bypassing the 15 min rate limit. This runs on every
ModelUpdater check.

---

## 5. Full pipeline diagram

The end-to-end data flow from user confirmation, through HF staging, the
four democratic mergers, central training, and back to every install lives
in its own document: [`DATA_LIFECYCLE.md`](DATA_LIFECYCLE.md).

That document is the canonical reference for the staging-vs-data split,
the Z3 asymmetric thresholds, drain-on-promote, and the audit safety net.
This file (§§1–4 above) covers the ML model story; `DATA_LIFECYCLE.md`
covers the *data* story underneath it.

---

## 6. Priority rules

When WARP matches an icon, the following priority applies:

```
1. Community pHash knowledge override   (~/.cache/warp/knowledge_cache.json)
   — exact perceptual hash match, used only if the crop also resembles
     the gallery's pictures of the named item (see below)
2. Template matching + HSV histogram    (community crop library + cargo icons)
3. ArcFace embedder k-NN                (icon_embedder.pt + embedding_index.npz)
4. Softmax classifier                   (icon_classifier.pt)
5. Session examples                     (crops confirmed during the current
                                         run, used to bridge the gap before
                                         the next central retrain)
```

The classifier and embedder are the *same* files for every install —
`warp/models/icon_classifier.pt`, `warp/models/icon_embedder.pt`. There is
no per-user variant on disk. ModelUpdater replaces them only when the
remote `trained_at` is strictly later than the local one (§4).

### A hash hit is a claim about a picture (2026-09-25)

The community table maps a 64-bit perceptual hash (`_compute_phash`) to the
name the community voted for. What a hit establishes is narrower than it
looks: *some* picture with this hash was voted to be X. The hash is built
from greyscale and keeps little: across the 3891 entries a median of 12 of
its 64 bits is set, so different pictures do share hashes. A hit used to
be taken as the answer at 1.00, and on the screenshot that exposed this an
Omni-Directional Pahvan beam array was read as a `Phaser Turret`.

So a hit is now checked against pictures of X before it is used.
`SETSIconMatcher._knowledge_picture_sim` takes the crop's best similarity to
gallery rows labelled X, which are the crops the community confirmed as X
plus X's wiki art. It costs no extra model run: the cross-check that already
asks the embedder "is this slot empty?" on every hit leaves the crop's
similarity to every gallery row behind, and the check reads it from there. At or above
`KNOWLEDGE_PICTURE_MIN_SIM` (0.40) the hit stands as the community's
verdict at 1.00. Below it the hit is treated as a collision: it is logged
with the hash and both names, and the slot is matched as if there had been
no hit. When no embedder is loaded nothing can be compared, so the name is
still offered, but at `KNOWLEDGE_UNVERIFIED_CONF` (0.74). That is below
WARP CORE's default auto-accept threshold, so a person looks at it first.

Measured over 7308 user-confirmed crops, 4524 of which hit the table
(`dev/phash_verify_measure.py`, and the shipped `match()` reproduced its
prediction exactly in `dev/phash_verify_shipped.py`):

| | hits | kept after the check |
|---|---|---|
| hit names the confirmed item | 4163 | 4157 |
| hit names another item | 361 | 38 |

These figures were re-measured after a defect in the evaluation itself:
annotations are keyed by image hash, and joining them to screenshots by
filename had cut 224 of 8228 confirmed boxes out of the wrong image, since
names such as `image.png` repeat. The first run gave 4427 hits, 357 wrong
and 319 of them caught. The conclusion did not change, and the numbers
above come from the corrected join (`dev/gt_crops.py`).

The six correct hits lost are a floor, not an estimate: many of these crops
are in the gallery themselves and match themselves. The 38 wrong hits that
survive are mostly pairs of items whose icons look alike to the embedder
(similarity 0.8–0.9): `Auxiliary Battery` and `Auxiliary Battery - Large`,
`Advanced` and `Sensor-Linked Phaser Beam Array`. They are not the same
art. On the wiki the Sensor-Linked array has gold rails where the Advanced
one has grey, and the Large battery shows two units. None of them is in
`icon_equivalence.json`, correctly. The picture check cannot separate them
because the embedder cannot yet; that is a recognition gap, not a hash one.

One hash can also carry votes for several items. Measured on the live
repository on 2026-09-25, one hash had been voted a console, a trait, a
beam bank and two more. The backend used to keep only the leader and forget
the rest, so the check would have rejected every picture but the leader's
as a collision. `knowledge.json` now carries the full tally, `votes`, phash
→ {name: votes}. `WARPSyncClient.get_knowledge_votes` returns it, and
`_knowledge_names` gives the check every name. The chosen name is the one
whose pictures the crop resembles most, provided it clears the floor.
Without an embedder the most-voted name is offered at
`KNOWLEDGE_UNVERIFIED_CONF`. A backend that predates the tally sends only
`knowledge`, and its leader stands alone. Built with the backend's real merger over the live table and
the contributions left on HF, the tally raised correct knowledge answers
on user-confirmed crops from 4174 to 4270 and lowered wrong ones from 39
to 37 (`dev/phash_tally_effect.py`).

### A session example must have structure (2026-08-31)

Both template stages score with `cv2.matchTemplate(..., TM_CCOEFF_NORMED)`,
which divides by the template's standard deviation. For a template of one
flat colour that is 0/0, and OpenCV's guard resolves it to exactly **1.00 —
against every query**, colourful or not. The reverse is harmless: a flat
*query* against a real template scores 0.00, which is the right answer.

Two such crops (pure black, one 29×22 from a `Boff Universal` slot and one
90×70 from `Body Armor`, both labelled `__empty__`) reached the community
pool. Their label is not wrong — that is why the seed-time poison guard, which
looks for a *colourful* crop under a virtual label, let them through — but as
templates they are degenerate. Every real icon in every screenshot was
therefore also offered `__empty__` at `0.80 + 0.20 · histogram` ≈ 0.80–0.85,
and the anti-virtual suppression rules (mapped in
`docs/client_user_view_filter.md` §2.2) had to shoot it down slot by slot.

Observable without reading any of this: three `guard fired` warnings per
filled slot in `warp_detection.log`, and a `session` column that never left
the 0.80s in the per-image match summary.

Measured over the 301 rows `recog_runs.jsonl` held before the guard — the
append-only record of what each stage scored, one line per matched slot —
**not one session score fell between 0 and 0.80** (231 at or above it, 70 at
zero, where a knowledge override short-circuited the stage). No genuine
session match below that floor could surface at all.

`SETSIconMatcher._template_is_degenerate` now rejects such a crop in
`add_session_example`, which every seeder and the trainer's Accept callback
go through. The rejection is logged at info level, so a name missing from the
pool is visible rather than silent:

```
WARP: session example rejected — '__empty__' (community) is a single flat
colour, which would match every query at 1.00
```

The cutoff is exact zero variance rather than a tolerance: one differing pixel
in 4096 already scores 0.006 against an unrelated query, so there is no grey
zone to allow for. 1373 virtual examples remain in the pool afterwards, so the
ability to recognise a genuinely empty slot is unaffected.

The guard is client-side only. The annotation is written before the session
example is offered, so a flat crop still reaches `annotations.json` and still
uploads; the two community crops keep arriving on every install until they are
rejected upstream. What changes is that they can no longer answer a query.

---

## 7. Data stored on HuggingFace

| Repo | Path | Contents |
|------|------|----------|
| `sets-sto/sto-icon-dataset` | `staging/<install_id>/crops/` | Icon crop PNGs (64×64) + Ship Type/Tier text crops |
| `sets-sto/sto-icon-dataset` | `staging/<install_id>/annotations.jsonl` | Label records (icon + text; includes `ml_name` for text slots) |
| `sets-sto/sto-icon-dataset` | `staging/<install_id>/screen_types/<TYPE>/` | Screen type PNGs |
| `sets-sto/warp-knowledge` | `models/` | Trained .pt files, label_map.json, model_version.json |
| `sets-sto/warp-knowledge` | `models/ship_type_corrections.json` | OCR correction map: `{raw_ocr: corrected_name}` |
| `sets-sto/warp-knowledge` | `knowledge.json` | pHash → item_name community overrides, and `votes`: every name each hash was voted for |
| `sets-sto/warp-knowledge` | `models/training_manifest.json` | SHA set from last training run |

---

## 8. Key files (local)

| File | Purpose |
|------|---------|
| `warp/models/icon_classifier.pt` | Active icon classifier (local or community) |
| `warp/models/screen_classifier.pt` | Active screen type classifier |
| `warp/models/label_map.json` | `{index: item_name}` for icon classifier |
| `warp/models/screen_classifier_labels.json` | `{index: screen_type}` for screen classifier |
| `warp/models/model_version.json` | `trained_at`, `source`, `val_acc` of current model |
| `warp/models/model_version_remote_cache.json` | Timestamp of last remote version check |
| `warp/training_data/annotations.json` | All confirmed annotations |
| `warp/training_data/crops/<sha>.png` | Confirmed icon crops + Ship Type/Tier text region crops |
| `warp/training_data/screen_types/<TYPE>/<file>.png` | Confirmed screen type screenshots |
| `warp/models/ship_type_corrections.json` | OCR correction map downloaded from community (optional) |
| `warp/knowledge/install_id.txt` | Anonymous UUID identifying this installation |

---

## 9. Local bootstrap trainer (one-shot)

The icon embedder (`icon_embedder.pt` + `embedding_index.npz`) is normally
trained centrally on community contributions — the same democratic-voting
pipeline described above for the softmax classifier. When the central
embedder is missing entire class regions (e.g. ground BOFF abilities, whose
icons rarely show up in space-side screenshots), waiting for those crops to
arrive via HF staging is infeasible: 10,600 synthetic crops at
1000 uploads/install/UTC day per `MAX_DAILY_UPLOADS` = ~11 days through
normal sync, plus the wall-clock cost of waiting for those days to pass.

For that one-time gap-closure, sto-warp ships a **local** bootstrap path:

```
1. Generate synthetic crops from cargo wiki PNGs (64×64 BGR, augmented).
2. Train the ArcFace embedder locally on real + synthetic crops.
3. Manually upload the resulting .pt + gallery + label map to central HF.
4. Resume normal central training thereafter.
```

This path is **not** a replacement for central training — it is a single
intervention to seed unknown classes into the gallery so subsequent
community contributions have something to vote against.

### 9.1 Synthetic crop generator

File: `warp/trainer/synthetic_crop_generator.py`

```bash
python -m warp.trainer.synthetic_crop_generator --env ground -n 100
```

For each ability name in `boff_abilities()[env]`, loads the cargo wiki PNG
from `icons_dir()` and emits `n` augmented 64×64 BGR variants under:

```
~/.local/share/warp/training_data/synthetic_crops/<env>/<class_slug>/<seq>.png
```

Augmentations approximate STO's icon-on-UI domain:

| Step | Detail |
|------|--------|
| Background | Random dark BGR gradient + Gaussian noise (mimics navy/grey UI surfaces) |
| Composite | Alpha-aware (icons are BGRA with transparent corners) |
| Scale | Icon height uniform in [44, 60] px before centring |
| Position | ±3 px bbox jitter from canvas centre |
| Colour | HSV: ±15 % brightness, ±20 % saturation, ±5° hue |
| Cooldown | 10 % chance of radial dim sweep overlay |
| Codec | JPEG re-encode at quality 70–95 (gameplay screens are JPEG) |

Synthetic crops live alongside real crops in `training_data/` but are
**never** synced to HF — they are local bootstrap data, not community
contributions.

### 9.2 Local embedder trainer

File: `warp/trainer/embedder_trainer.py`

```bash
# Generate + train in one shot
python -m warp.trainer.embedder_trainer --generate-synthetic --env ground -n 100 --train

# Or train against pre-generated crops
python -m warp.trainer.embedder_trainer --train
```

Architecture mirrors the central trainer (`admin_train_metric.py`):

| Component | Value |
|-----------|-------|
| Backbone | EfficientNet-B0 (ImageNet warm start, then warm-started from existing `icon_embedder.pt` when present) |
| Projection | Linear → 256-d → L2-normalize |
| Head | ArcFace, margin = 0.5, scale = 30.0 |
| Sampler | PK: P = 8 classes × K = 4 samples / batch = 32 |
| Optimizer | AdamW, lr = 3e-4, CosineAnnealingLR |
| Loss | CrossEntropy on ArcFace logits |
| Stop | Early stop on val recall@1, patience = 5, max 30 epochs |

Data sources:
- **Real crops** from `~/.local/share/warp/training_data/crops/` —
  filenames are `<slot>__<slug>__<hash>.png`; canonical labels are pulled
  from `crops/crop_index.json` (key `name`) so the resulting label map
  stays consistent with the existing embedder.
- **Synthetic crops** from `training_data/synthetic_crops/<env>/<slug>/` —
  class slug is reverse-mapped to canonical name via
  `boff_abilities()[env]`.

Outputs in `userdata.models_dir()` (= `~/.cache/warp/models/`):

```
icon_embedder.pt           — backbone + projection state_dict
embedding_index.npz        — full-train gallery (no aug)
embedder_label_map.json    — {index: canonical_name}
icon_embedder_meta.json    — hyper-params + val_recall@1 + source='local-bootstrap'
```

### 9.3 Manual upload to central

After training, copy the four output files to
`sets-sto/warp-knowledge/models/` on HuggingFace (web UI or `huggingface-cli upload`).
The next `ModelUpdater` tick on every install will pull them via the same
flow as in §4. After upload, **central training resumes normal operation**
on community-contributed real crops — no further local intervention.

The upload is a one-time admin action; users do not run this themselves.

---

## 10. Open questions

1. **`add_session_example` silently drops the screen types it does not know.**
   Its guard is `stype not in SCREEN_TYPES`, and that list omits `SPACE_BOFFS`
   and `GROUND_BOFFS`, which the backend admits as trainable classes and which
   the user can pick in WARP CORE. Setting one of those types therefore
   contributes no session example, with nothing said in the UI or the log.

   Not the same defect as the label map — the session k-NN is a per-run
   convenience, so the cost is a weaker within-session assist, not a wrong
   answer. But it is a silent rejection, and the list it gates on is now kept
   for this purpose alone, which makes it the only thing still asserting a
   fixed vocabulary on the client. Decide whether that vocabulary should come
   from the published label map instead.
