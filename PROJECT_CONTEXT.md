# sto-warp — Project Context

> **Scope of this document.** Design intent, repo layout, and a map of
> where to find the details. Implementation details live in the
> linked documents — this file does not duplicate them.

---

## 1. What sto-warp is

**sto-warp** is a Star Trek Online screenshot recognition + ML training
toolkit. It reads STO screenshots and identifies every slotted item
(weapons, consoles, traits, BOFF abilities, …), and ships with a
trainer UI to review/correct those results and feed them back into the
community model.

It is the standalone successor to the **WARP / WARP CORE** modules that
used to live inside the **sets-warp** build planner. The split was
motivated by:

- **Independent release cadence.** Recognition and trainer evolve
  faster than the SETS build planner; bundling them forced lockstep
  releases.
- **Distribution.** sto-warp ships as its own pipx-installable package
  (`sto-warp`, console script of the same name) with no hard
  dependency on a build planner.
- **Reusability.** Other planners or tools can consume sto-warp as a
  library; a thin **bridge package** (e.g. `sets-warp-bridge`) is the
  intended way to publish recognised builds into SETS or any other
  consumer.

Distribution: **pipx → PyPI → native packages** (Arch AUR, `.deb`,
`.rpm`, Windows MSI). See [`INSTALLATION.md`](INSTALLATION.md).

---

## 2. The two halves — WARP and WARP CORE

| Half        | Role                                                                  | Entry path                |
|-------------|-----------------------------------------------------------------------|---------------------------|
| **WARP**    | Detection pipeline. OCR + layout + icon matching → slot assignments.  | `warp/recognition/`       |
| **WARP CORE** | Trainer UI. Reviews WARP output, captures corrections, fine-tunes.  | `warp/trainer/`           |

Strict separation of concerns — enforced by the core rule in
[`CLAUDE.md`](CLAUDE.md):

> **WARP = detection. WARP CORE = trains WARP.
> `annotations.json` = training data ONLY.**

WARP must never short-circuit to `annotations.json` to "look up" what an
icon is. That data is ground truth for the trainer, not for the
detector. Bypassing this hides real recognition bugs.

---

## 3. Repo map (one line per area)

```
sto-warp/
├── pyproject.toml          # package metadata, entry point: sto-warp = warp.cli:main
├── README.md               # user-facing entry
├── INSTALLATION.md         # install methods
├── CHANGELOG.md            # release notes
├── CLAUDE.md               # AI-assisted-dev rules and core architectural rule
├── PROJECT_CONTEXT.md      # this file
├── docs/                   # technical + user docs (see §4)
└── warp/
    ├── cli.py              # console-script entry point
    ├── config.py           # XDG paths, install_id
    ├── debug.py            # standalone logger (file + stderr mirror)
    ├── recognition/        # detection — see docs/ML_PIPELINE.md, docs/warp_ml_roadmap.md
    ├── trainer/            # WARP CORE — trainer window, fine-tuning workers, sync
    ├── knowledge/          # community knowledge + crops sync — see docs/SYNC_ARCHITECTURE.md
    ├── data/               # cargo + asset sync — see docs/SYNC_ARCHITECTURE.md, docs/CARGO_DATA_PLAN.md
    ├── sync/               # (reserved for future sync utilities)
    ├── gui/                # launcher, cold-start splash, sync coordinator, log view
    ├── tools/              # developer utilities (scrapers, validators)
    ├── models/             # downloaded at runtime — icon/screen classifiers (.pt + labels)
    ├── resources/          # bundled icons / themes
    ├── training_data/      # local training crops & annotations (user-local)
    ├── warp_importer.py    # screenshot-to-build orchestration
    └── sets_export.py      # SETS v3.0.0 JSON exporter
```

Runtime artefacts (downloaded on first run, never committed):
`warp/models/` (.pt + label maps), `warp/data/` cache,
`~/.config/warp/` (install_id, logs, knowledge cache, marker file,
crops manifest).

---

## 4. Where to find the details

This file intentionally stops at "what exists and why". For the
internals of each area, the canonical references are:

### User-facing

| Document                                                        | Audience            | Covers                                                                                  |
|-----------------------------------------------------------------|---------------------|-----------------------------------------------------------------------------------------|
| [`README.md`](README.md)                                        | New user            | Install, first-run download, quick orientation                                          |
| [`INSTALLATION.md`](INSTALLATION.md)                            | Installer           | pipx, native packages, desktop entry, data locations                                    |
| [`docs/WARP_GUIDE.md`](docs/WARP_GUIDE.md)                      | End user            | Full walkthrough: launcher tabs, recognise→export, WARP CORE, first-run setup splash    |
| [`CHANGELOG.md`](CHANGELOG.md)                                  | Anyone              | Release notes (pre-1.0 — fresh)                                                         |

### Technical — subsystems

| Document                                                                  | Subsystem                                                                       |
|---------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| [`docs/SYNC_ARCHITECTURE.md`](docs/SYNC_ARCHITECTURE.md)                  | Cold-start splash, periodic refresh, TTLs, marker file, all 7 data sources      |
| [`docs/ML_PIPELINE.md`](docs/ML_PIPELINE.md)                              | Local + community training, model delivery, HF backend boundary                 |
| [`docs/warp_ml_roadmap.md`](docs/warp_ml_roadmap.md)                      | Layout-detector strategies, full-scan pipeline, current status                  |
| [`docs/BOFF_DETECTION.md`](docs/BOFF_DETECTION.md)                        | BOFF panel detection — colour markers + classifier                              |
| [`docs/TRAIT_DETECTION.md`](docs/TRAIT_DETECTION.md)                      | Trait grid detection — structure-first, ML probe per section                    |
| [`docs/sto_slots_rules.md`](docs/sto_slots_rules.md)                      | STO slot rules + how WARP enforces (or doesn't) each constraint                 |
| [`docs/CARGO_DATA_PLAN.md`](docs/CARGO_DATA_PLAN.md)                      | Cargo-data sourcing rationale (STOCD/SETS-Data over local scraper)              |
| [`docs/REMOTE_SYNC_AUDIT.md`](docs/REMOTE_SYNC_AUDIT.md)                  | Backend/HF capacity audit, channels in use, scaling headroom                    |
| [`docs/data_source_audit.md`](docs/data_source_audit.md)                  | Full data-flow audit (HF dataset structure, virtual classes, governance Zs)     |
| [`docs/client_user_view_filter.md`](docs/client_user_view_filter.md)      | Client-side filter map for virtual classes (`__empty__`, `__inactive__`, …)     |
| [`docs/FAST_CORRECTION_MODE.md`](docs/FAST_CORRECTION_MODE.md)            | Fast Correction Mode internals: staging, ephemeral annotations, send-back loop  |
| [`docs/gpu_setup.md`](docs/gpu_setup.md)                                  | Optional CUDA setup (embedder retraining only — most users don't need this)     |

### Repository rules

| Document                                                                  | Purpose                                                                         |
|---------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| [`CLAUDE.md`](CLAUDE.md)                                                  | Core architectural rule, language rules, logging conventions, repo discipline   |

---

## 5. Runtime stack (summary)

Full per-component rationale lives in the linked docs above; this
section only lists what is on disk.

- **Python 3.14+** — runtime.
- **PySide6** — Qt 6 GUI (launcher, WARP CORE, dialogs, splash).
- **PyTorch + torchvision** — local training and inference (native
  `.pt` for both icon classifier and screen classifier).
- **OpenCV (`opencv-python-headless`)** — template matching, crop
  geometry, histogram comparison.
- **EasyOCR** — ship name / tier / slot label extraction.
- **HuggingFace Hub** — read-only client-side: models, knowledge,
  community crops tarball, icon-equivalence map. Write side
  (contributions, screen-type uploads) is server-side only, on the
  sto-warp Space backend.
- **`requests`** — HTTP for GitHub raw / backend.

Logging policy: **always** through `warp.debug.log` (stderr + file
mirror at `~/.config/warp/warp_debug.log`). `logging.getLogger(...)`
is forbidden because it bypasses the file mirror. See
[`CLAUDE.md`](CLAUDE.md) for the full convention.

---

## 6. External boundaries (no SETS coupling at runtime)

sto-warp does not import the SETS build planner. The two interaction
modes intended for the planner are:

1. **Library use.** A bridge package imports sto-warp (`warp.*`) and
   forwards results into its own data model. The bridge owns all
   planner-specific knowledge.
2. **JSON export.** `warp/sets_export.py` produces a SETS-v3.0.0
   compatible build JSON that a planner can ingest as a file.

Data origins consumed by sto-warp itself:

- **`STOCD/SETS-Data`** (GitHub raw) — `equipment.json`, `traits.json`,
  `ships.json`, `boff_abilities.json`, item icon mirror, ship images.
  No live stowiki dependency.
- **sto-warp Space backend** (HF Spaces) — `/knowledge`,
  `/model/version`, contribution endpoints. Holds the only HF write
  token; the client is read-only against HF.
- **HF dataset `sets-sto/sto-icon-dataset`** — community-confirmed
  crops tarball + `icon_equivalence.json`.

For TTLs, fetch order, failure handling, and the splash that gates the
first run, see [`docs/SYNC_ARCHITECTURE.md`](docs/SYNC_ARCHITECTURE.md).

---

## 7. Active gaps and constraints

Tracked in their natural homes — listed here only as pointers:

- **Fore/aft weapon cross-validation gap** — see
  [`docs/sto_slots_rules.md`](docs/sto_slots_rules.md) ("Weaponry").
- **BOFF rank ambiguity in MIXED screens** — see
  [`docs/BOFF_DETECTION.md`](docs/BOFF_DETECTION.md).
- **Layout detection strategy ladder** — see
  [`docs/warp_ml_roadmap.md`](docs/warp_ml_roadmap.md).
- **Cargo / knowledge / model / crops freshness** — see
  [`docs/SYNC_ARCHITECTURE.md`](docs/SYNC_ARCHITECTURE.md) §7
  ("Cargo-staleness postmortem") for the most recent class-of-bug
  fix and its prevention.

When adding a new constraint or gap, document it in the subsystem doc
that owns it, then add a one-line pointer here if it crosses subsystem
boundaries.
