# sto-warp — Claude Code Context

## Project overview

**sto-warp** is the standalone successor to the WARP/WARP CORE modules that
previously lived inside **sets-warp**. It is a Star Trek Online screenshot
recognition + ML training toolkit, distributed as its own pip / pipx package
(`sto-warp`) with no hard dependency on a build-planner GUI.

- **WARP** *(Weaponry & Armament Recognition Platform)* — screenshot recognition pipeline (OCR + layout + icon matching).
- **WARP CORE** — Qt trainer UI for reviewing recognition output and fine-tuning the EfficientNet / MobileNetV3 models.
- **Bridges** — thin adapters that publish recognition results to external build planners (e.g. SETS v3.0.0) live in **separate** packages and consume sto-warp as a library.

**Stack:** Python 3.14+, PySide6, OpenCV, PyTorch, EasyOCR
**Entry point:** `sto-warp` (console script, installed by pipx) → `warp.cli:main`

`warp/cli.py` is the single entry point — an argparse dispatcher, not a
console-only layer. The windows themselves live in `warp/gui/` and
`warp/trainer/`:

| Subcommand | Runs |
|---|---|
| *(none)* / `launcher` | GUI — `warp.gui.launcher:main` (WARP + WARP CORE tabs) |
| `gui` | GUI — `warp.gui.warp_window:main` |
| `warp-core` | GUI — `WarpCoreWindow` (`warp/trainer/trainer_window.py`) |
| `check` | console — verifies the recognition modules import |
| `install-desktop` | console — writes the menu entry (.desktop / .lnk) |

There is no `warp/app.py` in this repo — that name belongs to the SETS
bridge (see "Things NOT in this repo").

---

## Language rules

**All code must be in English** — comments, log messages, docstrings, variable
names, string literals visible in logs. No Polish in source files. When editing
existing code that contains Polish log messages or comments, translate them to
English.

---

## Rules

1. First think through the problem, read the codebase for relevant files.
2. Before you make any major changes, check in with me and I will verify the plan.
3. Please every step of the way just give me a high level explanation of what changes you made.
4. Make every task and code change you do as simple as possible yet not naive. We want to avoid making any massive or complex changes. Every change should impact as little code as possible. Everything is about simplicity.
5. Maintain a documentation file that describes how the architecture of the app works inside and out.
6. Maintain documentation files in the project. Recognize which are technical and which are more human-readable (manual, program description, readme).
7. Never speculate about code you have not opened. If the user references a specific file, you MUST read the file before answering. Make sure to investigate and read relevant files BEFORE answering questions about the codebase. Never make any claims about code before investigating unless you are certain of the correct answer — give grounded and hallucination-free answers.
8. Never use workarounds. Especially never change existing code just to fix your freshly made problem. Only recent changes are supposed to be fixed. If situation requires fixing existing code it requires user one-time approval.
9. NEVER EVER USE -Force or -f (force attribute) in terminal commands. It is strictly forbidden! If there is no other way you NEED to ask the user to run the command in terminal themselves providing justification.

---

## CORE ARCHITECTURAL RULE

**WARP = detection. WARP CORE = trains WARP. `annotations.json` = training data ONLY.**

WARP must NEVER use `annotations.json` as direct import output. If WARP falls
back to reading user-confirmed ground truth instead of performing detection,
we:

- Hide real detection bugs behind seemingly-good recognition results
- Cannot measure actual recognition quality
- Defeat the whole purpose of improving the detector

Only **WARP CORE** (the trainer) reads annotations — to display for user review
and feed back into training data for the EfficientNet / MobileNetV3 models.

Enforcement: a single boolean (`_use_confirmed = _is_trainer_call`) gates all
annotation lookups inside the recognition pipeline. Do NOT re-introduce the
old `'MIXED' in build_type or _is_trainer_call` shortcut without explicit
user approval.

---

## Testing

When modifying or adding code under `warp/`, always write or update a
corresponding test in `tests/`. Follow these conventions:

- **Framework:** pytest (not unittest). Use fixtures, `monkeypatch`, `tmp_path`.
- **Isolation:** never touch the user's real XDG dirs or network — use
  `monkeypatch.setenv` to redirect `WARP_CACHE_DIR`, `XDG_CONFIG_HOME`, etc.
  to `tmp_path`.
- **GUI tests:** `conftest.py` sets `QT_QPA_PLATFORM=offscreen` globally.
  Create `QApplication` via `QApplication.instance() or QApplication([])`.
  Use `addCleanup(widget.close)` for widget teardown.
- **Heavy deps:** if a test needs opencv / torch / easyocr, gate it with
  `@pytest.mark.skipif(not _has_dep(), reason='...')` so the lightweight
  suite stays green without the full ML stack.
- **Naming:** `test_<module_under_test>.py`, e.g. `test_userdata.py` tests
  `warp/userdata.py`. Test functions: `test_<behaviour_being_verified>`.
- **Scope:** keep tests focused — one assertion concept per test. Prefer
  many small tests over few large ones.

Run the suite: `python -m pytest tests/ -v`

---

## Diagnostic scripts (`dev/`)

`tests/` holds **validation tests only** — code that asserts the program
behaves correctly, and that is expected to pass. Nothing else belongs there.

Benchmarks, ad-hoc probes, proofs of concept and one-off calibration scripts
live in **`dev/`**, which is ignored by git in its entirety. They are the
maintainer's local working set: never committed, never run by CI, free to be
messy or broken.

References to specific `dev/*.py` paths in docs, code comments or memory are
for on-disk reproduction by the maintainer, not a guarantee the file exists in
any checkout.

---

## Repository structure

```
sto-warp/
├── pyproject.toml                 # package metadata, pipx entry points
├── README.md                      # human-readable (manual)
├── CHANGELOG.md                   # release notes
├── INSTALLATION.md                # install / pipx / packages
├── PROJECT_CONTEXT.md             # design context (carried over)
├── CLAUDE.md                      # this file
├── docs/                          # technical + user docs
├── tests/                         # validation tests only (pytest)
├── dev/                           # maintainer scripts — gitignored entirely
├── tools/                         # repo-level helper (crops tarball)
├── packaging/                     # native packages (aur, windows)
└── warp/
    ├── cli.py                     # console script → subcommand dispatcher
    ├── debug.py                   # standalone logger (replaces src.setsdebug)
    ├── warp_importer.py           # recognition orchestrator (ship-first)
    ├── build_writer.py            # result → SETS-shaped build dict
    ├── sets_export.py             # build dict → SETS v3.0.0 JSON
    ├── userdata.py, config.py     # XDG paths, settings
    ├── themes.py, style.py, ui_helpers.py, folder_picker.py
    ├── recognition/               # detection pipeline
    │   ├── boff_keys.py, boff_marker.py
    │   ├── eq_geometry.py, ground_eq_geometry.py
    │   ├── icon_matcher.py, layout_detector.py
    │   ├── screen_classifier.py, skill_grid.py
    │   ├── text_extractor.py, trait_grid.py
    │   └── ui_translations.py
    ├── gui/                       # WARP windows (launcher, warp_window, …)
    ├── trainer/                   # WARP CORE (trainer_window, sync, …)
    ├── knowledge/                 # community pHash sync
    │   └── sync_client.py
    ├── data/                      # cargo access + committed baseline
    │   ├── cargo.py, asset_sync.py, empty_build.py
    │   └── baseline/              # offline fallback snapshot (COMMITTED)
    └── tools/                     # maintainer CLI (staging review, scrub, …)
```

Runtime artefacts (created or downloaded at first run, never committed):

- `warp/models/` — `icon_classifier.pt`, `screen_classifier.pt`, label maps
- `warp/training_data/` — annotations, crops, recognition history
- `~/.config/warp/` — everything user-local: `cache/` (cargo JSON + ETag
  meta), `icons/` (icon DB from `STOCD/SETS-Data`), logs, knowledge.json,
  install_id

Note: downloads land under `~/.config/warp/`, **not** under `warp/data/`
(`cargo._cache_dir()` / `cargo.icons_dir()`). The `.gitignore` entries for
`warp/data/icons/`, `warp/data/cache/` and `warp/data/item_db.json` are
legacy guards — those paths are never created. Everything under
`warp/data/` is source, including the committed `baseline/` snapshot.

---

## Logging

```python
from warp.debug import log
log.info('message')   # stderr + the channel's log file
log.debug('...')
log.warning('...')
```

Three channels (`warp/debug.py:36`), each with its own file under
`~/.config/warp/` (override with `WARP_LOG_DIR` / `XDG_CONFIG_HOME`); the
previous run is rotated to `.log.bak`:

| Channel | File |
|---|---|
| `detection` | `warp_detection.log` |
| `detection_core` | `warp_detection_core.log` |
| `system` | `warp_system.log` |

**"Logging" always means both:** writing to the log file **and** printing to
stderr. Never log to only one destination. Always use `warp.debug.log` — do
NOT introduce `logging.getLogger(__name__)`, as that bypasses the file
mirror.

All WARP CORE logs are prefixed with context (e.g. `WarpImporter:`,
`LayoutDetector:`, `AW.zoom`).

---

## Data sources (no SETS dependency)

Cargo and reference data are fetched on first run and cached locally. Two
sources are tried in order (`UPSTREAM_BASES` in `warp/data/cargo.py`):

1. **`raman78/warp-cargo-data`** — our own mirror of the stowiki cargo
   tables. A superset of the fields SETS-Data carries.
2. **`STOCD/SETS-Data`** — the community mirror, kept as a fallback.

The mirror is **not** on an automatic schedule: it is rebuilt best-effort by
the maintainer, by hand, on a machine that has to be switched on. Treat its
freshness as "usually no worse than the fallback", never as a guarantee, and
do not describe it to users as self-refreshing.

Both were verified to carry the same records by building all caches from
each source and diffing all 47 buckets. An ETag is only replayed against
the source that issued it, so a fallback cannot answer 304 with the other
mirror's bytes.

They are **not** typed the same way. `ship_list.json` from our mirror is
the raw `Special:CargoExport` output — every field a string, list columns
comma-joined — while SETS-Data and `warp/data/baseline/` serve lists and
numbers. `cargo._normalise_ship` coerces both into the typed shape, and
every reader of that file must apply it (`cargo._build_ships`,
`warp_importer.ShipDB._load`). Never assume a cargo field's type from one
source alone; check a live cache and the baseline. See
`docs/CARGO_DATA_PLAN.md`.

| File | Purpose |
|---|---|
| `equipment.json` | weapon / shield / console / device DB |
| `traits.json` | personal / starship / reputation traits |
| `boff_abilities.json` | BOFF ability metadata (rank Roman numerals) |
| `ships.json` | ship roster (used for type-first disambiguation) |

One more file comes from a different path: `scraped/scraped_ground_weapons.json`
(`OVERLAY_FILES`, `OVERLAY_BASE`). Elite Fleet, Colony Security and K-13 ground
weapons are listed on a wiki page but stored in no cargo table, so they are
harvested from that page and published beside the mirror. Only our own mirror
serves it, it is **optional** (a missing overlay logs a warning and continues),
and `cargo._merge_overlay` never lets an overlay row shadow a real cargo row.
Each row carries `source`; the publisher drops rows the cargo tables start
carrying, so the overlay is designed to shrink to nothing.

Cache path: `~/.config/warp/cache/` (per-file mtime + 24 h refresh window;
ETag-aware via `If-None-Match` when available).

---

## Distribution

Primary: **pipx** — `pipx install sto-warp` installs into an isolated venv
and exposes the `sto-warp` console script. Python 3.14+ required.

Native packages (planned, in this order): Arch AUR, Debian/Ubuntu `.deb`,
Fedora COPR `.rpm`, Windows MSI/EXE.

PyPI name: **`sto-warp`**. Console-script name: **`sto-warp`**.

---

## Things NOT in this repo (left in sets-warp)

- SETS build planner (`src/app.py`, `buildupdater.py`, …)
- The thin SETS bridge layer (`warp/warp_button.py`, `warp/warp_dialog.py`,
  `warp/app.py`) — these will move to a separate `sets-warp-bridge` package
  that depends on `sto-warp` and `sets`.

`warp/warp_importer.py` and `warp/sets_export.py` **are** in this repo:
the importer drives the ship-first recognition strategy, and the exporter
writes a SETS v3.0.0 build JSON from the dict `warp.build_writer` produces.
Neither imports `src.*`.

When porting code from `sets-warp/warp/`, replace `from src.setsdebug import
log` with `from warp.debug import log`. Reject any new `from src.*` import.
