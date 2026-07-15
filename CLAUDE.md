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
**Entry point:** `sto-warp` (console script, installed by pipx) → `warp.app:main`

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

## Diagnostic scripts (`tests/diag_*.py`)

These are **local-only developer benchmarks / ad-hoc probes** — ignored by
git (`tests/diag_*.py` in `.gitignore`) and not part of the test suite.
References to specific `diag_*.py` paths in docs/memory are for on-disk
reproduction by the maintainer, not a guarantee the file is in the repo.

---

## Repository structure (target — populated incrementally)

```
sto-warp/
├── pyproject.toml                 # package metadata, pipx entry points
├── README.md                      # human-readable (manual)
├── CHANGELOG.md                   # release notes (fresh — pre-1.0)
├── INSTALLATION.md                # install / pipx / packages
├── PROJECT_CONTEXT.md             # design context (carried over)
├── CLAUDE.md                      # this file
├── docs/                          # technical + user docs
└── warp/
    ├── __init__.py
    ├── debug.py                   # standalone logger (replaces src.setsdebug)
    ├── recognition/               # detection pipeline
    │   ├── boff_keys.py
    │   ├── boff_marker.py
    │   ├── eq_geometry.py
    │   ├── ground_eq_geometry.py
    │   ├── icon_matcher.py
    │   ├── layout_detector.py
    │   ├── screen_classifier.py
    │   ├── text_extractor.py
    │   └── trait_grid.py
    ├── trainer/                   # WARP CORE
    │   ├── ocr_diag.py
    │   └── screen_type_trainer.py
    └── knowledge/                 # community pHash sync
        └── sync_client.py
```

Runtime artefacts (downloaded at first run, never committed):

- `warp/models/` — `icon_classifier.pt`, `screen_classifier.pt`, label maps
- `warp/data/` — cargo / icon DB (fetched from `STOCD/SETS-Data` GitHub raw)
- `~/.config/warp/` — user-local cache (knowledge.json, install_id, logs)

---

## Logging

```python
from warp.debug import log
log.info('message')   # stderr + ~/.config/warp/warp_debug.log
log.debug('...')
log.warning('...')
```

**"Logging" always means both:** writing to the log file **and** printing to
stderr. Never log to only one destination. Always use `warp.debug.log` — do
NOT introduce `logging.getLogger(__name__)`, as that bypasses the file
mirror.

All WARP CORE logs are prefixed with context (e.g. `WarpImporter:`,
`LayoutDetector:`, `AW.zoom`).

---

## Data sources (no SETS dependency)

Cargo and reference data are fetched from the community SETS-Data GitHub
mirror (`STOCD/SETS-Data`) on first run and cached locally:

| File | Purpose |
|---|---|
| `equipment.json` | weapon / shield / console / device DB |
| `traits.json` | personal / starship / reputation traits |
| `boff_abilities.json` | BOFF ability metadata (rank Roman numerals) |
| `ships.json` | ship roster (used for type-first disambiguation) |

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
  `warp/warp_importer.py`, `warp/sets_export.py`, `warp/app.py`) — these
  will move to a separate `sets-warp-bridge` package that depends on
  `sto-warp` and `sets`.

When porting code from `sets-warp/warp/`, replace `from src.setsdebug import
log` with `from warp.debug import log`. Reject any new `from src.*` import.
