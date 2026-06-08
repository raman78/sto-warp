# STO-WARP

Star Trek Online screenshot recognition + ML training, distributed as a
standalone Python package.

## What is this?

**STO-WARP** combines two tools in one package:

**WARP** *(Weaponry & Armament Recognition Platform)* — reads your in-game screenshots and automatically fills in your SETS build. Detects equipment, traits, bridge officers, and ship information using computer vision and machine learning.

**WARP CORE** — trainer interface built into WARP. Review and correct recognition results, confirm annotations, and retrain the local ML models (Icon Classifier + Layout Regressor) on your own data to improve accuracy over time.

## Technology

- **WARP** — recognition pipeline (OCR + layout detection + icon matching).
  Reads STO screenshots and emits structured slot/item results.
- **WARP CORE** — Qt trainer UI for reviewing recognition output and
  fine-tuning the EfficientNet / MobileNetV3 models that drive WARP.

## Install

The recommended way to install `sto-warp` is via our universal installation scripts, which will automatically configure an isolated environment using `pipx`.

**Linux / macOS:**
```bash
curl -sSL https://raw.githubusercontent.com/raman78/sto-warp/main/install.sh | bash
```

**Windows (PowerShell):**
```powershell
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/raman78/sto-warp/main/install.ps1" -OutFile "install.ps1"; .\install.ps1; Remove-Item "install.ps1"
```

On first launch sto-warp registers itself in your system menu — Start
Menu on Windows, Launchpad / Spotlight on macOS, Activities / KRunner
on Linux — so subsequent runs can be started by clicking the icon
instead of opening a terminal.

For manual installation methods and more details, see `INSTALLATION.md`.

## Data and models

On first run sto-warp downloads:

- ML models (`icon_classifier.pt`, `screen_classifier.pt`) from the
  Hugging Face hub.
- Cargo / ship / trait JSON from the community
  [`STOCD/SETS-Data`](https://github.com/STOCD/SETS-Data) repository.
- Community knowledge (`knowledge.json`), admin icon equivalence map,
  and a seed tarball of confirmed crops.

A small **setup splash** blocks the launcher on the very first start
until all of the above finish, so the application opens with a
complete data set. On every later run the cache is refreshed
silently in the background. Everything is cached under
`~/.config/warp/` (or `$XDG_CONFIG_HOME/warp/` when set). Nothing is
committed to this repository. See `docs/WARP_GUIDE.md` for the
user-facing walkthrough and `docs/SYNC_ARCHITECTURE.md` for the
technical view.

## License

GPL-3.0 — see `LICENSE`.

## Project docs

- `INSTALLATION.md` — install methods (pipx, native packages).
- `CHANGELOG.md` — release notes.
- `PROJECT_CONTEXT.md` — design context.
- `CLAUDE.md` — repository rules for AI-assisted development.
- `docs/` — technical deep-dives (BOFF / trait detection, ML pipeline, …).
