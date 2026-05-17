# sto-warp

Star Trek Online screenshot recognition + ML training, distributed as a
standalone Python package.

> sto-warp is the successor to the WARP / WARP CORE modules that used to
> live inside [sets-warp](https://github.com/raman78/sets-warp). It runs
> on its own, with no SETS build planner dependency, so you can use it as
> a library, a CLI, or behind any build planner that wants screenshot
> input.

## Components

- **WARP** — recognition pipeline (OCR + layout detection + icon matching).
  Reads STO screenshots and emits structured slot/item results.
- **WARP CORE** — Qt trainer UI for reviewing recognition output and
  fine-tuning the EfficientNet / MobileNetV3 models that drive WARP.

## Install (recommended: pipx)

```bash
pipx install sto-warp
sto-warp check        # verify install
sto-warp              # launch WARP CORE GUI (once trainer is wired)
```

`pipx` keeps sto-warp and its heavy dependencies (PyTorch, EasyOCR, Qt)
in an isolated venv — uninstall cleanly with `pipx uninstall sto-warp`.

Plain `pip install sto-warp` inside your own venv works too.

Native packages are planned for Arch (AUR), Debian/Ubuntu (`.deb`),
Fedora (COPR), and Windows (MSI/EXE) — see `INSTALLATION.md`.

## Data and models

On first run sto-warp downloads:

- ML models (`icon_classifier.pt`, `screen_classifier.pt`) from the
  Hugging Face hub.
- Cargo / ship / trait JSON from the community
  [`STOCD/SETS-Data`](https://github.com/STOCD/SETS-Data) repository.

Everything is cached under `~/.config/warp/` (or `$XDG_CONFIG_HOME/warp/`
when set). Nothing is committed to this repository.

## Development

```bash
git clone https://github.com/raman78/sto-warp.git
cd sto-warp
git checkout develop
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
sto-warp check
```

## License

GPL-3.0 — see `LICENSE`.

## Project docs

- `INSTALLATION.md` — install methods (pipx, native packages).
- `CHANGELOG.md` — release notes.
- `PROJECT_CONTEXT.md` — design context.
- `CLAUDE.md` — repository rules for AI-assisted development.
- `docs/` — technical deep-dives (BOFF / trait detection, ML pipeline, …).
