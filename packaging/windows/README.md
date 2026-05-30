# Windows installer

This directory contains the build recipe for the Windows `.exe`
installer that gets attached to every GitHub Release.

| File | Role |
|---|---|
| `build_icon.py` | Converts `warp/resources/SETS_icon_small.png` into `_build/sto-warp.ico` (all standard sizes). Needs Pillow. |
| `sto-warp.spec` | PyInstaller spec — produces a one-folder bundle at `dist/sto-warp/`. |
| `sto-warp.iss`  | Inno Setup script — wraps the PyInstaller bundle into `dist/installer/sto-warp-{version}-setup.exe`. |

## Local reproduction (Windows)

```powershell
# 1. Fresh venv with CPU-only torch
py -3.14 -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install .
pip install pyinstaller pillow

# 2. Build icon + bundle
python packaging\windows\build_icon.py
pyinstaller --noconfirm --clean packaging\windows\sto-warp.spec

# 3. Build installer (Inno Setup 6 must be on PATH as `iscc`)
iscc /DSTOWarpVersion=1.0.11 packaging\windows\sto-warp.iss
```

Output: `dist\installer\sto-warp-1.0.11-setup.exe`.

In CI this whole sequence runs from `.github/workflows/windows-installer.yml`.

## Why CPU-only torch

`warp/recognition/` performs zero CUDA calls (EasyOCR is pinned to
`gpu=False`; no `.to('cuda')` anywhere). The screen-type trainer also
runs on CPU. Only `warp/trainer/embedder_trainer.py` opportunistically
uses CUDA when available — a power-user / maintainer workload that
runs once during model bootstrap. Shipping CPU-only torch keeps the
installer under ~500 MB instead of ~2.5 GB.

Maintainers who want CUDA for embedder training: see
`docs/gpu_setup.md`.
