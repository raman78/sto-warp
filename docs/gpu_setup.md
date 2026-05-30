# GPU acceleration for power users

**TL;DR — most users don't need this.** sto-warp's recognition
pipeline (the screenshot-importer used by 99% of users) is CPU-only by
design. The only workload that benefits from a CUDA GPU is
`warp.trainer.embedder_trainer`, used to retrain the BOFF-ability
embedder during local model bootstrap. Speedup: roughly 5–10× (e.g.
20 min on CPU → 2–4 min on a modern NVIDIA card). If you don't train
embedders, stop reading — the default install is what you want.

---

## Who this is for

- Maintainers preparing a new release of the embedder model
- Contributors iterating on `warp/trainer/embedder_trainer.py`
- Power users who want faster local bootstrap after collecting their
  own training crops

## What you need

- An NVIDIA GPU with compute capability ≥ 5.0 (Maxwell / 2014 or newer)
- NVIDIA driver ≥ 525 (CUDA 12 runtime requirement)
- ~2 GB extra disk space for the CUDA-enabled PyTorch wheels

AMD and Intel GPUs are **not** supported by this path — PyTorch's
ROCm builds only target Linux, and the recognition pipeline gets zero
benefit from GPU anyway. If you have AMD/Intel hardware, the
CPU-only install is the recommended setup.

## Swapping CPU torch for CUDA torch

### pipx install (Linux, or Windows pipx)

```bash
pipx inject sto-warp torch torchvision \
  --pip-args="--index-url https://download.pytorch.org/whl/cu121 --upgrade --force-reinstall"
```

`--force-reinstall` is required because `torch` is already present in
the venv (as the CPU build). The injection replaces it with the CUDA
build inside the same isolated environment, so all other sto-warp
dependencies stay untouched.

### Windows `.exe` install

The Windows installer ships its own Python runtime inside
`%LOCALAPPDATA%\Programs\sto-warp\`. To swap torch in that bundle:

```powershell
cd $env:LOCALAPPDATA\Programs\sto-warp
.\python.exe -m pip install --index-url https://download.pytorch.org/whl/cu121 `
  --upgrade --force-reinstall torch torchvision
```

This only works if you installed per-user (the default — install
folder is writable without admin). If you installed system-wide,
either re-install per-user or run the command from an elevated
PowerShell.

## Verifying

```bash
sto-warp check   # imports recognition stack
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"
```

`embedder_trainer` auto-detects CUDA via `torch.cuda.is_available()`
and logs the chosen device at startup, e.g.:

```
EmbedderTrainer: device=cuda, batches/epoch=...
```

If you see `device=cpu` despite a CUDA-enabled torch install, the
driver is likely too old — `nvidia-smi` will report the supported
CUDA runtime version in the top-right corner of its output.

## Reverting to CPU

```bash
pipx inject sto-warp torch torchvision \
  --pip-args="--index-url https://download.pytorch.org/whl/cpu --upgrade --force-reinstall"
```

Or, on a Windows `.exe` install, replace `cu121` with `cpu` in the
PowerShell command above.
