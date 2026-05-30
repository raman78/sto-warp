# Installation Guide

**STO-WARP** is distributed as a standalone Python package. Because it relies on heavy machine learning libraries (like PyTorch) and a GUI framework (PySide6), we highly recommend installing it in an isolated environment using **`pipx`**.

To make this as easy as possible, we provide universal installation scripts for both Linux/macOS and Windows.

> **A note on PyTorch.** sto-warp's recognition pipeline is CPU-only by
> design. The Windows `.exe` installer ships a CPU-only build of PyTorch
> (~400 MB total). On Linux, the default `pipx install` pulls the
> standard PyPI `torch` wheel which bundles the CUDA runtime (~2 GB) —
> harmless but bigger than needed. To save disk space, install with the
> CPU-only index instead:
>
> ```bash
> pipx install sto-warp \
>   --pip-args="--index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple"
> ```
>
> Maintainers / contributors who want CUDA for embedder training: see
> [`docs/gpu_setup.md`](docs/gpu_setup.md).

---

## 🪟 Windows: one-click installer (recommended)

The simplest way to install on Windows — no Python required, no
terminal commands.

1. Download the latest **`sto-warp-<version>-setup.exe`** from the
   [Releases page](https://github.com/raman78/sto-warp/releases/latest).
2. Double-click to launch the installer.
3. Accept the license, pick a folder (default
   `%LOCALAPPDATA%\Programs\sto-warp\` — **no admin required**),
   choose whether to add a Desktop icon, and click **Install**.
4. When the wizard finishes, sto-warp is available from the Start
   Menu. The first launch downloads the recognition models from the
   community Hugging Face mirror (one-off, ~150 MB).

The installer ships a self-contained Python 3.14 runtime and CPU-only
PyTorch — nothing else needs to be installed on the machine. To
update, download the newer `.exe` and run it over the existing install;
your data in `%APPDATA%\warp\` is preserved.

**SmartScreen on first run.** Because the installer is not yet
code-signed, Windows may display *"Windows protected your PC"*. Click
**More info → Run anyway**. This appears once.

---

## 🚀 The Easy Way (One-Command Install)

### Linux & macOS
Open your terminal and run the following command:

```bash
curl -sSL https://raw.githubusercontent.com/raman78/sto-warp/main/install.sh | bash
```

**What this does:**
1. Verifies you have Python installed.
2. Checks for `pipx`. If missing, it installs `pipx` via your system's package manager (`apt`, `dnf`, `pacman`, or `brew`).
3. Installs `sto-warp` in an isolated environment.
4. Makes the `sto-warp` command globally available.

### Windows
Open **PowerShell** and run the following command:

```powershell
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/raman78/sto-warp/main/install.ps1" -OutFile "install.ps1"; .\install.ps1; Remove-Item "install.ps1"
```

**What this does:**
1. Checks if Python is installed. If not, installs **Python 3.14** via Windows Package Manager (`winget`).
2. Checks if `pipx` is installed. If not, installs it.
3. Installs `sto-warp` globally so you can launch it from anywhere.

**Windows tips and common issues**

- **PowerShell execution policy.** If running `install.ps1` fails
  with *"running scripts is disabled on this system"*, run the
  install line from an elevated PowerShell as
  `PowerShell -ExecutionPolicy Bypass -File install.ps1` instead.
  The policy only blocks the local script — pipx itself is not
  affected.
- **`sto-warp` not recognised after install.** pipx places the
  shim in `%USERPROFILE%\.local\bin\sto-warp.exe`. The installer
  runs `pipx ensurepath`, but the new `PATH` only takes effect
  in **a freshly opened terminal** — close and reopen PowerShell
  or Windows Terminal.
- **Python version mismatch.** sto-warp requires Python 3.14+.
  If an older Python is already on the system, winget may install
  3.14 side-by-side. Verify with `py -3.14 --version`. If `pipx`
  picked the wrong interpreter, force the right one with
  `pipx install --python python3.14 sto-warp`.
- **Start Menu shortcut.** sto-warp adds itself to the Start Menu
  on first launch — see the *Desktop launcher icon* section below.
  No extra step is needed.
- **Windows Defender / SmartScreen on first launch.** Because
  `sto-warp.exe` is generated locally by pipx (not code-signed by
  Microsoft), SmartScreen may show *"Windows protected your PC"*.
  Click **More info → Run anyway**. This appears once.

---

## 🛠️ The Manual Way (Using pipx)

If you prefer to handle the installation yourself, ensure you have Python 3.14+ installed on your system.

1. **Install pipx**
   Follow the [official pipx installation instructions](https://pipx.pypa.io/stable/installation/) for your operating system.

2. **Install sto-warp**
   ```bash
   pipx install sto-warp
   ```

3. **Verify the installation**
   ```bash
   sto-warp check
   ```

## ✅ Verifying the installation

After install, two quick checks confirm everything is wired up:

```bash
sto-warp --version    # prints the installed version
sto-warp check        # imports the recognition pipeline and reports OK
```

If `sto-warp` is not found, the `~/.local/bin` folder (where `pipx`
places its shims) is probably missing from `PATH`. Run
`pipx ensurepath` once and reopen the terminal.

---

## 🖥️ Desktop launcher icon

sto-warp adds a clickable menu entry for itself so it can be
launched from a graphical menu instead of the terminal.

**Linux.** The first time the application is started from the
terminal (`sto-warp` or `sto-warp launcher`), the program writes a
`.desktop` entry to `~/.local/share/applications/` and copies its
icon to `~/.local/share/icons/sto-warp.png`. From then on, the entry
**"sto-warp"** appears in KDE Plasma, GNOME Activities, KRunner,
XFCE and any other XDG-compliant menu.

**Windows.** The first time `sto-warp` is started from a terminal,
a Start Menu shortcut is written to
`%APPDATA%\Microsoft\Windows\Start Menu\Programs\sto-warp-*.lnk`
and the bundled PNG icon is converted to a multi-size `.ico` cached
in `%APPDATA%\warp\icons\sto-warp.ico`. The shortcut shows up in
the Start menu and in Windows search, and can be pinned to the
taskbar from there. PowerShell is required for shortcut creation;
if it is blocked by a corporate execution policy the launcher logs
a warning and continues without the shortcut — a shortcut can then
be created manually (right-click `sto-warp.exe` in
`%USERPROFILE%\.local\bin\` → **Send to → Desktop (create shortcut)**,
then move the resulting `.lnk` into the Start Menu Programs folder
above).

**Refreshing or re-installing the entry.** If the menu entry was
deleted, the icon vanished after a theme change, or sto-warp was
moved to a different Python environment and the old target path
became stale, run:

```bash
sto-warp install-desktop
```

This rewrites the entry with the current binary path and refreshes
the icon. It is safe to re-run after every `pipx upgrade`.

**macOS.** No native menu entry is created on macOS yet. Launch
from the terminal with `sto-warp`.

---

## 📁 Where data lives

sto-warp stores everything outside the install location, so updates
and re-installs never lose user data:

| Folder | Contents |
|---|---|
| `~/.config/warp/` | install ID, logs, community-sync state |
| `~/.cache/warp/` | downloaded ML models and cargo / icon database |
| `~/.local/share/warp/training_data/` | screenshots, crops and `annotations.json` for the trainer |

Removing sto-warp via `pipx uninstall` leaves these folders intact.
Delete them by hand only if a true clean slate is desired.

---

## 📦 Migrating from sets-warp

Users coming from the older sets-warp checkout (where WARP lived
inside the build planner) can carry their install ID and caches over
in one command:

```bash
sto-warp migrate-from-sets-warp
```

This looks for a sets-warp checkout in the usual locations (or the
`$SETS_WARP_ROOT` environment variable) and copies whatever is found
into the sto-warp folders listed above. Existing sto-warp data is
preserved; pass `--overwrite-id` only when intentionally adopting the
sets-warp install ID.

---

## 🔄 Updating STO-WARP

Regardless of how you installed it, since `sto-warp` is managed by `pipx`, you can always update to the latest version by running:

```bash
pipx upgrade sto-warp
```

## 🗑️ Uninstalling

To completely remove `sto-warp` and all its dependencies from your system:

```bash
pipx uninstall sto-warp
```

*(Note: Downloaded data and machine learning models cached in `~/.config/warp/` are not removed automatically. You can delete that folder manually if you wish to clear all data).*
