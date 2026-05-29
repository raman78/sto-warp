# Installation Guide

**STO-WARP** is distributed as a standalone Python package. Because it relies on heavy machine learning libraries (like PyTorch) and a GUI framework (PySide6), we highly recommend installing it in an isolated environment using **`pipx`**.

To make this as easy as possible, we provide universal installation scripts for both Linux/macOS and Windows.

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
1. Checks if Python is installed. If not, installs it via Windows Package Manager (`winget`).
2. Checks if `pipx` is installed. If not, installs it.
3. Installs `sto-warp` globally so you can launch it from anywhere.

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

## 🖥️ Desktop launcher icon (Linux)

On Linux, sto-warp adds a clickable menu entry for itself so it can
be launched from KDE Plasma, GNOME Activities, KRunner, XFCE and any
other XDG-compliant menu — no terminal needed once it's there.

**When it's created:** the first time the application is started
from the terminal (`sto-warp` or `sto-warp launcher`), the program
writes a `.desktop` entry to
`~/.local/share/applications/` and copies its icon to
`~/.local/share/icons/sto-warp.png`. From then on, the menu entry
**"sto-warp"** is available in your application launcher.

**Refreshing or re-installing the entry.** If the menu entry was
deleted, the icon vanished after a theme change, or sto-warp was
moved to a different Python environment and the old `Exec=` path
became stale, run:

```bash
sto-warp install-desktop
```

This rewrites the `.desktop` file with the current binary path and
re-copies the icon. It is also safe to run after a `pipx upgrade`.

**macOS and Windows.** The Linux `.desktop` mechanism does not apply
on macOS or Windows; on those systems sto-warp is launched from the
terminal or the Windows Start menu shortcut that `pipx` creates.

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
