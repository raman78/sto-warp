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
