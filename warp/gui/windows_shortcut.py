"""First-run Start Menu shortcut installer (Windows only).

pipx installs the `sto-warp.exe` shim into `%USERPROFILE%\\.local\\bin`
but does NOT create a Start Menu entry, so the user can only launch
the app from a terminal. This module mirrors the Linux .desktop
installer: on the first launcher run (or on `sto-warp install-desktop`),
drop a `.lnk` file into the per-user Start Menu Programs folder so
the app shows up in Start, taskbar pinning, and Windows search.

The shortcut is created via PowerShell + WScript.Shell COM — no
pywin32 / winshell dependency.

Idempotent: re-runs are no-ops unless `force=True`. The shortcut is
keyed by an 8-char hash of `sys.executable` so a pipx install and a
parallel editable venv install do not overwrite each other.

No-op outside Windows.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

from warp.debug import syslog as log
from warp.resources import resource_path


_PS_TEMPLATE = (
    "$WshShell = New-Object -ComObject WScript.Shell; "
    "$Shortcut = $WshShell.CreateShortcut('{lnk}'); "
    "$Shortcut.TargetPath = '{target}'; "
    "$Shortcut.WorkingDirectory = '{workdir}'; "
    "$Shortcut.IconLocation = '{icon}'; "
    "$Shortcut.Description = 'STO screenshot recognition + ML trainer'; "
    "$Shortcut.Save()"
)


def _resolve_binary() -> str | None:
    """Return the absolute path to the installed `sto-warp.exe`, if any."""
    found = shutil.which('sto-warp')
    if found:
        return found
    # pipx on Windows defaults to %USERPROFILE%\.local\bin
    candidate = Path.home() / '.local' / 'bin' / 'sto-warp.exe'
    if candidate.is_file():
        return str(candidate)
    return None


def _install_id() -> str:
    h = hashlib.sha256(sys.executable.encode('utf-8', errors='replace')).hexdigest()
    return h[:8]


def _icon_cache_dir() -> Path:
    """Per-user folder where the converted .ico lives.

    %APPDATA%\\warp\\icons\\ — survives pipx upgrades and stays on the
    same filesystem as the Start Menu .lnk, so the icon path in the
    shortcut is stable across upgrades.
    """
    base = Path(os.environ.get('APPDATA') or (Path.home() / 'AppData' / 'Roaming'))
    return base / 'warp' / 'icons'


def _prepare_icon() -> str:
    """Convert the bundled PNG to an .ico in the user cache and return its path.

    Returns an empty string on failure (Windows shows a generic icon).
    Idempotent: if the .ico already exists and is newer than the
    source PNG, the existing copy is reused.
    """
    try:
        src = resource_path('SETS_icon_small.png')
        if not src.is_file():
            return ''
        dst_dir = _icon_cache_dir()
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / 'sto-warp.ico'
        if dst.is_file() and dst.stat().st_mtime >= src.stat().st_mtime:
            return str(dst)
        from PIL import Image
        img = Image.open(src).convert('RGBA')
        # Standard Windows icon sizes — Explorer / taskbar pick the right one.
        sizes = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
        img.save(dst, format='ICO', sizes=sizes)
        return str(dst)
    except Exception as e:
        log.debug(f'Windows shortcut: icon prep failed: {e}')
        return ''


def install_windows_shortcut(force: bool = False) -> Path | None:
    """Install (or refresh) the Start Menu .lnk for sto-warp.

    Silent no-op when:
      - we're not on Windows
      - the `sto-warp` binary cannot be located
      - the shortcut already exists and `force=False`
    """
    if sys.platform != 'win32':
        return None

    target = _resolve_binary()
    if not target:
        log.debug('Windows shortcut: sto-warp.exe not on PATH — skipping')
        return None

    appdata = Path(os.environ.get('APPDATA') or (Path.home() / 'AppData' / 'Roaming'))
    start_menu = appdata / 'Microsoft' / 'Windows' / 'Start Menu' / 'Programs'
    lnk_name = f'sto-warp-{_install_id()}.lnk'
    lnk_path = start_menu / lnk_name

    if lnk_path.is_file() and not force:
        return lnk_path

    icon_path = _prepare_icon()
    workdir = str(Path(target).parent)

    try:
        start_menu.mkdir(parents=True, exist_ok=True)
        ps = _PS_TEMPLATE.format(
            lnk=str(lnk_path).replace("'", "''"),
            target=target.replace("'", "''"),
            workdir=workdir.replace("'", "''"),
            icon=icon_path.replace("'", "''"),
        )
        result = subprocess.run(
            ['powershell', '-NoProfile', '-NonInteractive',
             '-ExecutionPolicy', 'Bypass', '-Command', ps],
            check=False, capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0:
            log.warning(
                f'Windows shortcut: PowerShell failed (rc={result.returncode}): '
                f'{result.stderr.strip()[:300]}'
            )
            return None
        if not lnk_path.is_file():
            log.warning(
                f'Windows shortcut: PowerShell reported success but '
                f'{lnk_path} is missing'
            )
            return None
        log.info(f'Windows shortcut: wrote {lnk_path}')
        return lnk_path
    except FileNotFoundError:
        log.warning('Windows shortcut: powershell.exe not found — skipping')
        return None
    except Exception as e:
        log.warning(f'Windows shortcut: failed to write {lnk_path}: {e}')
        return None
