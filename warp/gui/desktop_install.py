"""First-run XDG desktop entry installer (Linux only).

Modern pip / PEP 517 build backends don't expose a post-install hook,
so we follow the pattern used by SETS-WARP: on the first launcher run,
drop an `~/.local/share/applications/sto-warp.desktop` entry and copy
the app icon to `~/.local/share/icons/` so it shows up in menus,
launchers (KRunner, GNOME Activities, Plasma kickoff, …) and KDE/GNOME
Activities.

Idempotent: re-runs are no-ops once the .desktop file exists. The
install is keyed by an 8-char hash of `sys.executable` so two parallel
installs (e.g. pipx vs editable venv) don't overwrite each other.

No-op on macOS / Windows / non-Linux platforms.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import sys
from pathlib import Path

from warp.debug import syslog as log
from warp.resources import resource_path

_DESKTOP_TEMPLATE = """\
[Desktop Entry]
Type=Application
Name=sto-warp
GenericName=STO screenshot recognition
Comment=Star Trek Online screenshot recognition + ML trainer
Exec={exec_path} %U
Icon={icon_path}
Terminal=false
Categories=Game;Utility;
StartupNotify=true
StartupWMClass=sto-warp
"""


def _resolve_binary() -> str | None:
    """Return the absolute path to the installed `sto-warp` binary, if any."""
    found = shutil.which('sto-warp')
    if found:
        return found

    # pipx installs into ~/.local/bin even when not on PATH; cover that.
    candidate = Path.home() / '.local' / 'bin' / 'sto-warp'
    if candidate.is_file():
        return str(candidate)

    return None


def _install_id() -> str:
    h = hashlib.sha256(sys.executable.encode('utf-8', errors='replace')).hexdigest()
    return h[:8]


def install_desktop_entry(force: bool = False) -> Path | None:
    """Install (or refresh) the .desktop file. Returns the path written, or None.

    Silent no-op when:
      - we're not on Linux
      - the `sto-warp` binary cannot be located (e.g. running directly via
        `python -m warp.cli` without installing the project)
      - the file already exists and `force=False`
    """
    if sys.platform != 'linux':
        return None

    exec_path = _resolve_binary()
    if not exec_path:
        log.debug('Desktop installer: sto-warp binary not on PATH — skipping')
        return None

    data_home = Path(os.environ.get('XDG_DATA_HOME') or (Path.home() / '.local' / 'share'))
    apps_dir = data_home / 'applications'
    icons_dir = data_home / 'icons'

    desktop_name = f'sto-warp-{_install_id()}.desktop'
    desktop_path = apps_dir / desktop_name

    if desktop_path.is_file() and not force:
        return desktop_path

    # Copy the app icon next to other user icons so the entry has a stable
    # absolute path regardless of where the wheel lives.
    icon_path: Path | str = ''
    try:
        src = resource_path('SETS_icon_small.png')
        if src.is_file():
            icons_dir.mkdir(parents=True, exist_ok=True)
            dst = icons_dir / 'sto-warp.png'
            shutil.copy2(src, dst)
            icon_path = dst
    except Exception as e:
        log.debug(f'Desktop installer: icon copy failed: {e}')

    try:
        apps_dir.mkdir(parents=True, exist_ok=True)
        desktop_path.write_text(
            _DESKTOP_TEMPLATE.format(exec_path=exec_path, icon_path=icon_path),
            encoding='utf-8',
        )
        log.info(f'Desktop installer: wrote {desktop_path}')
    except Exception as e:
        log.warning(f'Desktop installer: failed to write {desktop_path}: {e}')
        return None

    # Best-effort cache refresh — harmless if the tool is missing.
    try:
        import subprocess
        subprocess.run(
            ['update-desktop-database', str(apps_dir)],
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except Exception:
        pass

    return desktop_path
