"""First-run desktop / Launchpad / Start Menu integrator.

Modern pip / PEP 517 build backends don't expose a post-install hook,
so we follow the pattern used by SETS-WARP: on the first launcher run,
register the app with the host OS so it shows up alongside other
GUI applications.

  - Linux  → write an `~/.local/share/applications/sto-warp.desktop`
             entry + rounded-corner icon under `~/.local/share/icons/`.
  - macOS  → delegate to `macos_app_bundle.install_macos_app_bundle`
             which drops a `.app` into `~/Applications/`.
  - Windows → delegate to `windows_shortcut.install_windows_shortcut`
             which drops a `.lnk` into the Start Menu Programs folder.

Idempotent: re-runs are no-ops once the entry exists. The install is
keyed by an 8-char hash of `sys.executable` so two parallel installs
(e.g. pipx vs editable venv) don't overwrite each other.

No-op on platforms other than the three above.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import sys
from pathlib import Path

from warp.debug import syslog as log
from warp.gui.icon_round import rounded_icon
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


def _sweep_stale_entries(apps_dir: Path, our_exec: str, our_name: str) -> None:
    """Remove duplicate sto-warp-*.desktop entries pointing to our Exec.

    pipx upgrades recreate the venv, so `sys.executable` changes and the
    next launch's `_install_id()` differs from the old one. The old
    .desktop file still points at the same `~/.local/bin/sto-warp` shim,
    so without cleanup the user ends up with a fresh menu entry after
    every upgrade. This sweep removes any sibling entry that targets the
    same binary as the current install. Entries targeting a *different*
    Exec (a genuinely parallel install — e.g. editable dev venv) are
    left alone.
    """
    if not apps_dir.is_dir():
        return
    for f in apps_dir.glob('sto-warp-*.desktop'):
        if f.name == our_name:
            continue
        try:
            for line in f.read_text(encoding='utf-8', errors='replace').splitlines():
                if line.startswith('Exec='):
                    file_exec = line[len('Exec='):].rsplit(' %', 1)[0].strip()
                    if file_exec == our_exec:
                        f.unlink()
                        log.info(f'Desktop installer: removed stale entry {f.name}')
                    break
        except Exception as e:
            log.debug(f'Desktop installer: sweep skip {f.name}: {e}')


def install_desktop_entry(force: bool = False) -> Path | None:
    """Install (or refresh) the menu entry. Returns the path written, or None.

    On Linux, drops a `.desktop` file in the XDG applications folder.
    On Windows, delegates to `windows_shortcut.install_windows_shortcut`
    to drop a Start Menu `.lnk`.

    Silent no-op when:
      - we're on an unsupported platform (macOS, BSD, …)
      - the `sto-warp` binary cannot be located (e.g. running directly via
        `python -m warp.cli` without installing the project)
      - the entry already exists and `force=False`
    """
    if sys.platform == 'win32':
        from warp.gui.windows_shortcut import install_windows_shortcut
        return install_windows_shortcut(force=force)
    if sys.platform == 'darwin':
        from warp.gui.macos_app_bundle import install_macos_app_bundle
        return install_macos_app_bundle(force=force)
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

    # Clean up duplicates left behind by pipx upgrades (new venv → new
    # install_id → new file, but the Exec shim is the same). Runs on
    # every launch so the menu stays tidy even after multiple upgrades.
    _sweep_stale_entries(apps_dir, exec_path, desktop_name)

    if desktop_path.is_file() and not force:
        return desktop_path

    # Write a rounded-corner copy of the app icon next to other user icons
    # so the entry has a stable absolute path regardless of where the
    # wheel lives. The squircle mask matches the macOS / Windows variants.
    icon_path: Path | str = ''
    try:
        src = resource_path('SETS_icon_small.png')
        if src.is_file():
            icons_dir.mkdir(parents=True, exist_ok=True)
            dst = icons_dir / 'sto-warp.png'
            rounded_icon(src, size=512).save(dst, format='PNG')
            icon_path = dst
    except Exception as e:
        log.debug(f'Desktop installer: icon prep failed: {e}')

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
