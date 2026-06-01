"""First-run macOS .app bundle installer (darwin only).

pipx installs the `sto-warp` shim into `~/.local/bin`, but macOS does
not surface CLI binaries in Launchpad or Spotlight, so the user can
only launch the app from a terminal. This module mirrors the Linux
`desktop_install`: on the first launcher run (or on `sto-warp
install-desktop`), drop a minimal `.app` bundle into `~/Applications/`
so the app shows up in Launchpad, Spotlight, and the Dock.

The bundle contains only the three pieces macOS needs to recognise it
as an application:

    Contents/Info.plist                metadata (bundle id, icon ref, …)
    Contents/MacOS/sto-warp            shell stub that execs the pipx shim
    Contents/Resources/sto-warp.icns   multi-resolution rounded icon

No code signing is involved — because the user generates the bundle
themselves by running the app, Gatekeeper does not apply the
quarantine flag, so the bundle launches without the "damaged app"
warning that ships-and-downloaded apps get without a Developer ID.

Idempotent: re-runs are no-ops unless `force=True`. The bundle
directory name is keyed by an 8-char hash of `sys.executable` so two
parallel installs (e.g. pipx + editable venv) do not overwrite each
other. The `CFBundleIdentifier` stays stable so Launch Services
caches keep working across upgrades.

No-op outside macOS.
"""
from __future__ import annotations

import hashlib
import os
import plistlib
import shutil
import sys
from pathlib import Path

from warp.debug import syslog as log
from warp.gui.icon_round import rounded_icon
from warp.resources import resource_path


_BUNDLE_ID = 'com.stocd.sto-warp'

_LAUNCHER_TEMPLATE = """\
#!/bin/sh
exec "{exec_path}" "$@"
"""


def _resolve_binary() -> str | None:
    """Return the absolute path to the installed `sto-warp` binary, if any."""
    found = shutil.which('sto-warp')
    if found:
        return found
    candidate = Path.home() / '.local' / 'bin' / 'sto-warp'
    if candidate.is_file():
        return str(candidate)
    return None


def _install_id() -> str:
    h = hashlib.sha256(sys.executable.encode('utf-8', errors='replace')).hexdigest()
    return h[:8]


def _build_icns(dst: Path) -> bool:
    """Write a multi-resolution rounded .icns file at `dst`.

    Cached by source mtime: rebuilds only when the bundled PNG is
    newer than the existing .icns. Returns True if the file is in
    place after the call.
    """
    try:
        src = resource_path('SETS_icon_small.png')
        if not src.is_file():
            return False
        if dst.is_file() and dst.stat().st_mtime >= src.stat().st_mtime:
            return True
        dst.parent.mkdir(parents=True, exist_ok=True)
        img = rounded_icon(src, size=1024)
        # Pillow's ICNS encoder packs the required Apple icon sizes
        # (16, 32, 128, 256, 512, 1024 with @2x) from the input image.
        img.save(dst, format='ICNS')
        return True
    except Exception as e:
        log.debug(f'macOS bundle: icns build failed: {e}')
        return False


def _write_info_plist(dst: Path) -> None:
    plist = {
        'CFBundleName': 'sto-warp',
        'CFBundleDisplayName': 'sto-warp',
        'CFBundleIdentifier': _BUNDLE_ID,
        'CFBundleVersion': '1',
        'CFBundleShortVersionString': '1',
        'CFBundlePackageType': 'APPL',
        'CFBundleExecutable': 'sto-warp',
        'CFBundleIconFile': 'sto-warp',
        'NSHighResolutionCapable': True,
        'LSMinimumSystemVersion': '11.0',
        'LSUIElement': False,
        'NSHumanReadableCopyright': 'STO screenshot recognition + ML trainer',
    }
    with dst.open('wb') as f:
        plistlib.dump(plist, f)


def _sweep_stale_bundles(apps_dir: Path, our_exec: str, our_name: str) -> None:
    """Remove duplicate sto-warp-*.app bundles pointing at our binary.

    pipx upgrades recreate the venv, so `sys.executable` changes and
    `_install_id()` differs from the previous launch. The old bundle
    still points at the same `~/.local/bin/sto-warp` shim, so without
    this sweep the user accumulates a fresh Launchpad tile after every
    upgrade. Bundles whose launcher targets a *different* binary (a
    genuinely parallel install — e.g. editable dev venv) are left
    alone.
    """
    if not apps_dir.is_dir():
        return
    for f in apps_dir.glob('sto-warp-*.app'):
        if f.name == our_name:
            continue
        stub = f / 'Contents' / 'MacOS' / 'sto-warp'
        try:
            if not stub.is_file():
                continue
            for line in stub.read_text(encoding='utf-8', errors='replace').splitlines():
                if line.startswith('exec '):
                    parts = line.split('"')
                    if len(parts) >= 2 and parts[1] == our_exec:
                        shutil.rmtree(f, ignore_errors=True)
                        log.info(f'macOS bundle: removed stale bundle {f.name}')
                    break
        except Exception as e:
            log.debug(f'macOS bundle: sweep skip {f.name}: {e}')


def install_macos_app_bundle(force: bool = False) -> Path | None:
    """Install (or refresh) the .app bundle. Returns the bundle path or None.

    Silent no-op when:
      - we're not on macOS
      - the `sto-warp` binary cannot be located on PATH
      - the bundle already exists and `force=False`
    """
    if sys.platform != 'darwin':
        return None

    exec_path = _resolve_binary()
    if not exec_path:
        log.debug('macOS bundle: sto-warp binary not on PATH — skipping')
        return None

    apps_dir = Path.home() / 'Applications'
    bundle_name = f'sto-warp-{_install_id()}.app'
    bundle_path = apps_dir / bundle_name

    _sweep_stale_bundles(apps_dir, exec_path, bundle_name)

    if bundle_path.is_dir() and not force:
        return bundle_path

    try:
        contents = bundle_path / 'Contents'
        macos_dir = contents / 'MacOS'
        resources_dir = contents / 'Resources'
        macos_dir.mkdir(parents=True, exist_ok=True)
        resources_dir.mkdir(parents=True, exist_ok=True)

        launcher = macos_dir / 'sto-warp'
        launcher.write_text(
            _LAUNCHER_TEMPLATE.format(exec_path=exec_path),
            encoding='utf-8',
        )
        launcher.chmod(0o755)

        _build_icns(resources_dir / 'sto-warp.icns')
        _write_info_plist(contents / 'Info.plist')

        # Touch the bundle so Launch Services rescans the icon/metadata —
        # otherwise an upgrade keeps showing the previous icon until the
        # user logs out.
        try:
            os.utime(bundle_path, None)
        except OSError:
            pass

        log.info(f'macOS bundle: wrote {bundle_path}')
        return bundle_path
    except Exception as e:
        log.warning(f'macOS bundle: failed to write {bundle_path}: {e}')
        return None
