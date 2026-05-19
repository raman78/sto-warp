"""sto-warp console entry point.

Exposed via `pyproject.toml` as the `sto-warp` console script. The real
WARP CORE Qt window will be wired here once the trainer modules are
ported. For now this only verifies the foundation imports cleanly so
`pipx install sto-warp` produces a working command.
"""
from __future__ import annotations

import argparse
import sys

from warp import __version__
from warp.debug import log


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog='sto-warp',
        description='Star Trek Online screenshot recognition (standalone WARP).',
    )
    parser.add_argument('--version', action='version', version=f'sto-warp {__version__}')
    sub = parser.add_subparsers(dest='cmd')
    sub.add_parser('check', help='Verify installation and import the recognition pipeline.')
    sub.add_parser('launcher', help='Launch the combined WARP + WARP CORE tabbed window (default).')
    sub.add_parser('gui', help='Launch the standalone WARP recognition window.')
    sub.add_parser('warp-core', help='Launch the WARP CORE trainer window.')
    sub.add_parser('install-desktop', help='Install or refresh the Linux .desktop entry.')

    p_mig = sub.add_parser(
        'migrate-from-sets-warp',
        help='Copy install_id / hub_token / caches from a sets-warp checkout.')
    p_mig.add_argument(
        '--overwrite-id', action='store_true',
        help='Replace existing sto-warp install_id (backed up as install_id.txt.bak).')
    p_mig.add_argument(
        '--path', default=None,
        help='Path to the sets-warp checkout (defaults to $SETS_WARP_ROOT or common locations).')

    args = parser.parse_args(argv)

    if args.cmd == 'check':
        log.info('sto-warp check: importing recognition modules...')
        from warp.recognition import boff_keys, boff_marker, eq_geometry  # noqa: F401
        log.info('sto-warp check: OK')
        print(f'sto-warp {__version__} — foundation modules import OK.')
        return 0

    if args.cmd == 'migrate-from-sets-warp':
        from pathlib import Path
        from warp.userdata import migrate_from_sets_warp
        root = Path(args.path).expanduser() if args.path else None
        moved = migrate_from_sets_warp(
            overwrite_install_id=args.overwrite_id, sets_warp_root=root)
        if not moved:
            print('migrate-from-sets-warp: nothing migrated (no sets-warp checkout found).')
            return 1
        for key, did in moved.items():
            print(f'  {key:<16} {"copied" if did else "skipped (already exists)"}')
        return 0

    if args.cmd == 'install-desktop':
        from warp.gui.desktop_install import install_desktop_entry
        path = install_desktop_entry(force=True)
        if path is None:
            print('install-desktop: no .desktop file written '
                  '(non-Linux, or `sto-warp` not on PATH).')
            return 1
        print(f'install-desktop: wrote {path}')
        return 0

    if args.cmd == 'warp-core':
        from PySide6.QtWidgets import QApplication
        from warp.trainer.trainer_window import WarpCoreWindow
        app = QApplication.instance() or QApplication(argv or sys.argv)
        win = WarpCoreWindow()
        win.show()
        return app.exec()

    if args.cmd == 'gui':
        from warp.gui.warp_window import main as gui_main
        return gui_main(argv)

    if args.cmd in (None, 'launcher'):
        from warp.gui.launcher import main as launcher_main
        return launcher_main(argv)

    parser.print_help()
    return 0


if __name__ == '__main__':
    sys.exit(main())
