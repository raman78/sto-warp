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
    sub.add_parser('install-desktop',
                   help='Install or refresh the menu entry '
                        '(Linux .desktop / Windows Start Menu .lnk).')
    validate = sub.add_parser('validate-build',
                              help='Check a build JSON against the SETS format contract.')
    validate.add_argument('path', help='build JSON file to check')
    validate.add_argument('--no-cargo', action='store_true',
                          help='skip name lookups against the cargo cache (offline)')

    args = parser.parse_args(argv)

    if args.cmd == 'validate-build':
        import json
        from pathlib import Path
        from warp.sets_schema import contract, validate_sets_build

        build = json.loads(Path(args.path).read_text(encoding='utf-8'))
        cache = None
        if not args.no_cargo:
            from warp.data.cargo import cache_view
            cache = cache_view()
        violations = validate_sets_build(build, cache)
        if not violations:
            print(f'{args.path}: clean against SETS {contract()["sets_tag"]}')
            return 0
        print(f'{args.path}: {len(violations)} violations against '
              f'SETS {contract()["sets_tag"]}', file=sys.stderr)
        for violation in violations:
            print(f'  [{violation.severity}] {violation}', file=sys.stderr)
        return 1

    if args.cmd == 'check':
        log.info('sto-warp check: importing recognition modules...')
        from warp.recognition import boff_keys, boff_marker, eq_geometry  # noqa: F401
        log.info('sto-warp check: OK')
        print(f'sto-warp {__version__} — foundation modules import OK.')
        return 0

    if args.cmd == 'install-desktop':
        from warp.gui.desktop_install import install_desktop_entry
        path = install_desktop_entry(force=True)
        if path is None:
            print('install-desktop: no menu entry written '
                  '(unsupported platform, `sto-warp` not on PATH, or '
                  'shortcut creation blocked).')
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
