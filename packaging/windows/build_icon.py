"""Generate sto-warp.ico from the shipped PNG resource.

Run from the repo root before invoking PyInstaller:

    python packaging/windows/build_icon.py

Writes ``packaging/windows/_build/sto-warp.ico`` with all standard
Windows icon sizes embedded. Pillow is the only dependency.
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image


SRC = Path(__file__).resolve().parent.parent.parent / 'warp' / 'resources' / 'SETS_icon_small.png'
DST = Path(__file__).resolve().parent / '_build' / 'sto-warp.ico'
SIZES = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]


def main() -> int:
    if not SRC.is_file():
        print(f'build_icon: source PNG missing: {SRC}', file=sys.stderr)
        return 1
    DST.parent.mkdir(parents=True, exist_ok=True)
    Image.open(SRC).convert('RGBA').save(DST, format='ICO', sizes=SIZES)
    print(f'build_icon: wrote {DST}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
