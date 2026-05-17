"""Refresh `warp/data/baseline/` from STOCD/SETS-Data.

Run by the maintainer before cutting a minor release so the wheel ships
a recent offline-fallback snapshot of the cargo data files.

Usage:
    python -m warp.tools.make_baseline           # download into baseline/
    python -m warp.tools.make_baseline --check   # only verify sizes match remote
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

UPSTREAM = 'https://raw.githubusercontent.com/STOCD/SETS-Data/main/cargo'

FILES = (
    'equipment.json',
    'ship_list.json',
    'boff_abilities.json',
    'traits.json',
    'starship_traits.json',
)

BASELINE_DIR = Path(__file__).resolve().parents[1] / 'data' / 'baseline'


def _fetch(name: str) -> bytes:
    url = f'{UPSTREAM}/{name}'
    with urllib.request.urlopen(url, timeout=30) as resp:
        if resp.status != 200:
            raise RuntimeError(f'{url}: HTTP {resp.status}')
        return resp.read()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Refresh baseline cargo snapshot.')
    parser.add_argument('--check', action='store_true',
                        help='Compare local sizes with remote without overwriting.')
    args = parser.parse_args(argv)

    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    rc = 0

    for name in FILES:
        target = BASELINE_DIR / name
        try:
            payload = _fetch(name)
        except Exception as exc:
            print(f'  FAIL  {name}: {exc}', file=sys.stderr)
            rc = 1
            continue

        if args.check:
            local_size = target.stat().st_size if target.exists() else 0
            tag = 'OK' if local_size == len(payload) else 'STALE'
            print(f'  {tag:5s} {name:24s} local={local_size}  remote={len(payload)}')
            if tag == 'STALE':
                rc = 2
        else:
            target.write_bytes(payload)
            print(f'  OK    {name:24s} {len(payload)} bytes')

    return rc


if __name__ == '__main__':
    sys.exit(main())
