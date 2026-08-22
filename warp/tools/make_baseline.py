"""Refresh `warp/data/baseline/` — the offline-fallback cargo snapshot.

Run by the maintainer before cutting a minor release so the wheel ships a
recent snapshot for users whose very first run has no network.

Sources and their order come from `warp.data.cargo`, and so does the download
itself: reusing `cargo._fetch` means the fallback chain and the "is this
actually valid JSON" check exist once, instead of drifting apart in two
copies.

The snapshot is refused if a file comes back materially smaller than the one
already committed. Upstream data has silently lost valid items before, and
baking that into the wheel would ship the regression to every offline user —
past the point where a later upstream fix could help them.

Usage:
    python -m warp.tools.make_baseline           # download into baseline/
    python -m warp.tools.make_baseline --check   # compare sizes, change nothing
    python -m warp.tools.make_baseline --allow-shrink   # accept a real shrink
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from warp.data.cargo import UPSTREAM_BASES, _fetch

FILES = (
    'equipment.json',
    'ship_list.json',
    'boff_abilities.json',
    'traits.json',
    'starship_traits.json',
    # Harvested overlay; `cargo._fetch` routes it to its own base. Shipping
    # it means a first run without network still knows the fleet and colony
    # ground weapons the cargo tables omit.
    'scraped_ground_weapons.json',
)

BASELINE_DIR = Path(__file__).resolve().parents[1] / 'data' / 'baseline'

# Shrink beyond this against the committed snapshot needs a conscious decision.
MAX_SHRINK = 0.05


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Refresh baseline cargo snapshot.')
    parser.add_argument('--check', action='store_true',
                        help='Compare local sizes with remote without overwriting.')
    parser.add_argument('--allow-shrink', action='store_true',
                        help='Write even if a file came back much smaller.')
    args = parser.parse_args(argv)

    print(f'sources: {" -> ".join(UPSTREAM_BASES)}')
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    rc = 0

    for name in FILES:
        target = BASELINE_DIR / name
        try:
            payload, _etag, base = _fetch(name)
        except Exception as exc:
            print(f'  FAIL  {name}: {exc}', file=sys.stderr)
            rc = 1
            continue
        if payload is None:                       # only possible with an ETag
            continue

        local_size = target.stat().st_size if target.exists() else 0
        origin = base.rsplit('/', 3)[1] if '/' in base else base

        if args.check:
            tag = 'OK' if local_size == len(payload) else 'STALE'
            print(f'  {tag:5s} {name:24s} local={local_size}  '
                  f'remote={len(payload)}  ({origin})')
            if tag == 'STALE':
                rc = 2
            continue

        if local_size:
            drop = (local_size - len(payload)) / local_size
            if drop > MAX_SHRINK and not args.allow_shrink:
                print(f'  SKIP  {name:24s} {local_size} -> {len(payload)} bytes '
                      f'({drop:.1%} smaller) — rerun with --allow-shrink if the '
                      f'loss is real', file=sys.stderr)
                rc = 3
                continue

        target.write_bytes(payload)
        print(f'  OK    {name:24s} {len(payload)} bytes  ({origin})')

    return rc


if __name__ == '__main__':
    sys.exit(main())
