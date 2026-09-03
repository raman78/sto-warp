"""Print the tally of items SETS drops from WARP's exports.

Every export records the items the validator flagged `not_in_sets` — items
recognition got right and SETS refuses on import. This reads that ledger back
and groups it by which project the gap belongs to, so a request upstream can
be backed by "recognised in N real builds" rather than by one screenshot.

The two groups ask different people for different things, which is why the
ledger keeps them apart: `missing-from-cargo` is a wiki request (add a cargo
table row for an item the wiki already documents), `sets-loader-skips` is a
SETS request (its build loader passes over rows cargo does carry).

Usage:
    python -m warp.tools.upstream_gaps_report          # the tally
    python -m warp.tools.upstream_gaps_report --path   # where the file lives
"""
from __future__ import annotations

import argparse
import sys

from warp import upstream_gaps


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--path', action='store_true',
                    help='print the ledger file path and exit')
    args = ap.parse_args()

    if args.path:
        print(upstream_gaps.ledger_path())
        return 0

    print(upstream_gaps.summary())
    return 0


if __name__ == '__main__':
    sys.exit(main())
