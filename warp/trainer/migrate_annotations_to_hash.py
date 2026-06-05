"""
migrate_annotations_to_hash.py — one-shot migration CLI
========================================================
Converts pre-content-hash annotations.json entries (keyed by filename)
to the new sha16 content-hash schema. The user supplies a directory of
the original screenshots; the script hashes each one, looks up matching
filename-keyed legacy entries, and promotes them into the active
sha16-keyed bucket via TrainingDataManager.migrate_legacy_by_path.

Files that are not present in the supplied directory remain inert under
their filename keys — they are never silently dropped. Re-run after
locating more originals to migrate additional entries.

Usage:
    python -m warp.trainer.migrate_annotations_to_hash /path/to/screenshots
    python -m warp.trainer.migrate_annotations_to_hash --dry-run /path/to/screenshots

Default data directory is `warp.userdata.training_data_dir()`. Override
with --data-dir for testing against a copy.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from warp import userdata
from warp.trainer.training_data import TrainingDataManager


_IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}


def _iter_images(root: Path):
    for p in sorted(root.iterdir()):
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS:
            yield p


def main() -> int:
    ap = argparse.ArgumentParser(
        description='Migrate legacy filename-keyed annotations to sha16 content hashes.')
    ap.add_argument('source', type=Path,
                    help='directory of original screenshots referenced by legacy entries')
    ap.add_argument('--data-dir', type=Path, default=None,
                    help='override warp training_data_dir (testing)')
    ap.add_argument('--dry-run', action='store_true',
                    help='report what would be migrated without writing anything')
    args = ap.parse_args()

    if not args.source.is_dir():
        print(f'error: source is not a directory: {args.source}', file=sys.stderr)
        return 2

    data_dir = args.data_dir or userdata.training_data_dir()
    mgr = TrainingDataManager(data_dir)

    legacy_files = set(mgr._legacy_annotations.keys()) | set(mgr._legacy_screen_types.keys())
    if not legacy_files:
        print('Nothing to migrate — no legacy entries found.')
        return 0

    print(f'Legacy entries on disk:      {len(legacy_files)}')

    matched: list[Path] = []
    seen_names: set[str] = set()
    for p in _iter_images(args.source):
        if p.name in seen_names:
            continue
        seen_names.add(p.name)
        if p.name in legacy_files:
            matched.append(p)

    print(f'Originals supplied that match: {len(matched)}')
    if not matched:
        print('Nothing to do; supply more original screenshots and re-run.')
        return 0

    if args.dry_run:
        for p in matched:
            print(f'  would migrate: {p.name}')
        print(f'\n--dry-run: no changes written ({len(matched)} would migrate).')
        return 0

    total_anns = 0
    migrated_files = 0
    for p in matched:
        n = mgr.migrate_legacy_by_path(p)
        total_anns += n
        migrated_files += 1
        print(f'  {p.name}: {n} annotation row(s) promoted')

    print(f'\nMigrated {migrated_files} screenshot(s), {total_anns} annotation row(s).')
    remaining = (set(mgr._legacy_annotations.keys())
                 | set(mgr._legacy_screen_types.keys()))
    print(f'Legacy entries remaining:    {len(remaining)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
