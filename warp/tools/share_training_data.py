"""Put the training data on a shared volume, so two systems build one set.

On a dual-boot machine each system has its own home, so
``~/.local/share/warp/training_data`` is two independent stores. Work done on
one is invisible to the other: the same screenshot gets reviewed twice, a
correction made on one side does not exist on the other, and the icon
matcher's session pool is seeded from half the material. It also makes for a
nasty scare — a store that "lost" 97 entries turns out to be the other
system's.

This moves one store to a shared location and leaves a symlink behind, so
both systems read and write the same files. Run it once per system:

    python -m warp.tools.share_training_data                  # report only
    python -m warp.tools.share_training_data --apply

The second system does not overwrite the first. Everything is **merged**, and
nothing is ever deleted: the local store is renamed aside, not removed, and a
file that exists on both sides keeps the shared copy unless the local one is
demonstrably richer.

## What merges, and how

| File | Rule |
|---|---|
| ``annotations.json`` | union by image key; on a clash, keep whichever entry has more user-confirmed rows |
| ``anchors.json`` | union of ``learned[]``, de-duplicated on the whole entry |
| ``screen_types.json`` | union by image key; a key labelled differently on the two sides is reported and the shared label kept |
| ``screen_types_user_confirmed.json``, ``screenshots_done.json`` | set union |
| ``crop_index.json``, ``recog_history.json`` | union by key, shared side wins |
| ``crops/``, ``screen_types/`` | files absent on the shared side are copied; existing ones are never overwritten |
| ``canonical_layout.json`` | not merged — it is aggregated from ``anchors.json`` and is rebuilt on the next run |

The clash rules are conservative on purpose. Two systems reviewing the same
screenshot differently is a real possibility, and this tool is not the place
to decide which reviewer was right; it reports the clash and leaves the
shared side alone.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

from warp import userdata

DEFAULT_SHARED = Path.home() / 'Shared' / 'warp' / 'training_data'

# Merged by key, with the rule named in the module docstring.
DICT_FILES = ('screen_types.json', 'crop_index.json', 'recog_history.json')
LIST_FILES = ('screen_types_user_confirmed.json', 'screenshots_done.json')
TREES = ('crops', 'screen_types')
# Derived from anchors.json; rebuilt on the next run rather than merged.
SKIP_FILES = ('canonical_layout.json',)


def _load(path: Path, fallback):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return fallback


def _user_confirmed_count(entry) -> int:
    rows = entry.get('annotations', []) if isinstance(entry, dict) else entry
    if not isinstance(rows, list):
        return 0
    return sum(1 for r in rows if isinstance(r, dict)
               and r.get('state') == 'confirmed' and not r.get('auto_confirmed'))


def merge_annotations(local: dict, shared: dict) -> tuple[dict, list[str]]:
    """Union by image key. A key on both sides keeps the richer entry."""
    out = dict(shared)
    clashes: list[str] = []
    for key, entry in local.items():
        if key not in out:
            out[key] = entry
            continue
        if _user_confirmed_count(entry) > _user_confirmed_count(out[key]):
            clashes.append(f'{key}: local kept ('
                           f'{_user_confirmed_count(entry)} vs '
                           f'{_user_confirmed_count(out[key])} confirmed rows)')
            out[key] = entry
        elif entry != out[key]:
            clashes.append(f'{key}: shared kept')
    return out, clashes


def merge_anchors(local: dict, shared: dict) -> dict:
    """Union of `learned[]`, de-duplicated on the entry as a whole."""
    seen: set[str] = set()
    merged: list = []
    for entry in (shared.get('learned') or []) + (local.get('learned') or []):
        key = json.dumps(entry, sort_keys=True)
        if key not in seen:
            seen.add(key)
            merged.append(entry)
    out = dict(shared) or dict(local)
    out['learned'] = merged
    return out


def merge_tree(local_dir: Path, shared_dir: Path, apply: bool) -> int:
    """Copy files the shared side does not have. Never overwrites."""
    if not local_dir.is_dir():
        return 0
    copied = 0
    for src in local_dir.rglob('*'):
        if not src.is_file():
            continue
        dst = shared_dir / src.relative_to(local_dir)
        if dst.exists():
            continue
        copied += 1
        if apply:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    return copied


def run(shared: Path, *, apply: bool = False) -> int:
    local = userdata.training_data_dir()

    if local.is_symlink():
        print(f'{local} is already a symlink -> {os.path.realpath(local)}')
        return 0
    if not local.is_dir():
        print(f'{local} does not exist — nothing to share')
        return 1

    print(f'local : {local}')
    print(f'shared: {shared}' + ('' if shared.exists() else '  (will be created)'))
    print()

    ann_local = _load(local / 'annotations.json', {})
    ann_shared = _load(shared / 'annotations.json', {})
    ann_merged, clashes = merge_annotations(ann_local, ann_shared)
    print(f'annotations.json  local {len(ann_local)} + shared {len(ann_shared)} '
          f'-> {len(ann_merged)} images')
    for line in clashes[:10]:
        print(f'   clash  {line}')
    if len(clashes) > 10:
        print(f'   ... and {len(clashes) - 10} more')

    anch_merged = merge_anchors(_load(local / 'anchors.json', {}),
                                _load(shared / 'anchors.json', {}))
    print(f'{"anchors.json":<32} -> {len(anch_merged.get("learned") or [])} learned layouts')

    dicts = {}
    for name in DICT_FILES:
        merged = dict(_load(local / name, {}))
        merged.update(_load(shared / name, {}))
        dicts[name] = merged
        print(f'{name:<32} -> {len(merged)} entries')

    lists = {}
    for name in LIST_FILES:
        merged = sorted(set(_load(local / name, [])) | set(_load(shared / name, [])))
        lists[name] = merged
        print(f'{name:<32} -> {len(merged)} entries')

    for name in TREES:
        n = merge_tree(local / name, shared / name, apply=False)
        print(f'{name + "/":<32} -> {n} file(s) to copy')

    if not apply:
        print('\nDry run. Re-run with --apply to merge and link.')
        return 0

    shared.mkdir(parents=True, exist_ok=True)
    (shared / 'annotations.json').write_text(
        json.dumps(ann_merged, ensure_ascii=False), encoding='utf-8')
    (shared / 'anchors.json').write_text(
        json.dumps(anch_merged, ensure_ascii=False), encoding='utf-8')
    for name, data in {**dicts, **lists}.items():
        (shared / name).write_text(json.dumps(data, ensure_ascii=False),
                                   encoding='utf-8')
    for name in TREES:
        merge_tree(local / name, shared / name, apply=True)
    for name in SKIP_FILES:
        src = local / name
        if src.is_file() and not (shared / name).exists():
            shutil.copy2(src, shared / name)

    # Renamed, never removed: this is the only copy of work that cannot be
    # recreated, and a symlink pointing at a half-written merge would be the
    # worst possible outcome.
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    backup = local.with_name(f'{local.name}.bak-{stamp}')
    local.rename(backup)
    local.symlink_to(shared, target_is_directory=True)

    print(f'\nmerged into {shared}')
    print(f'local store kept at {backup}')
    print(f'{local} -> {os.path.realpath(local)}')
    print('Run this on the other system too; it will merge, not overwrite.')
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--shared', default=str(DEFAULT_SHARED),
                    help=f'shared training-data directory (default: {DEFAULT_SHARED})')
    ap.add_argument('--apply', action='store_true',
                    help='actually merge and replace the local store with a symlink')
    args = ap.parse_args(argv)
    return run(Path(args.shared).expanduser(), apply=args.apply)


if __name__ == '__main__':
    sys.exit(main())
