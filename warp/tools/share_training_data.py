"""Move the training data to a chosen directory, merging what is already there.

The store lives under the user's data directory
(``~/.local/share/warp/training_data``), so a second WARP CORE installation —
another account, another machine, a restored backup — builds a second,
independent one. Two stores mean the same screenshot gets reviewed twice, a
correction made in one does not exist in the other, and the icon matcher's
session pool is seeded from half the material.

This relocates a store to a directory of the caller's choosing and leaves a
symlink behind, so the installation keeps reading and writing through its
usual path:

    python -m warp.tools.share_training_data                  # report only
    python -m warp.tools.share_training_data --apply
    python -m warp.tools.share_training_data --shared /path   # elsewhere

Pointing a second installation at a destination that already holds a store
**merges** the two. Nothing is deleted: the local store is renamed aside
rather than removed, and a file present on both sides is never silently
clobbered.

## What merges, and how

| File | Rule |
|---|---|
| ``annotations.json`` | union by image key, and by row inside a key both sides know: a row only one side has is added; a collision (same slot, overlapping box) goes to whichever row the user confirmed, and if both did, to the one confirmed **later** — the answer given after the correction. Every difference is reported |
| ``anchors.json`` | union of ``learned[]``, de-duplicated on the whole entry |
| ``screen_types.json`` | union by image key; a key labelled differently on the two sides is reported and the shared label kept |
| ``screen_types_user_confirmed.json``, ``screenshots_done.json`` | set union |
| ``crop_index.json``, ``recog_history.json`` | union by key, shared side wins |
| ``crops/``, ``screen_types/`` | files absent on the shared side are copied; existing ones are never overwritten |
| ``canonical_layout.json`` | not merged — it is aggregated from ``anchors.json`` and is rebuilt on the next run |

The clash rules are conservative on purpose. The same screenshot reviewed
twice, with different answers, is a real possibility, and this tool is not
the place to decide which review was right; it reports the difference and
records it.

Row level rather than entry level because choosing a whole entry throws away
whichever side loses. Measured on a pair of real stores: of the 70
screenshots present in both, 48 held the same number of user-confirmed rows
and different content, so a tiebreak decided them — and in one case that
meant keeping a misspelt ship class over the corrected one, because the
misspelt side happened to carry one row more.
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


IOU_SAME_BOX = 0.3


def _rows(entry) -> list:
    return entry.get('annotations', []) if isinstance(entry, dict) else entry


def _collides(row: dict, kept: dict) -> bool:
    """Do these two rows describe the same thing?

    The trainer's own rule, from `_merge_recognition`: a slot that can hold
    only one item collides on the slot alone, everything else on bbox
    overlap. Without it a row-level union would stack two boxes on one icon.
    """
    from warp.trainer.training_data import SINGLE_INSTANCE_SLOTS, _bbox_iou
    if row.get('slot') != kept.get('slot'):
        return False
    if row.get('slot') in SINGLE_INSTANCE_SLOTS:
        return True
    a, b = row.get('bbox'), kept.get('bbox')
    if not a or not b:
        return False
    return _bbox_iou(tuple(a), tuple(b)) >= IOU_SAME_BOX


def confirmed_at(base: Path, row: dict) -> float | None:
    """When this row was confirmed, as far as the store can say.

    Nothing in `annotations.json` carries a timestamp — not the entry, not
    the row. What does is the crop the row points at: writing it is what
    confirming does. Measured across a real pair of stores, 94% and 97% of
    user-confirmed rows resolve to a crop that still exists, which is enough
    to decide the handful of rows the two stores disagree about.

    `shutil.copy2` preserves mtimes, so merging a store does not reset this.
    """
    name = row.get('crop_name') or ''
    if not name:
        return None
    try:
        return (base / name).stat().st_mtime
    except OSError:
        return None


def merge_entry(local, shared, local_base: Path | None = None,
                shared_base: Path | None = None) -> tuple[list, list[str]]:
    """Union the two sides' rows for one screenshot.

    Picking a whole entry, which is what this used to do, throws away
    whichever side loses. Measured on a pair of real stores: 70 screenshots
    were present in both, 48 of them with the same number of user-confirmed
    rows and different content, so the winner was decided by a tiebreak that
    meant nothing. One of those kept a misspelt ship class over the corrected
    one purely because that side carried one row more.

    Row level keeps both sides' work. A collision — same slot, overlapping
    box — is resolved in favour of the row the user confirmed; if both are
    confirmed and they disagree, the later answer wins and the difference is
    reported, because this tool cannot know which review was right.
    """
    kept = list(_rows(shared))
    notes: list[str] = []
    for row in _rows(local):
        if not isinstance(row, dict):
            continue
        hit = next((k for k in kept if _collides(row, k)), None)
        if hit is None:
            kept.append(row)
            continue
        row_ok = row.get('state') == 'confirmed' and not row.get('auto_confirmed')
        hit_ok = hit.get('state') == 'confirmed' and not hit.get('auto_confirmed')
        if row_ok and not hit_ok:
            kept[kept.index(hit)] = row
        elif row_ok and hit_ok and (row.get('name') or '') != (hit.get('name') or ''):
            # Both sides' reviewer was the user, and they disagree. The later
            # answer is the one given after whatever correction prompted it,
            # so it wins — the user's own reading of this data. When either
            # side has no crop to date, nothing is known and the shared side
            # stays put; either way the difference is written down.
            t_row = confirmed_at(local_base, row) if local_base else None
            t_hit = confirmed_at(shared_base, hit) if shared_base else None
            if t_row is not None and t_hit is not None and t_row > t_hit:
                kept[kept.index(hit)] = row
                notes.append(f'{row.get("slot")}: kept {row.get("name")!r} '
                             f'(newer), dropped {hit.get("name")!r}')
            else:
                why = 'newer' if (t_row is not None and t_hit is not None) else 'undated'
                notes.append(f'{row.get("slot")}: kept {hit.get("name")!r} '
                             f'({why}), dropped {row.get("name")!r}')
    return kept, notes


def merge_annotations(local: dict, shared: dict, local_base: Path | None = None,
                      shared_base: Path | None = None) -> tuple[dict, list[str]]:
    """Union by image key, and by row within a key both sides know."""
    out = dict(shared)
    clashes: list[str] = []
    for key, entry in local.items():
        if key not in out:
            out[key] = entry
            continue
        merged_rows, notes = merge_entry(entry, out[key], local_base, shared_base)
        name = (out[key].get('filename') if isinstance(out[key], dict) else '') or key
        before = len(_rows(out[key]))
        if isinstance(out[key], dict):
            out[key] = {**out[key], 'annotations': merged_rows}
        else:
            out[key] = merged_rows
        added = len(merged_rows) - before
        for note in notes:
            clashes.append(f'{name}: {note}')
        if added and not notes:
            clashes.append(f'{name}: +{added} row(s) from the other store')
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
    ann_merged, clashes = merge_annotations(ann_local, ann_shared, local, shared)
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
    # The first ten clashes are printed; all of them are written down. A
    # disagreement between two reviews of the same screenshot is exactly the
    # thing a summary line loses, and it is not recoverable afterwards — the
    # dropped row is gone from the merged store.
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    if clashes:
        report = shared / f'merge-report-{stamp}.txt'
        report.write_text('\n'.join(clashes) + '\n', encoding='utf-8')
        print(f'\n{len(clashes)} clash(es) written to {report}')
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
    backup = local.with_name(f'{local.name}.bak-{stamp}')
    local.rename(backup)
    local.symlink_to(shared, target_is_directory=True)

    print(f'\nmerged into {shared}')
    print(f'local store kept at {backup}')
    print(f'{local} -> {os.path.realpath(local)}')
    print('Point another installation at the same directory to merge it in.')
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
