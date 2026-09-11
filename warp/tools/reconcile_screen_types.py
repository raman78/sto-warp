"""Make the screen-type training folders agree with the labels.

A screenshot has exactly one screen type. `screen_types.json` holds that, one
entry per screenshot, and it is right. The folders beside it —
``screen_types/<TYPE>/<filename>`` — were only ever written to, never cleaned,
because ``TrainingDataManager.set_screen_type`` copied instead of moving. So
every type a screenshot was ever given left a file behind, and the first of
those is usually not the user's answer but the classifier's guess: the
auto-classification path writes a copy too, before anybody looks at it.

Measured on the maintainer's store 2026-09-11, before this existed: 534 files
against 287 labels, and 110 screenshots filed under two or three mutually
exclusive types at once — 20 as both ``BOFFS`` and ``SPACE_BOFFS``, 19 as both
``GROUND_MIXED`` and ``SPACE_MIXED``.

Three things went wrong because of it, in increasing order of cost:

1. The uploader walks these folders, so it sent both labels every cycle.
2. The upload cache is keyed on the content hash alone and cannot hold two
   types for one hash, so the copies overwrote each other's entry and were
   re-sent for ever. The trainer's "not yet shared" count sat at 129 for days
   while 233 screenshots went up every hour.
3. The backend counts one vote per (install, sha) and takes whichever copy its
   file walk reaches first. So this install's vote was an arbitrary pick
   between the user's correction and the guess they had corrected — the
   correction was being thrown away half the time.

``set_screen_type`` now removes the other copies as it writes, so this is a
one-off for what is already on disk. It is a dry run unless ``--apply`` is
given, and it only ever deletes a file whose content hash matches a screenshot
the labels place under a *different* type. Anything it cannot account for is
reported and left alone.

Usage::

    python -m warp.tools.reconcile_screen_types              # report only
    python -m warp.tools.reconcile_screen_types --apply      # delete the stale copies
    python -m warp.tools.reconcile_screen_types --training-dir /custom/path
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def _sha16(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:16]


def _labels(training_dir: Path) -> dict[str, str]:
    """`{sha16: screen type}` — the one label per screenshot, from the store.

    Legacy entries are keyed by filename rather than by hash. They are kept
    out: a filename is not an identity, and acting on one could delete the
    wrong screenshot.
    """
    p = training_dir / 'screen_types.json'
    try:
        raw = json.loads(p.read_text(encoding='utf-8'))
    except Exception as e:                            # noqa: BLE001
        print(f'cannot read {p}: {e}', file=sys.stderr)
        return {}
    return {k: v for k, v in raw.items()
            if isinstance(v, str) and len(k) == 16 and not k.endswith('.png')}


def reconcile(training_dir: Path, apply: bool) -> int:
    """Report — and with *apply*, remove — copies the labels contradict."""
    root = training_dir / 'screen_types'
    if not root.is_dir():
        print(f'no screen_types folder under {training_dir}')
        return 0

    labels = _labels(training_dir)
    print(f'{len(labels)} labelled screenshots in screen_types.json')

    on_disk: dict[str, list[Path]] = defaultdict(list)
    total = 0
    for type_dir in sorted(root.iterdir()):
        if not type_dir.is_dir():
            continue
        for png in sorted(type_dir.glob('*.png')):
            total += 1
            try:
                on_disk[_sha16(png)].append(png)
            except OSError as e:                      # noqa: PERF203
                print(f'  unreadable, skipped: {png} ({e})')
    print(f'{total} files in screen_types/')

    stale: list[Path] = []
    unlabelled: list[Path] = []
    for sha, paths in on_disk.items():
        want = labels.get(sha)
        if want is None:
            # No label to judge by. Left alone even when it sits in several
            # folders: deleting on a guess is how training data disappears.
            if len(paths) > 1:
                unlabelled.extend(paths)
            continue
        for p in paths:
            if p.parent.name != want:
                stale.append(p)

    by_pair = Counter(f'{p.parent.name} → {labels[_sha16(p)]}' for p in stale) \
        if stale else Counter()
    print(f'\n{len(stale)} file(s) contradict the label:')
    for pair, n in by_pair.most_common():
        print(f'   {n:4d}  filed as {pair}')
    if unlabelled:
        print(f'\n{len(unlabelled)} file(s) in more than one folder with no '
              f'label to judge by — left alone:')
        for p in sorted(unlabelled)[:10]:
            print(f'   {p.parent.name}/{p.name}')
        if len(unlabelled) > 10:
            print(f'   … and {len(unlabelled) - 10} more')

    if not apply:
        print('\ndry run — pass --apply to remove the contradicting copies')
        return len(stale)

    removed = 0
    for p in stale:
        try:
            p.unlink()
            removed += 1
        except OSError as e:                          # noqa: PERF203
            print(f'  could not remove {p}: {e}')
    print(f'\nremoved {removed} of {len(stale)}')

    # The upload cache remembers `{sha: type}` for what has gone up. Entries
    # naming a type this store no longer holds would keep the screenshot
    # looking "already shared" under the wrong label, so they are dropped and
    # the next sync re-sends each one once, under its real type.
    cache_file = training_dir / '.sync_uploaded_screen_hashes.json'
    try:
        cache = json.loads(cache_file.read_text(encoding='utf-8'))
    except Exception:
        cache = None
    if isinstance(cache, dict):
        drop = [s for s, t in cache.items()
                if s in labels and t != labels[s]]
        if drop:
            for s in drop:
                cache.pop(s, None)
            cache_file.write_text(json.dumps(cache, indent=0, sort_keys=True))
            print(f'cleared {len(drop)} upload-cache entr(ies) naming a type '
                  f'the labels disagree with — they go up again once, '
                  f'correctly')
    return len(stale)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--apply', action='store_true',
                    help='actually remove the contradicting copies')
    ap.add_argument('--training-dir', type=Path, default=None,
                    help='override the training store location')
    args = ap.parse_args(argv)

    if args.training_dir is not None:
        training_dir = args.training_dir
    else:
        from warp import userdata
        training_dir = userdata.training_data_dir()
    print(f'store: {training_dir}\n')
    reconcile(training_dir, args.apply)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
