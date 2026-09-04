#!/usr/bin/env python3
# warp/tools/reconcile_uploads.py
#
# REPO OWNER TOOL — deep diagnosis of whether this install's contributions
# reached the community dataset. Ships in the package like its neighbours in
# this folder, and like them it is not part of the user-facing program: a user
# is told only that something is pending, by the counter in WARP CORE's status
# bar. Interpreting *why* — a lost upload against a lost vote — is maintainer
# work and needs the published dataset to answer.

"""Did this machine's contributions reach the community dataset.

The published dataset is what the trainers read and what every install
receives. This store is one contributor's opinion, so it holding a different
label is normally the tally working rather than a fault. The question worth
asking is therefore not "do the two agree" but **was my decision ever
submitted**, and that is kept strictly apart from the answer it got:

    unsent     confirmed here, and this install never sent that label.
               A transport fault — the dataset could not have weighed it.
    outvoted   sent under this label, and the tally settled on another.
               Not a fault. Listed so it can be reviewed, never scored.
    absent     in the dataset, not in this store. Usually a maintainer
               rejection, and equally not a fault.

The split comes from this install's own upload cache, which records the label
each item was last sent under. Without it the two are indistinguishable, and
reporting a lost upload and a lost vote alike amounts to arguing that the
consensus should be corrected to match one machine.

Everything it reads on the remote side is public, so no token is needed and
nothing is uploaded. `WARPSyncClient` and `SyncWorker` own the hashing and the
cache format; this calls them rather than reimplementing either, because a
second copy of "how a sha is truncated" is exactly the kind of drift that
turns an empty result into "everything agrees".

Usage:
    python -m warp.tools.reconcile_uploads
    python -m warp.tools.reconcile_uploads --domain screens
    python -m warp.tools.reconcile_uploads --store /path/to/training_data --json
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import urllib.request
from pathlib import Path

DATASET = 'sets-sto/sto-icon-dataset'
_BASE   = f'https://huggingface.co/datasets/{DATASET}/resolve/main'

# Not a label: the screen-type menu offers it so a user can undo a wrong pick,
# and the backend refuses it. Counting it would report a backlog nobody can
# clear.
UNCLASSIFIED = 'UNKNOWN'


def _fetch(path: str) -> str:
    """One public file from the dataset. Anonymous by design, as the client's
    other reads of this repo are."""
    req = urllib.request.Request(f'{_BASE}/{path}',
                                 headers={'User-Agent': 'sto-warp reconcile'})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read().decode('utf-8', errors='replace')


def _sha(path: Path) -> str:
    from warp.trainer.sync import SyncWorker
    return SyncWorker._file_sha256(path)


# ── This machine ───────────────────────────────────────────────────────────

def local_screens(store: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    root = store / 'screen_types'
    if not root.is_dir():
        return out
    for type_dir in sorted(root.iterdir()):
        if not type_dir.is_dir() or type_dir.name == UNCLASSIFIED:
            continue
        for png in type_dir.glob('*.png'):
            try:
                out[_sha(png)] = type_dir.name
            except Exception:
                pass
    return out


def local_crops(store: Path) -> dict[str, str]:
    """`{sha: 'slot|name'}` from `annotations.json`, not from the filenames.

    A crop's filename carries the label it had when the file was written, so
    reading it there would hide a later correction — which is exactly where a
    correction matters.
    """
    out: dict[str, str] = {}
    ann, crops = store / 'annotations.json', store / 'crops'
    if not ann.exists() or not crops.is_dir():
        return out
    try:
        data = json.loads(ann.read_text(encoding='utf-8'))
    except Exception:
        return out

    by_id: dict[str, str] = {}
    for rec in data.values():
        if not isinstance(rec, dict):
            continue
        for a in rec.get('annotations') or []:
            if isinstance(a, dict) and a.get('ann_id') and a.get('name'):
                by_id[str(a['ann_id'])] = f"{a.get('slot', '')}|{a['name']}"

    for png in crops.rglob('*.png'):
        label = by_id.get(png.stem.rsplit('__', 1)[-1])
        if label:
            try:
                out[_sha(png)] = label
            except Exception:
                pass
    return out


def sent_labels(store: Path, domain: str) -> dict[str, str]:
    """What this install last *sent* for each sha, from its own upload cache.

    A legacy screen cache was a bare list of shas and recorded no label, so
    nothing in it can support an `outvoted` claim; those entries map to an
    empty label, which never equals a real one.
    """
    if domain == 'screens':
        from warp.trainer.sync import SyncWorker
        return SyncWorker._load_screen_type_cache(
            store / '.sync_uploaded_screen_hashes.json')
    try:
        raw = json.loads(
            (store / '.sync_uploaded_labels.json').read_text(encoding='utf-8'))
    except Exception:
        return {}
    return {str(k): str(v) for k, v in raw.items()} if isinstance(raw, dict) else {}


# ── The dataset ────────────────────────────────────────────────────────────

def published_screens() -> dict[str, str]:
    out: dict[str, str] = {}
    for line in _fetch('data/screen_types/metadata.jsonl').splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get('sha') and r.get('type'):
            out[r['sha']] = r['type']
    return out


def published_crops() -> dict[str, str]:
    out: dict[str, str] = {}
    for line in _fetch('data/annotations.jsonl').splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        sha = (r.get('crop_sha256') or '')
        if sha and r.get('name'):
            # The dataset stores the full digest for crops; the client
            # truncates. Compare on the client's length, or nothing matches
            # and the empty result reads as "everything agrees".
            out[sha[:32]] = f"{r.get('slot', '')}|{r['name']}"
    return out


def compare(local: dict[str, str], remote: dict[str, str],
            sent: dict[str, str]) -> dict[str, list]:
    unsent: list[tuple[str, str]] = []
    outvoted: list[tuple[str, str, str]] = []

    for sha, mine in local.items():
        theirs = remote.get(sha)
        if theirs == mine:
            continue
        was_sent = sent.get(sha) == mine
        if theirs is None:
            (outvoted if was_sent else unsent).append(
                (sha, mine, '<dropped>') if was_sent else (sha, mine))
        elif was_sent:
            outvoted.append((sha, mine, theirs))
        else:
            unsent.append((sha, mine))

    absent = sorted(sha for sha in remote if sha not in local)
    return {'unsent': sorted(unsent), 'outvoted': sorted(outvoted),
            'absent': absent}


def _report(domain: str, local: dict, remote: dict, v: dict) -> None:
    print(f'\n=== {domain} ===')
    print(f'  here {len(local)}   published {len(remote)}   '
          f'(the published dataset is the reference)')
    print(f'  unsent    {len(v["unsent"]):5}  never submitted from here — a fault')
    print(f'  outvoted  {len(v["outvoted"]):5}  submitted, tally settled otherwise')
    print(f'  absent    {len(v["absent"]):5}  in the dataset, not in this store')

    if v['unsent']:
        print('\n  never submitted from here:')
        for lab, n in collections.Counter(l for _, l in v['unsent']).most_common(15):
            print(f'    {n:5}  {lab!r}')
    if v['outvoted']:
        print('\n  submitted, and the dataset settled on something else:')
        print('  (not a fault — review only if a pairing looks wrong)')
        pairs = collections.Counter((m, t) for _, m, t in v['outvoted'])
        for (m, t), n in pairs.most_common(15):
            print(f'    {n:5}  sent {m!r}\n           kept {t!r}')


def main() -> int:
    from warp import userdata

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--store', type=Path, default=None,
                    help='training store (default: this install\'s own)')
    ap.add_argument('--domain', choices=('screens', 'crops', 'both'), default='both')
    ap.add_argument('--json', action='store_true')
    args = ap.parse_args()

    store = args.store or userdata.training_data_dir()
    if not Path(store).is_dir():
        print(f'ERROR: no training store at {store}', file=sys.stderr)
        return 2

    result: dict[str, dict] = {}
    if args.domain in ('screens', 'both'):
        l, r = local_screens(store), published_screens()
        result['screens'] = {'local': len(l), 'published': len(r),
                             **compare(l, r, sent_labels(store, 'screens'))}
        if not args.json:
            _report('screen types', l, r, result['screens'])
    if args.domain in ('crops', 'both'):
        l, r = local_crops(store), published_crops()
        result['crops'] = {'local': len(l), 'published': len(r),
                           **compare(l, r, sent_labels(store, 'crops'))}
        if not args.json:
            _report('crops', l, r, result['crops'])

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    unsent = sum(len(d['unsent']) for d in result.values())
    print(f'\n{unsent} decision(s) made here never reached the dataset.'
          if unsent else '\nEverything decided here has been submitted.')
    # Non-zero only for a transport fault. Being outvoted is the tally doing
    # its job, and scoring it would teach whoever runs this that the exit
    # status means nothing.
    return 1 if unsent else 0


if __name__ == '__main__':
    sys.exit(main())
