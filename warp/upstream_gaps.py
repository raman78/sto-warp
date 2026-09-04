"""
upstream_gaps.py — a running record of what SETS drops from our exports
=======================================================================

Some items WARP recognises correctly cannot survive an import into SETS.
The export already tells the user so, per build. This keeps the tally, so
"SETS loses some of my gear" can become "these items were recognised in N
real builds and SETS drops every one of them" — which is an argument
somebody upstream can act on.

Two reasons, and they lead to different places:

    missing-from-cargo   The wiki lists the item but no cargo table stores
                         it, so it reaches WARP only through the harvested
                         overlay. SETS reads the same tables and cannot
                         resolve it. The ask belongs to the wiki.

    sets-loader-skips    Cargo carries the row — type, rarity and all — and
                         SETS's own loader passes over it. The 179
                         Advanced/Elite hangars are the whole of this case
                         today. The ask belongs to SETS.

Recording them together without the reason would merge two requests to two
different projects into one unusable list, so the reason is part of the key.

The file lives beside the other user-local state and is plain JSON: one
entry per item name, with the reason, the slots it turned up in, how many
exports carried it and when it was last seen. Nothing here runs during
recognition; it is appended once per export and read only when asked for.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from warp import userdata
from warp.debug import log

_FILENAME = 'upstream_gaps.json'

# Reasons, in the words the report uses. `sets_schema` phrases the violation;
# this maps it back to which project the gap belongs to.
REASON_CARGO  = 'missing-from-cargo'
REASON_LOADER = 'sets-loader-skips'


def ledger_path() -> Path:
    return userdata.data_dir() / _FILENAME


def _classify(got: str) -> str:
    """Which upstream the gap belongs to, from the violation's own text."""
    return REASON_LOADER if 'loader skips' in got else REASON_CARGO


def _slot(path: str) -> str:
    """The slot a violation points at, read off its own path.

    Violation paths read `/space/fore_weapons[0]`, so the environment and
    the slot are already there; nothing needs to pass them separately.
    """
    parts = (path or '').strip('/').split('/')
    if len(parts) < 2:
        return ''
    return f'{parts[0]}/{parts[1].split("[")[0]}'


def record(violations) -> int:
    """Add this export's `not_in_sets` violations to the ledger.

    Returns the number of entries touched. Never raises: a build must export
    whether or not the bookkeeping works, so a failure here is logged and
    swallowed.
    """
    rows = [v for v in violations if getattr(v, 'rule', '') == 'not_in_sets']
    if not rows:
        return 0
    try:
        path = ledger_path()
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            data = {}
        if not isinstance(data, dict):
            data = {}

        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        for v in rows:
            # `got` reads like "'Item Name' (reason)" — the name is what the
            # upstream request is about.
            name = (v.got or '').split(' (')[0].strip().strip("'\"")
            if not name:
                continue
            reason = _classify(v.got or '')
            key = f'{reason}\t{name}'
            entry = data.get(key) or {'name': name, 'reason': reason,
                                      'slots': [], 'exports': 0,
                                      'first_seen': today}
            entry['exports'] = int(entry.get('exports') or 0) + 1
            entry['last_seen'] = today
            slot = _slot(v.path)
            # An item can appear in more than one slot across builds; keep
            # every slot it was seen in, since that is part of the request.
            if slot and slot not in entry.get('slots', []):
                entry.setdefault('slots', []).append(slot)
            data[key] = entry

        path.write_text(json.dumps(data, indent=2, ensure_ascii=False,
                                   sort_keys=True), encoding='utf-8')
        return len(rows)
    except Exception as e:
        log.warning(f'upstream_gaps: could not record ({e})')
        return 0


def snapshot() -> list[dict]:
    """This install's current ledger, in the shape the backend accepts.

    The whole ledger goes up at once and replaces whatever this install sent
    before, so the upload is idempotent and an item that stops appearing
    expires by itself. Export counts stay local: what the backend needs is
    how many *installs* hit an item, and sending our own count would let one
    user's repeated exports read as demand from many.
    """
    try:
        data = json.loads(ledger_path().read_text(encoding='utf-8'))
    except Exception:
        return []
    if not isinstance(data, dict):
        return []
    return [
        {'name': e['name'], 'reason': e['reason'], 'slots': e.get('slots') or []}
        for e in data.values()
        if isinstance(e, dict) and e.get('name') and e.get('reason')
    ]


def summary() -> str:
    """A human-readable tally, grouped by which project the gap belongs to."""
    try:
        data = json.loads(ledger_path().read_text(encoding='utf-8'))
    except Exception:
        return 'No items have been recorded yet.'
    if not isinstance(data, dict) or not data:
        return 'No items have been recorded yet.'

    groups: dict[str, list[dict]] = {}
    for entry in data.values():
        groups.setdefault(entry.get('reason', '?'), []).append(entry)

    where = {
        REASON_CARGO:  'the wiki has the item, no cargo table stores it '
                       '— ask the wiki',
        REASON_LOADER: "cargo stores the item, SETS's loader passes over it "
                       '— ask SETS',
    }
    out: list[str] = []
    for reason, entries in sorted(groups.items()):
        entries.sort(key=lambda e: -int(e.get('exports') or 0))
        total = sum(int(e.get('exports') or 0) for e in entries)
        out.append(f'{reason} — {len(entries)} item(s), {total} export(s)')
        out.append(f'  ({where.get(reason, "")})')
        for e in entries:
            slots = ', '.join(e.get('slots') or [])
            out.append(f'    {e.get("exports", 0):4d}×  {e.get("name", "")}'
                       + (f'  [{slots}]' if slots else ''))
        out.append('')
    return '\n'.join(out).rstrip()
