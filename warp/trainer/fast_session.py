"""Fast Correction Mode session staging.

WARP Fast Correction Mode treats each screenshot as a *new instance* so
that corrections made in the ephemeral tab cannot overwrite the user's
existing training data. To keep that isolation while still letting the
user return to the same batch later (instead of losing in-progress work
on the first crash / mistake / tab switch), each batch is staged to
disk under a deterministic content hash.

Layout::

    ~/.cache/warp/fast_correction/
        <12-hex-hash>/
            session.json                  # metadata + last_used_at
            fc_<hash>__<orig_name>.png    # snapshot copies of the inputs

The 12-char hash is `sha256(sorted basenames)[:12]` — re-entering Fast
Mode with the same file set returns to the same directory and resumes
the in-progress annotations. File copies (not hardlinks) give snapshot
semantics and work on Windows / cross-filesystem layouts.

Garbage collection is invoked by the launcher at startup and removes any
session whose `last_used_at` is older than 14 days.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from warp.debug import log
from warp.userdata import cache_dir


_PREFIX = 'fc_'
_SEP    = '__'
_HASH_LEN = 12
_META_NAME = 'session.json'


def staging_root() -> Path:
    p = cache_dir() / 'fast_correction'
    p.mkdir(parents=True, exist_ok=True)
    return p


def session_hash(paths) -> str:
    """Stable 12-hex sha256 over the sorted set of input basenames.

    Basename-only so the hash survives the user moving the screenshot
    folder around. Sorted so input order doesn't change the identity.
    """
    names = sorted({Path(p).name for p in paths})
    h = hashlib.sha256('\n'.join(names).encode('utf-8')).hexdigest()
    return h[:_HASH_LEN]


def staged_name(session: str, orig_name: str) -> str:
    return f'{_PREFIX}{session}{_SEP}{orig_name}'


def display_name(name: str) -> str:
    """Strip the `fc_<hash>__` prefix from a staged filename, if present."""
    if not name.startswith(_PREFIX):
        return name
    sep_at = name.find(_SEP, len(_PREFIX))
    if sep_at < 0:
        return name
    return name[sep_at + len(_SEP):]


@dataclass
class FastSession:
    hash: str
    dir: Path
    # staged Path → original Path
    paths_map: dict = field(default_factory=dict)

    @property
    def staged_paths(self) -> list:
        return list(self.paths_map.keys())

    def orig_for(self, staged) -> Path | None:
        return self.paths_map.get(Path(staged))

    def touch(self) -> None:
        meta = self.dir / _META_NAME
        try:
            data = {}
            if meta.is_file():
                data = json.loads(meta.read_text(encoding='utf-8'))
            data['last_used_at'] = time.time()
            data['hash'] = self.hash
            data['orig_paths'] = {p.name: str(orig)
                                  for p, orig in self.paths_map.items()}
            meta.write_text(json.dumps(data, indent=2), encoding='utf-8')
        except Exception as e:
            log.debug(f'FastSession: touch failed for {self.dir}: {e}')


def prepare(orig_paths) -> FastSession:
    """Stage `orig_paths` into a content-hashed session dir.

    Files already present (matching basename) are left alone so resuming
    a session doesn't re-copy unchanged screenshots. Returns a
    `FastSession` whose `paths_map` is staged → original.
    """
    origs = [Path(p) for p in orig_paths]
    h = session_hash(origs)
    sdir = staging_root() / h
    sdir.mkdir(parents=True, exist_ok=True)

    paths_map: dict = {}
    for orig in origs:
        target = sdir / staged_name(h, orig.name)
        if not target.exists():
            try:
                shutil.copy2(orig, target)
            except Exception as e:
                log.warning(
                    f'FastSession: copy {orig} → {target} failed: {e}')
                continue
        paths_map[target] = orig

    sess = FastSession(hash=h, dir=sdir, paths_map=paths_map)
    sess.touch()
    log.info(
        f'FastSession: prepared {h} with {len(paths_map)} file(s) at {sdir}')
    return sess


def gc_old_sessions(max_age_days: int = 14) -> int:
    """Remove staging dirs whose `last_used_at` is older than `max_age_days`.

    Returns the number of session directories removed. Missing or
    unreadable metadata files are treated as "stale" — better to GC a
    corrupt session than leak disk forever.
    """
    root = staging_root()
    cutoff = time.time() - max_age_days * 86400
    removed = 0
    for sub in root.iterdir():
        if not sub.is_dir():
            continue
        meta = sub / _META_NAME
        last = 0.0
        if meta.is_file():
            try:
                last = float(json.loads(meta.read_text(encoding='utf-8'))
                             .get('last_used_at', 0))
            except Exception:
                last = 0.0
        if last >= cutoff:
            continue
        try:
            shutil.rmtree(sub, ignore_errors=False)
            removed += 1
        except Exception as e:
            log.warning(f'FastSession: gc {sub} failed: {e}')
    if removed:
        log.info(f'FastSession: gc removed {removed} old session(s) '
                 f'(>{max_age_days}d)')
    return removed
