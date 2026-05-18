"""GitHub-backed binary asset sync for sto-warp.

Port of `sets-warp/src/syncmanager.py`, trimmed for the standalone WARP
use-case: only the two binary asset groups WARP cares about — item
icons (`images/`) and ship images (`ship_images/`) — are mirrored from
`STOCD/SETS-Data` via the GitHub Tree API.

What's kept from the original:
  - Tree API + SHA1 diff so reruns only download missing/changed files.
  - 1h tree cache (`github_tree_cache.json`) so cold start doesn't
    re-fetch the full manifest on every launch.
  - Bounded thread pool (5 workers), per-file 404→permanent /
    other-error single retry, 403-circuit-breaker.
  - 7-day failed-file TTL so dead URLs aren't retried every run.

What's dropped:
  - Wiki fallback for ship images. sto-warp recognition doesn't depend
    on wiki-only icons (boff/skill icons are unused).
  - `_TermProgress` + the `setsdebug` monkey-patch — sto-warp callers
    pass an `on_progress` callback and route to whatever UI they want.
  - Cargo group. `warp.data.cargo` already handles JSONs on demand
    with its own ETag flow; duplicating that here would race the
    cache. AssetSyncManager touches binary assets only.
"""
from __future__ import annotations

import hashlib
import json
import queue
import threading
import time
from pathlib import Path
from typing import Callable
from urllib.parse import quote

import requests

from warp.debug import log
from warp.data.cargo import _cache_dir, icons_dir, ship_images_dir

GITHUB_API_TREE = (
    'https://api.github.com/repos/STOCD/SETS-Data/git/trees/main?recursive=1'
)
GITHUB_RAW_BASE = 'https://raw.githubusercontent.com/STOCD/SETS-Data/main'

TREE_CACHE_FILENAME    = 'github_tree_cache.json'
FAILED_CACHE_FILENAME  = 'sync_failed.json'
TREE_CACHE_MAX_AGE_S   = 60 * 60         # 1 hour
FAILED_RETRY_TTL_S     = 7 * 24 * 60 * 60  # 7 days

MAX_RETRIES     = 1
RETRY_DELAY_S   = 3
STALL_TIMEOUT_S = 10
MAX_THREADS     = 5
MAX_FORBIDDEN   = 3

ASSET_GROUPS: tuple[tuple[str, str, str], ...] = (
    ('Item Icons',  'images/',      'icon'),
    ('Ship Images', 'ship_images/', 'ship'),
)


# ── SHA helpers ──────────────────────────────────────────────────────────

def _git_sha1(filepath: Path) -> str | None:
    try:
        data   = filepath.read_bytes()
        header = f'blob {len(data)}\0'.encode()
        return hashlib.sha1(header + data).hexdigest()
    except OSError:
        return None


# ── Tree cache ───────────────────────────────────────────────────────────

def _load_tree_cache(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    age = time.time() - path.stat().st_mtime
    if age > TREE_CACHE_MAX_AGE_S:
        log.info(f'AssetSync: tree cache {age/60:.0f}min old — refreshing')
        return None
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        log.info(f'AssetSync: tree cache hit ({age/60:.0f}min old, {len(data)} files)')
        return data
    except Exception:
        return None


def _save_tree_cache(path: Path, tree: list[dict]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(tree), encoding='utf-8')
    except Exception as e:
        log.warning(f'AssetSync: tree cache write failed: {e}')


def _fetch_github_tree(session: requests.Session) -> list[dict] | None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(GITHUB_API_TREE, timeout=(10, STALL_TIMEOUT_S))
            if resp.ok:
                blobs = [e for e in resp.json().get('tree', [])
                         if e.get('type') == 'blob']
                log.info(f'AssetSync: tree fetched — {len(blobs)} files')
                return blobs
            if resp.status_code == 404:
                log.warning('AssetSync: tree 404 — repo not found?')
                return None
            error = f'HTTP {resp.status_code}'
        except Exception as e:
            error = str(e)
        log.warning(f'AssetSync: tree attempt {attempt}/{MAX_RETRIES} — {error}')
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY_S)
    return None


# ── Failed-asset persistence ─────────────────────────────────────────────

def _load_failed(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding='utf-8'))
        return {k: int(v) for k, v in raw.items() if isinstance(v, (int, float))}
    except Exception:
        return {}


def _save_failed(path: Path, failed: dict[str, int]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(failed, indent=2), encoding='utf-8')
    except Exception as e:
        log.warning(f'AssetSync: failed-cache write failed: {e}')


def _prune_failed(failed: dict[str, int]) -> dict[str, int]:
    now = time.time()
    return {k: ts for k, ts in failed.items()
            if now - ts < FAILED_RETRY_TTL_S}


# ── Main class ───────────────────────────────────────────────────────────

class AssetSyncManager:
    """Detect-and-download GitHub-backed binary assets for sto-warp.

    Usage:
        mgr = AssetSyncManager()
        report = mgr.run(on_progress=lambda label, current, total: ...)

    `on_progress` is called for each group's start, per-file update, and
    final summary. Safe to pass None.
    """

    def __init__(
        self,
        images_dir_: Path | None = None,
        ship_images_dir_: Path | None = None,
        cache_dir_: Path | None = None,
    ):
        self._images_dir      = Path(images_dir_)      if images_dir_      else icons_dir()
        self._ship_images_dir = Path(ship_images_dir_) if ship_images_dir_ else ship_images_dir()
        self._cache_dir       = Path(cache_dir_)       if cache_dir_       else _cache_dir()
        self._tree_cache_path   = self._cache_dir / TREE_CACHE_FILENAME
        self._failed_cache_path = self._cache_dir / FAILED_CACHE_FILENAME

        self._github_blocked   = False
        self._github_403_count = 0
        self._cb_lock          = threading.Lock()

        # Persistent dead-URL tracking — `path → unix_ts_of_last_failure`.
        # Entries past FAILED_RETRY_TTL_S are dropped on load.
        self._failed = _prune_failed(_load_failed(self._failed_cache_path))

    # ── Public ─────────────────────────────────────────────────────────────

    def run(
        self,
        on_progress: Callable[[str, int, int], None] | None = None,
    ) -> dict:
        prog = on_progress or (lambda label, c, n: None)
        prog('Checking for updates…', 0, 0)

        session = requests.Session()
        session.headers.update({'User-Agent': 'sto-warp/asset-sync'})

        tree = _load_tree_cache(self._tree_cache_path)
        if tree is None:
            prog('Fetching update manifest…', 0, 0)
            tree = _fetch_github_tree(session)
            if tree is None:
                log.warning('AssetSync: cannot reach GitHub — sync skipped')
                prog('Update check failed (offline?)', 0, 0)
                return {'checked': 0, 'updated': 0, 'failed': 0}
            _save_tree_cache(self._tree_cache_path, tree)

        total_updated = 0
        total_failed  = 0

        for (label, prefix, type_tag) in ASSET_GROUPS:
            entries = [e for e in tree if e.get('path', '').startswith(prefix)]
            to_update = self._diff_group(entries, type_tag)
            count = len(to_update)

            log.info(f'AssetSync [{label}]: {count}/{len(entries)} need download')

            if count == 0:
                prog(f'{label}: up to date', 0, 0)
                continue

            updated, failed_n = self._download_group(label, type_tag, to_update,
                                                     session, prog)
            total_updated += updated
            total_failed  += failed_n

        # Persist failed-asset map (entries added during run, plus pruned).
        _save_failed(self._failed_cache_path, self._failed)

        report = {
            'checked': len(tree),
            'updated': total_updated,
            'failed':  total_failed,
        }
        log.info(f'AssetSync: complete — {report}')
        if total_updated == 0 and total_failed == 0:
            prog('All assets up to date', 0, 0)
        else:
            prog(f'Sync done: {total_updated} updated, {total_failed} failed',
                 0, 0)
        return report

    # ── Diff ───────────────────────────────────────────────────────────────

    def _diff_group(self, entries: list[dict],
                    type_tag: str) -> list[tuple[dict, Path]]:
        result: list[tuple[dict, Path]] = []
        now = time.time()
        for entry in entries:
            path = entry.get('path', '')
            failed_ts = self._failed.get(path)
            if failed_ts is not None and now - failed_ts < FAILED_RETRY_TTL_S:
                continue  # skip dead URLs within TTL
            lp = self._local_path(path, type_tag)
            if lp and self._needs_update(lp, entry):
                result.append((entry, lp))
        return result

    def _local_path(self, github_path: str,
                    type_tag: str) -> Path | None:
        filename = github_path.split('/', 1)[1] if '/' in github_path else ''
        if not filename:
            return None
        if type_tag == 'icon':
            return self._images_dir / filename
        if type_tag == 'ship':
            return self._ship_images_dir / filename
        return None

    def _needs_update(self, local_path: Path, entry: dict) -> bool:
        if not local_path.exists():
            return True
        remote_size = entry.get('size', -1)
        if remote_size >= 0 and local_path.stat().st_size != remote_size:
            return True
        return _git_sha1(local_path) != entry.get('sha')

    # ── Download ───────────────────────────────────────────────────────────

    def _download_group(
        self,
        label: str,
        type_tag: str,
        to_update: list[tuple[dict, Path]],
        session: requests.Session,
        prog: Callable[[str, int, int], None],
    ) -> tuple[int, int]:
        count    = len(to_update)
        prog(label, 0, count)

        job_q: queue.Queue = queue.Queue()
        for item in to_update:
            job_q.put(item)

        counter  = [0]
        n_failed = [0]
        lock     = threading.Lock()

        def _worker(thread_num: int):
            while True:
                try:
                    entry, local_path = job_q.get_nowait()
                except queue.Empty:
                    return

                fname = entry['path'].split('/', 1)[1]
                ok, attempts = self._download_one(entry, local_path, session)

                with lock:
                    counter[0] += 1
                    if not ok:
                        n_failed[0] += 1
                        self._failed[entry['path']] = int(time.time())
                    c = counter[0]

                attempt_word = f'{attempts} attempt' + ('s' if attempts > 1 else '')
                if ok:
                    log.info(f'AssetSync [{label}] T-{thread_num}: '
                             f'OK ({attempt_word}) — {fname}')
                else:
                    log.warning(f'AssetSync [{label}] T-{thread_num}: '
                                f'FAILED ({attempt_word}) — {fname}')
                prog(label, c, count)
                job_q.task_done()

        n_threads = min(MAX_THREADS, count)
        threads = [threading.Thread(target=_worker, args=(i + 1,),
                                    name=f'asset-T{i+1}', daemon=True)
                   for i in range(n_threads)]
        for t in threads: t.start()
        for t in threads: t.join()

        updated = counter[0] - n_failed[0]
        summary = f'{updated} updated'
        if n_failed[0]:
            summary += f', {n_failed[0]} FAILED'
        log.info(f'AssetSync [{label}]: done — {summary}')
        prog(f'{label}: {summary}', count, count)
        return updated, n_failed[0]

    def _download_one(
        self,
        entry: dict,
        local_path: Path,
        session: requests.Session,
    ) -> tuple[bool, int]:
        if self._github_blocked:
            return False, 0

        url = f'{GITHUB_RAW_BASE}/{quote(entry["path"])}'
        last_status = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = session.get(url,
                                   timeout=(10, STALL_TIMEOUT_S), stream=False)
                last_status = resp.status_code
                if resp.ok and len(resp.content) >= 10:
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    local_path.write_bytes(resp.content)
                    return True, attempt
                if resp.status_code == 404:
                    return False, attempt
                error = f'HTTP {resp.status_code}'
            except Exception as e:
                error = str(e)
            log.warning(f'AssetSync: attempt {attempt}/{MAX_RETRIES} — '
                        f'{url} → {error}')
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_S)

        if last_status == 403:
            with self._cb_lock:
                self._github_403_count += 1
                if (self._github_403_count >= MAX_FORBIDDEN
                        and not self._github_blocked):
                    self._github_blocked = True
                    log.error('AssetSync: GitHub access BLOCKED after '
                              'repeated 403 errors.')
        return False, MAX_RETRIES
