# warp/knowledge/sync_client.py
#
# Client for the WARP Knowledge Backend.
#
# Responsibilities:
#   1. UPLOAD: send crop images + confirmed labels to the backend
#      (backend holds the HF write token — never exposed to users)
#
#   2. DOWNLOAD: fetch the community knowledge base at startup
#      knowledge.json = {phash_hex: item_name, ...}
#      Used by icon_matcher as an override layer before template matching.
#
# All network calls are:
#   - Non-blocking (background thread)
#   - Silent on failure (log warning, never crash the app)
#   - Rate-limited locally (max MAX_CONTRIBUTIONS_PER_DAY per installation)
#
# Configuration (warp/knowledge/config.json or hardcoded defaults):
#   BACKEND_URL   — base URL of the backend service
#   INSTALL_ID    — random UUID generated on first run, stored locally
#                   used for deduplication on the backend (not for user identity)

from __future__ import annotations

import base64
import hashlib
import json
import threading
import time
import uuid
from collections import deque
from datetime import date
from pathlib import Path
from typing import Callable

import numpy as np

from warp import userdata, config
from warp.debug import syslog as log

# ── Constants ──────────────────────────────────────────────────────────────────
# Backend lives on HF Spaces (sets-sto/warp-backend). Migrated from Render
# in 2026-05 so the shared HF write token could be removed from clients —
# the token now lives only as a Space secret. Render kept as a fallback
# during Phase 2 of the migration; will be turned off in Phase 3.
DEFAULT_BACKEND_URL       = 'https://sets-sto-warp-backend.hf.space'
MAX_CONTRIBUTIONS_PER_DAY = 200    # per installation, per day
KNOWLEDGE_MAX_AGE_HOURS   = 24     # re-download knowledge base after this

# Icon-equivalence is a small admin-curated JSON. We bypass the backend
# and read it straight from the HF dataset repo where it ships next to
# `data/crops/`. Keeping it out of the backend means a curated update
# reaches every client without redeploying the Space.
ICON_EQUIVALENCE_HF_REPO   = 'sets-sto/sto-icon-dataset'
ICON_EQUIVALENCE_HF_FILE   = 'icon_equivalence.json'
ICON_EQUIVALENCE_MAX_AGE_HOURS = 24

def _resolve_warp_version() -> str:
    """Best-effort sto-warp version for the ``WARP/<ver>`` User-Agent and
    the ``warp_version`` upload field. Tries installed package metadata
    first (pipx / PyPI installs); falls back to ``warp/_version.py``
    written by the build-time vcs-versioning hook (dev checkouts);
    finally ``'unknown'`` so we never crash on import."""
    try:
        from importlib.metadata import version
        return version('sto-warp')
    except Exception:
        pass
    try:
        from warp._version import __version__
        return __version__
    except Exception:
        return 'unknown'


WARP_VERSION = _resolve_warp_version()


class _ContributeQueue:
    """Single-worker bounded queue with disk persistence + TTL.

    Replaces the old "spawn one thread per `contribute()`" model. All
    uploads funnel through a single background worker, so:
      - at most one retry-burst runs at a time (no thread pile-up while
        the backend is sleeping / cold-starting)
      - circuit-breaker trips once per outage cycle, not once per call
      - log noise drops from N×4 lines to ~2 per outage cycle
      - items survive app restarts via append-only JSONL log
      - items older than TTL_DAYS are dropped on load (stale relative to
        current training session, not worth sending)

    File format (one JSON object per line):
        {"kind":"item","id":...,"ts":...,"phash":...,"crop_png_b64":...,
         "item_name":...,"wrong_name":...,"confirmed":...}
        {"kind":"ack","id":...}

    On load, items appear in `pending = items - acked`. After load the
    file is compacted when it grew past COMPACT_THRESHOLD lines.
    """

    MAX_IN_MEMORY     = 50      # hard cap on pending items
    TTL_DAYS          = 7       # drop items older than this on load
    COMPACT_THRESHOLD = 100     # rewrite file if it has more than N lines

    def __init__(self, client: 'WARPSyncClient', path: Path):
        self._client = client
        self._path   = path
        self._lock   = threading.Lock()
        self._cv     = threading.Condition(self._lock)
        self._queue: deque[dict] = deque()
        self._sent_since_drain   = 0
        self._overflow_warned    = False
        self._stop               = False
        self._load_from_disk()
        threading.Thread(
            target=self._worker_loop,
            daemon=True, name='warp-contribute-worker',
        ).start()

    # ── Public API ─────────────────────────────────────────────────────────
    def enqueue(self, item: dict) -> bool:
        """Append item to queue + persist. Returns False if queue full."""
        with self._cv:
            if len(self._queue) >= self.MAX_IN_MEMORY:
                if not self._overflow_warned:
                    log.warning(
                        f'WARPSync: contribute queue full ({self.MAX_IN_MEMORY}) — '
                        f'dropping new items until backend recovers'
                    )
                    self._overflow_warned = True
                return False
            self._overflow_warned = False
            self._queue.append(item)
            self._append_to_disk({'kind': 'item', **item})
            self._cv.notify()
            return True

    def depth(self) -> int:
        with self._lock:
            return len(self._queue)

    # ── Persistence ────────────────────────────────────────────────────────
    def _append_to_disk(self, rec: dict) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open('a', encoding='utf-8') as f:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        except Exception as e:
            log.debug(f'WARPSync: queue append failed: {e}')

    def _load_from_disk(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = self._path.read_text(encoding='utf-8').splitlines()
        except Exception as e:
            log.warning(f'WARPSync: queue file unreadable ({e}); starting empty')
            return
        items: dict[str, dict] = {}
        acked: set[str]        = set()
        cutoff = time.time() - self.TTL_DAYS * 86400
        for line in raw:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            kind = rec.get('kind')
            if kind == 'ack':
                iid = rec.get('id')
                if iid:
                    acked.add(iid)
            elif kind == 'item':
                if rec.get('ts', 0) < cutoff:
                    continue
                iid = rec.get('id')
                if iid:
                    rec.pop('kind', None)
                    items[iid] = rec
        pending = [it for iid, it in items.items() if iid not in acked]
        pending.sort(key=lambda r: r.get('ts', 0))
        if len(pending) > self.MAX_IN_MEMORY:
            dropped = len(pending) - self.MAX_IN_MEMORY
            log.warning(
                f'WARPSync: queue file had {len(pending)} pending; '
                f'trimmed {dropped} oldest to fit cap')
            pending = pending[-self.MAX_IN_MEMORY:]
        self._queue.extend(pending)
        if len(raw) > self.COMPACT_THRESHOLD:
            self._compact_locked()
        if pending:
            log.info(f'WARPSync: restored {len(pending)} pending contribution(s) from disk')

    def _ack_to_disk(self, item_id: str) -> None:
        self._append_to_disk({'kind': 'ack', 'id': item_id})

    def _compact_locked(self) -> None:
        """Rewrite queue file with only currently-pending items.

        Caller must hold self._lock (or be confident no concurrent write
        is in flight — only called from __init__ and the worker thread).
        """
        try:
            tmp = self._path.with_suffix(self._path.suffix + '.tmp')
            with tmp.open('w', encoding='utf-8') as f:
                for rec in self._queue:
                    f.write(json.dumps({'kind': 'item', **rec},
                                       ensure_ascii=False) + '\n')
            tmp.replace(self._path)
            log.debug(f'WARPSync: queue compacted ({len(self._queue)} pending)')
        except Exception as e:
            log.debug(f'WARPSync: compact failed: {e}')

    # ── Worker ─────────────────────────────────────────────────────────────
    def _worker_loop(self) -> None:
        # Tracks whether the previous iteration slept on an open breaker,
        # so we can emit a single "retrying" line at the moment the
        # breaker lifts. Without this signal the user sees the WARN, then
        # nothing for a full backoff window, even though the worker
        # actively re-attempts at the 5-min mark.
        was_waiting = False
        while not self._stop:
            with self._cv:
                while not self._queue and not self._stop:
                    self._cv.wait()
                if self._stop:
                    return
                item = self._queue[0]

            # If breaker is open, sleep until it expires (poll in chunks
            # so a stop signal can still bring us down quickly).
            remaining = self._client._backend_unavailable_until - time.time()
            if remaining > 0:
                was_waiting = True
                time.sleep(min(remaining + 1, 30))
                continue

            if was_waiting:
                log.info(
                    f'WARPSync: backoff window elapsed, retrying '
                    f'({len(self._queue)} queued)'
                )
                was_waiting = False

            outcome = self._client._send_one(item)
            with self._cv:
                if outcome in ('sent', 'dropped'):
                    if self._queue and self._queue[0] is item:
                        self._queue.popleft()
                    self._ack_to_disk(item['id'])
                    if outcome == 'sent':
                        self._sent_since_drain += 1
                    if not self._queue:
                        if self._sent_since_drain > 0:
                            log.info(
                                f'WARPSync: drained queue '
                                f'({self._sent_since_drain} contribution(s) sent)'
                            )
                            self._sent_since_drain = 0
                    elif (outcome == 'sent'
                          and self._sent_since_drain > 0
                          and self._sent_since_drain % self._client._PROGRESS_EVERY == 0):
                        # Periodic progress signal while draining a long
                        # queue — without this the user would see nothing
                        # between "backend recovered" and "drained queue"
                        # which can be many minutes apart at rate-limit
                        # pace.
                        log.info(
                            f'WARPSync: sent {self._sent_since_drain} '
                            f'contribution(s) ({len(self._queue)} remaining)'
                        )
                # outcome == 'retry': leave item at queue head; breaker is
                # now open, next loop iteration will sleep until it lifts.


class WARPSyncClient:
    """
    Non-blocking client for WARP community knowledge sync.

    Usage:
        client = WARPSyncClient()                 # start background download
        overrides = client.get_knowledge()        # {phash_hex: item_name}

        client.contribute(
            crop_bgr    = np.ndarray,             # icon crop (BGR)
            item_name   = 'Fleet Deflector...',   # confirmed name
            wrong_name  = 'Other Item...',        # what WARP guessed (optional)
            confirmed   = True,                   # user explicitly confirmed
        )
    """

    # Circuit breaker: after a 503/network error, stop contributing for this
    # many seconds to avoid log spam when the backend is sleeping (Render cold-start
    # takes ~50 s but we back off for longer to avoid hammering it).
    _BACKOFF_SECONDS    = 300    # 5 minutes between retries during outage
    _OUTAGE_HEARTBEAT_S = 300    # aligned with backoff so heartbeat fires right after each retry cycle
    _PROGRESS_EVERY     = 10     # emit progress INFO every N successful sends

    def __init__(self, backend_url: str | None = None):
        self._url       = (backend_url or self._load_config_url()
                           or DEFAULT_BACKEND_URL).rstrip('/')
        self._install_id = self._get_or_create_install_id()
        self._knowledge: dict[str, str] = {}   # phash_hex → item_name
        self._knowledge_lock = threading.Lock()

        # Icon equivalence classes (admin-curated). Each entry is a
        # frozenset of item names that share identical icon art. The
        # trainer consults `are_equivalent()` before raising a
        # community-conflict between two names — same icon ⇒ no nag.
        self._icon_equiv_classes: list[frozenset[str]] = []
        self._icon_equiv_index: dict[str, int] = {}   # name → class index
        self._icon_equiv_lock = threading.Lock()
        self._backend_unavailable_until: float = 0.0   # epoch seconds
        # True between the first WARN of an outage and the next successful
        # send. Suppresses repeat WARNs every backoff window — log noise
        # should reflect state transitions (READY → BACKOFF, BACKOFF →
        # drained), not periodic breaker re-trips while nothing changes.
        self._outage_warned: bool = False
        # Timestamp of the last outage-related log line (WARN or heartbeat
        # INFO). Used to emit a periodic "still down" heartbeat every
        # _OUTAGE_HEARTBEAT_S while the outage persists, so a long
        # silence doesn't make the worker look dead.
        self._outage_last_log_ts: float = 0.0

        # Single-worker contribute queue (replaces per-call thread spawn).
        # Constructed AFTER _backend_unavailable_until so the worker can
        # safely read it from the start.
        self._queue = _ContributeQueue(self, userdata.contribute_queue_file())

        # Start background knowledge download immediately
        threading.Thread(
            target=self._download_knowledge_bg,
            daemon=True, name='warp-knowledge-dl'
        ).start()
        # Same for the icon-equivalence JSON (independent endpoint, no
        # ordering between the two).
        threading.Thread(
            target=self._download_icon_equivalence_bg,
            daemon=True, name='warp-icon-equiv-dl'
        ).start()

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_knowledge(self) -> dict[str, str]:
        """
        Return current community knowledge dict {phash_hex: item_name}.
        Always returns instantly (uses cached value, updated in background).
        """
        with self._knowledge_lock:
            return dict(self._knowledge)

    def get_icon_equivalence_classes(self) -> list[frozenset[str]]:
        """Snapshot of the curated equivalence classes.

        Each frozenset is a group of STO item names that share identical
        icon art. Empty list = file not yet downloaded or repo missing
        the file. Always returns instantly (in-memory copy).
        """
        with self._icon_equiv_lock:
            return list(self._icon_equiv_classes)

    def are_equivalent(self, name_a: str, name_b: str) -> bool:
        """True when both names belong to the same equivalence class.

        Used by the trainer to skip community-conflict prompts when the
        on-disk label and the fresh community proposal point at items
        that look identical. Returns False if either name is unknown or
        if the equivalence file hasn't downloaded yet — the safe default
        is to keep the conflict UI behaviour intact.
        """
        if not name_a or not name_b or name_a == name_b:
            return name_a == name_b and bool(name_a)
        with self._icon_equiv_lock:
            idx_a = self._icon_equiv_index.get(name_a)
            idx_b = self._icon_equiv_index.get(name_b)
        return idx_a is not None and idx_a == idx_b

    def backend_status(self) -> str:
        """One-line snapshot of the circuit-breaker state for log/UI display.

        `READY`               — no recent failure recorded; next contribute
                                will hit the backend.
        `BACKOFF until HH:MM:SS` — last contribute failed; the queue worker
                                is waiting until the timestamp before
                                trying again. Pending items stay queued.

        When the queue has pending items, a `(N queued)` suffix is added.
        """
        remaining = self._backend_unavailable_until - time.time()
        depth = self._queue.depth() if hasattr(self, '_queue') else 0
        suffix = f' ({depth} queued)' if depth > 0 else ''
        if remaining <= 0:
            return f'READY{suffix}'
        until = time.strftime('%H:%M:%S',
                              time.localtime(self._backend_unavailable_until))
        return f'BACKOFF until {until}{suffix}'

    def contribute(
        self,
        crop_bgr:   np.ndarray,
        item_name:  str,
        wrong_name: str = '',
        confirmed:  bool = True,
        on_done:    Callable[[bool], None] | None = None,
    ) -> None:
        """
        Upload a crop + label to the community knowledge base.

        Non-blocking — encodes the crop, hands it to the single-worker
        contribute queue, and returns immediately. The actual POST may
        happen later (after backend recovery from a backoff window or
        after preceding queued items have drained).

        `on_done(success: bool)` is currently fired only on synchronous
        rejection paths (empty name, rate-limit, queue overflow). The
        async send path does not callback — callers should not depend on
        it for UI state. (Existing call sites at trainer_window.py:1877
        ignore the callback.)
        """
        _name = item_name.strip()
        if not _name:
            log.debug('WARPSync: contribute called with empty item_name, skipped')
            return

        # Mirror backend's poison-label filter (sets-warp-backend main.py:188).
        # Virtual classes (__empty__, __inactive__, __boff_*) are legitimate
        # for local training but must never reach the community knowledge
        # dataset, where they'd hard-override real icons. Filtering here
        # (instead of letting the backend reject with HTTP 400) avoids
        # filling the queue with items we know are going to be dropped,
        # plus the WARN spam on each rejection.
        if _name.startswith('__') or _name == 'Test Item Name':
            log.debug(f'WARPSync: skipping ineligible label {_name!r}')
            return

        if not self._check_rate_limit():
            log.debug('WARPSync: rate limit reached for today, skipping contribution')
            if on_done:
                on_done(False)
            return

        try:
            import cv2
            ok, buf = cv2.imencode('.png', crop_bgr)
            if not ok:
                raise ValueError('imencode failed')
            crop_b64 = base64.b64encode(buf.tobytes()).decode('ascii')
            icon64   = cv2.resize(crop_bgr, (64, 64), interpolation=cv2.INTER_AREA)
            phash    = _compute_phash(icon64)
        except Exception as e:
            log.warning(f'WARPSync: contribute encode failed ({e}); dropped')
            if on_done:
                on_done(False)
            return

        item = {
            'id':            str(uuid.uuid4()),
            'ts':            time.time(),
            'timestamp_iso': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'phash':         phash,
            'crop_png_b64':  crop_b64,
            'item_name':     item_name,
            'wrong_name':    wrong_name,
            'confirmed':     confirmed,
        }
        accepted = self._queue.enqueue(item)
        if not accepted and on_done:
            on_done(False)

    def refresh_knowledge(self) -> None:
        """Force re-download of knowledge base (ignores cache age)."""
        threading.Thread(
            target=self._download_knowledge_bg,
            args=(True,),
            daemon=True, name='warp-knowledge-refresh'
        ).start()

    # ── Background workers ─────────────────────────────────────────────────────

    def _download_knowledge_bg(self, force: bool = False) -> None:
        """Download knowledge.json from backend; update self._knowledge."""

        # Check cache freshness
        cache_path = userdata.knowledge_cache_file()
        if not force and cache_path.exists():
            try:
                mtime = cache_path.stat().st_mtime
                age_h = (time.time() - mtime) / 3600
                if age_h < KNOWLEDGE_MAX_AGE_HOURS:
                    data = json.loads(cache_path.read_text(encoding='utf-8'))
                    with self._knowledge_lock:
                        self._knowledge = data.get('knowledge', {})
                    log.debug(f'WARPSync: loaded {len(self._knowledge)} knowledge entries from cache')
                    return
            except Exception as e:
                log.debug(f'WARPSync: cache read failed: {e}')

        try:
            import urllib.request
            req = urllib.request.Request(
                f'{self._url}/knowledge',
                headers={'User-Agent': f'WARP/{WARP_VERSION}'},
            )
            with urllib.request.urlopen(req, timeout=config.SYNC_CONNECT_TIMEOUT + config.SYNC_READ_TIMEOUT) as resp:
                data = json.loads(resp.read().decode('utf-8'))

            knowledge = data.get('knowledge', {})

            with self._knowledge_lock:
                self._knowledge = knowledge

            # Save to cache
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps({'knowledge': knowledge, 'fetched_at': time.time()},
                           ensure_ascii=False, indent=2),
                encoding='utf-8'
            )
            log.info(f'WARPSync: downloaded {len(knowledge)} knowledge entries')

        except Exception as e:
            log.warning(f'WARPSync: knowledge download failed: {e}')
            # Fall back to cached data even if stale
            if cache_path.exists():
                try:
                    data = json.loads(cache_path.read_text(encoding='utf-8'))
                    with self._knowledge_lock:
                        self._knowledge = data.get('knowledge', {})
                    log.debug('WARPSync: using stale cache as fallback')
                except Exception:
                    pass

    def _apply_icon_equivalence(self, payload: dict) -> None:
        """Parse a curated icon_equivalence.json payload and publish the
        new in-memory snapshot. Tolerates either ``classes`` (a list of
        lists of names) or an absent / malformed payload (treated as
        "no classes")."""
        raw = payload.get('classes') if isinstance(payload, dict) else None
        classes: list[frozenset[str]] = []
        index: dict[str, int] = {}
        if isinstance(raw, list):
            for entry in raw:
                if not isinstance(entry, list):
                    continue
                names = {str(x) for x in entry if isinstance(x, str) and x}
                if len(names) < 2:
                    continue
                ci = len(classes)
                classes.append(frozenset(names))
                for n in names:
                    # First occurrence wins on duplicates across classes.
                    index.setdefault(n, ci)
        with self._icon_equiv_lock:
            self._icon_equiv_classes = classes
            self._icon_equiv_index = index

    def _download_icon_equivalence_bg(self, force: bool = False) -> None:
        """Mirror ``icon_equivalence.json`` from the HF dataset repo.

        Same cache-then-fetch policy as `_download_knowledge_bg`, but
        bypasses the WARP backend and hits the HF resolve URL directly
        because the file is small, public, and shouldn't depend on the
        Space being awake.
        """
        cache_path = userdata.icon_equivalence_cache_file()
        if not force and cache_path.exists():
            try:
                mtime = cache_path.stat().st_mtime
                age_h = (time.time() - mtime) / 3600
                if age_h < ICON_EQUIVALENCE_MAX_AGE_HOURS:
                    self._apply_icon_equivalence(
                        json.loads(cache_path.read_text(encoding='utf-8')))
                    log.debug(
                        f'WARPSync: loaded '
                        f'{len(self._icon_equiv_classes)} icon-equivalence '
                        f'classes from cache')
                    return
            except Exception as e:
                log.debug(f'WARPSync: icon-equiv cache read failed: {e}')

        url = (f'https://huggingface.co/datasets/{ICON_EQUIVALENCE_HF_REPO}'
               f'/resolve/main/{ICON_EQUIVALENCE_HF_FILE}')
        try:
            import urllib.request
            req = urllib.request.Request(
                url, headers={'User-Agent': f'WARP/{WARP_VERSION}'})
            with urllib.request.urlopen(
                    req,
                    timeout=config.SYNC_CONNECT_TIMEOUT
                            + config.SYNC_READ_TIMEOUT) as resp:
                raw = resp.read().decode('utf-8')
            payload = json.loads(raw)
            self._apply_icon_equivalence(payload)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(raw, encoding='utf-8')
            log.info(
                f'WARPSync: downloaded {len(self._icon_equiv_classes)} '
                f'icon-equivalence classes from HF')
        except Exception as e:
            # 404 = file not yet uploaded by admin — that's fine,
            # equivalence is opt-in. Anything else is just network noise.
            log.debug(f'WARPSync: icon-equiv download failed: {e}')
            if cache_path.exists():
                try:
                    self._apply_icon_equivalence(json.loads(
                        cache_path.read_text(encoding='utf-8')))
                    log.debug(
                        'WARPSync: using stale icon-equiv cache as fallback')
                except Exception:
                    pass

    def _send_one(self, item: dict) -> str:
        """Send a single queued item. Returns one of:

        - `'sent'`    — POST succeeded; caller should ack + drop the item.
        - `'dropped'` — backend rejected the payload with a 4xx (other
                        than 429); retrying would not help, so ack + drop.
        - `'retry'`   — network / 5xx error. Circuit breaker is now open;
                        the queue worker will sleep until it lifts and
                        try the same item again.

        Per-attempt errors log at DEBUG; only the final outcome of a
        burst (breaker trip or success drain) is logged at WARN/INFO
        level. This is the key noise reduction over the old design.
        """
        payload = json.dumps({
            'install_id':    self._install_id,
            'phash':         item['phash'],
            'crop_png_b64':  item['crop_png_b64'],
            'item_name':     item['item_name'],
            'wrong_name':    item.get('wrong_name', ''),
            'confirmed':     item.get('confirmed', True),
            'warp_version':  WARP_VERSION,
            'timestamp':     item.get('timestamp_iso',
                                      time.strftime('%Y-%m-%dT%H:%M:%SZ',
                                                    time.gmtime())),
        }, ensure_ascii=False).encode('utf-8')

        import urllib.request
        import urllib.error as _ue

        def _post() -> dict:
            req = urllib.request.Request(
                f'{self._url}/contribute',
                data=payload,
                headers={
                    'Content-Type': 'application/json',
                    'User-Agent':   f'WARP/{WARP_VERSION}',
                },
                method='POST',
            )
            with urllib.request.urlopen(req, timeout=config.SYNC_CONTRIBUTE_TIMEOUT) as resp:
                return json.loads(resp.read().decode('utf-8'))

        # Retry burst — Render free tier cold-starts in ~50 s and may
        # return 503 before it's ready. 3 attempts × 20 s covers that
        # without keeping the worker tied up indefinitely. Per-attempt
        # errors are intentionally not logged — the final outcome
        # (success / breaker trip) tells the user everything that
        # matters, and per-attempt lines just spam the log even at
        # DEBUG level when the backend is sleeping.
        last_err: Exception | None = None
        last_err_detail: str = ''
        result: dict | None = None
        for attempt in range(3):
            try:
                result = _post()
                break
            except _ue.HTTPError as e:
                last_err = e
                # Backend uses HTTPException(detail=...) which lands in the
                # response body as JSON {"detail":"..."}. urllib loses this
                # by default — read it so the WARN/dropped log carries the
                # real reason (e.g. "Storage unavailable, please try
                # later") instead of just "Service Unavailable".
                body = ''
                try:
                    body = e.read().decode('utf-8', errors='replace')[:200]
                    parsed = json.loads(body)
                    last_err_detail = str(parsed.get('detail', body))[:200]
                except Exception:
                    last_err_detail = body
                if 400 <= e.code < 500 and e.code != 429:
                    log.warning(
                        f'WARPSync: contribution rejected by backend '
                        f'(HTTP {e.code}: {last_err_detail or e.reason})'
                    )
                    return 'dropped'
                if attempt < 2:
                    time.sleep(20 if e.code == 503 else 5)
            except Exception as e:
                last_err = e
                last_err_detail = ''
                if attempt < 2:
                    time.sleep(5)

        if result is None:
            self._backend_unavailable_until = time.time() + self._BACKOFF_SECONDS
            now = time.time()
            depth = self._queue.depth()
            # Only WARN on the FIRST trip of an outage. Repeated trips at
            # the end of each 5-min backoff window add no new information
            # — the breaker state is still BACKOFF. To avoid going
            # completely silent on long outages (which makes the worker
            # look dead), emit an INFO heartbeat every
            # _OUTAGE_HEARTBEAT_S so the user can still see the queue
            # depth and confirm the worker is alive.
            if not self._outage_warned:
                reason = last_err_detail or str(last_err)
                log.warning(
                    f'WARPSync: backend unavailable ({reason}) — '
                    f'backing off {self._BACKOFF_SECONDS}s ({depth} queued)'
                )
                self._outage_warned     = True
                self._outage_last_log_ts = now
            elif now - self._outage_last_log_ts >= self._OUTAGE_HEARTBEAT_S:
                log.info(
                    f'WARPSync: backend still unavailable ({depth} queued)'
                )
                self._outage_last_log_ts = now
            return 'retry'

        success = result.get('ok', False)
        if success:
            # Recovery signal — first success after an outage tells the
            # user the worker just got through. depth() still includes
            # the item we're about to ack, so subtract 1 for "remaining
            # after this".
            if self._outage_warned:
                remaining = max(0, self._queue.depth() - 1)
                log.info(
                    f'WARPSync: backend recovered, resuming uploads '
                    f'({remaining} remaining)'
                )
                self._outage_warned      = False
                self._outage_last_log_ts = 0.0
            self._increment_rate_limit()
            return 'sent'
        else:
            log.warning(f'WARPSync: contribution rejected: {result.get("error")}')
            return 'dropped'

    # ── Rate limiting ──────────────────────────────────────────────────────────

    def _check_rate_limit(self) -> bool:
        """Return True if we're below the daily contribution limit."""
        try:
            p = userdata.rate_limit_file()
            if not p.exists():
                return True
            data = json.loads(p.read_text())
            today = str(date.today())
            if data.get('date') != today:
                return True
            return data.get('count', 0) < MAX_CONTRIBUTIONS_PER_DAY
        except Exception:
            return True

    def _increment_rate_limit(self) -> None:
        try:
            p = userdata.rate_limit_file()
            today = str(date.today())
            data = {}
            if p.exists():
                data = json.loads(p.read_text())
            if data.get('date') != today:
                data = {'date': today, 'count': 0}
            data['count'] = data.get('count', 0) + 1
            p.write_text(json.dumps(data))
        except Exception:
            pass

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _get_or_create_install_id(self) -> str:
        userdata.ensure_migrated()
        p = userdata.install_id_file()
        if p.exists():
            try:
                return p.read_text().strip()
            except Exception:
                pass
        install_id = str(uuid.uuid4())
        try:
            p.write_text(install_id)
        except Exception:
            pass
        return install_id

    def _load_config_url(self) -> str | None:
        try:
            userdata.ensure_migrated()
            p = userdata.backend_config_file()
            if p.exists():
                return json.loads(p.read_text()).get('backend_url')
        except Exception:
            pass
        return None


# ── pHash helper (standalone, mirrors icon_matcher) ───────────────────────────

def _compute_phash(icon64_bgr: np.ndarray) -> str:
    """
    Compute 64-bit perceptual hash of a 64×64 BGR icon.
    Returns lowercase hex string (16 chars).
    """
    import cv2
    gray  = cv2.cvtColor(icon64_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    small = cv2.resize(gray, (32, 32))
    dct   = np.fft.rfft2(small)
    low   = np.abs(dct[:8, :8]).flatten()
    mean  = low.mean()
    bits  = (low > mean).astype(np.uint8)
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return format(h & 0xFFFFFFFFFFFFFFFF, '016x')
