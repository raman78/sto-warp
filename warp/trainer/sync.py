# warp/trainer/sync.py
# Synchronises training data (annotations + icon crops) with Hugging Face Hub
# via the WARP backend proxy. The backend holds the HF write token; clients
# hold no credentials.
#
# SECURITY MODEL — two-folder staging (enforced on the backend side):
#
#   staging/<install_id>/annotations.jsonl   — contributed by users, unverified
#   staging/<install_id>/crops/<sha>.png
#   data/annotations.jsonl                   — approved by repo owner only
#   data/crops/<sha>.png
#
# Users POST to backend endpoints which write only to staging/ on HF.
# Repo owner runs approve_staging.py (or manually) to merge into data/.
# A single bad actor can only pollute their own staging/ folder.
#
# RATE LIMITING (client-side):
#   Max 1000 crops per day per install_id (client gate).
#   Backend also enforces a per-IP daily request cap.
#
# VALIDATION (client-side + backend-side):
#   - Crop must be non-empty PNG, ≥ 16×16 px (text crops use relaxed mins)
#   - Slot must be a known slot name
#   - Name must be non-empty, ≤ 120 chars, printable ASCII/Unicode
#   - No duplicate sha256 in the same upload batch

from __future__ import annotations

import base64
import json
import hashlib
import logging
import datetime
import urllib.request
import urllib.error
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal

from warp.trainer.training_data import TrainingDataManager, NON_ICON_SLOTS
from warp.knowledge.sync_client import DEFAULT_BACKEND_URL

logger = logging.getLogger(__name__)
try:
    from warp.debug import syslog as _slog
except Exception:
    _slog = logger

# Hugging Face dataset repository ID — kept for bootstrap reads (public).
# Writes go through the backend, not directly to HF.
HF_DATASET_REPO  = "sets-sto/sto-icon-dataset"
HF_REPO_TYPE     = "dataset"
CROPS_DIR        = "data/crops"           # approved
ANNOTATIONS_FILE = "data/annotations.jsonl"

STAGING_ROOT     = "staging"              # user contributions land here
MAX_DAILY_UPLOADS = 1000                  # per install_id, per UTC day
MAX_NAME_LEN      = 120
MIN_CROP_PX       = 16   # icon crops: minimum on both sides (BOFF ability icons can be ~22px)
MIN_TEXT_CROP_H   = 10   # text crops (ship_type/ship_tier): minimum height only
MIN_TEXT_CROP_W   = 50   # text crops: minimum width

# Backend batch sizes must stay ≤ MAX_BULK_* in sets-warp-backend/main.py.
BULK_CROPS_BATCH    = 50
BULK_SCREENS_BATCH  = 20
BULK_ANCHORS_BATCH  = 20
BACKEND_TIMEOUT_S   = 60


def _get_install_id() -> str:
    """Stable anonymous identifier for this installation."""
    try:
        from warp import userdata
        userdata.ensure_migrated()
        p = userdata.install_id_file()
        if p.exists():
            return p.read_text().strip()
        import uuid
        new_id = str(uuid.uuid4())
        try:
            p.write_text(new_id)
        except Exception:
            pass
        return new_id
    except Exception:
        import uuid
        return str(uuid.uuid4())[:16]


def _validate_annotation(item: dict) -> str | None:
    """Returns None if valid, or an error string."""
    name = item.get("name", "").strip()
    slot = item.get("slot", "").strip()
    if not name:
        return "empty name"
    if len(name) > MAX_NAME_LEN:
        return f"name too long ({len(name)})"
    if not slot:
        return "empty slot"
    if not name.isprintable():
        return "non-printable characters in name"
    return None


_TEXT_CROP_PREFIXES = ('ship_type_', 'ship_tier_')


def _stat_key(path: Path) -> tuple[int, int] | None:
    """Return (mtime_ns, size) for cache invalidation, or None if missing."""
    try:
        st = path.stat()
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size)


def _validate_crop(path: Path) -> str | None:
    """Returns None if valid PNG crop, or an error string.

    Text crops (ship_type / ship_tier) are wide horizontal bands — they use
    relaxed height/width minimums instead of the square icon minimum.
    """
    try:
        import cv2
        img = cv2.imread(str(path))
        if img is None:
            return "unreadable image"
        h, w = img.shape[:2]
        if any(path.name.startswith(p) for p in _TEXT_CROP_PREFIXES):
            if h < MIN_TEXT_CROP_H or w < MIN_TEXT_CROP_W:
                return f"too small ({w}×{h})"
        else:
            if h < MIN_CROP_PX or w < MIN_CROP_PX:
                return f"too small ({w}×{h})"
        return None
    except Exception as e:
        return str(e)


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

class SyncWorker(QThread):
    """
    Uploads confirmed crops to staging/ on HF Hub via the WARP backend proxy.
    Never writes to data/ — only repo owner can approve staging.

    Signals:
        progress(percent, message)
        finished(success: bool)
    """

    progress = Signal(int, str)
    finished = Signal(bool)

    def __init__(
        self,
        data_manager: TrainingDataManager,
        mode: str = "upload",
        backend_url: str | None = None,
    ):
        super().__init__()
        self._mgr  = data_manager
        self._mode = mode
        self._url  = (backend_url or DEFAULT_BACKEND_URL).rstrip('/')

    def run(self):
        try:
            if self._mode in ("upload", "both"):
                self._upload()
                self._upload_screen_types()
                self._upload_anchors_grid()
            if self._mode in ("download", "both"):
                self._download()
            self.finished.emit(True)
        except Exception as e:
            logger.error(f"Sync error: {e}")
            self.finished.emit(False)

    # ------------------------------------------------------------- backend POST

    def _post(self, path: str, payload: dict) -> dict:
        """POST JSON to backend; raise on error, return parsed response.

        Caller is expected to catch and handle exceptions — the queued-upload
        loop logs a single WARN per failed channel and moves on, mirroring the
        non-fatal behaviour the old direct-HF path had.
        """
        data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        req = urllib.request.Request(
            f'{self._url}{path}',
            data=data,
            headers={
                'Content-Type': 'application/json',
                'User-Agent':   'WARP-Trainer',
            },
            method='POST',
        )
        with urllib.request.urlopen(req, timeout=BACKEND_TIMEOUT_S) as resp:
            return json.loads(resp.read().decode('utf-8'))

    # ---------------------------------------------------------------- upload

    def _upload(self):
        """Upload new confirmed crops to staging/<install_id>/ via backend."""
        install_id   = _get_install_id()
        staging_dir  = f"{STAGING_ROOT}/{install_id}"
        staging_anno = f"{staging_dir}/annotations.jsonl"
        staging_crop = f"{staging_dir}/crops"

        confirmed = self._mgr.get_confirmed_crops()
        _slog.info(f'HF Sync: {len(confirmed)} confirmed crops locally')
        if not confirmed:
            self.progress.emit(100, "Nothing to upload.")
            return

        # Client-side rate limit check
        today     = datetime.datetime.utcnow().strftime('%Y-%m-%d')
        rl_file   = self._mgr._dir / '.sync_rate_limit.json'
        rl        = {}
        try:
            rl = json.loads(rl_file.read_text())
        except Exception:
            pass
        daily_count = rl.get(today, 0)
        _slog.info(f'HF Sync: {daily_count}/{MAX_DAILY_UPLOADS} uploads already sent today')
        if daily_count >= MAX_DAILY_UPLOADS:
            self.progress.emit(100, f"Daily upload limit ({MAX_DAILY_UPLOADS}) reached.")
            return

        # Load local cache of already-uploaded hashes (avoids list_repo_files on every sync).
        # Bootstrap reads are anonymous — the dataset is public.
        self.progress.emit(5, "Checking existing uploads…")
        existing_hashes = self._load_uploaded_hashes_cache()
        uploaded_labels = self._load_uploaded_labels_cache()
        if existing_hashes:
            _slog.info(f'HF Sync: {len(existing_hashes)} crops in local cache (skipping HF listing)')
        else:
            existing_hashes = self._fetch_staging_hashes(staging_crop)
            _slog.info(f'HF Sync: bootstrapped {len(existing_hashes)} hashes from HF listing')
            self._save_uploaded_hashes_cache(existing_hashes)
        if not uploaded_labels and existing_hashes:
            uploaded_labels = self._fetch_staging_labels(staging_anno)
            if uploaded_labels:
                _slog.info(f'HF Sync: bootstrapped {len(uploaded_labels)} labels from HF annotations.jsonl')
                self._save_uploaded_labels_cache(uploaded_labels)

        # Build the list of items to send. We send {slot, name, crop_png_b64,
        # ml_name?} per item; the backend handles annotations.jsonl merge
        # with last-wins per sha, so the client no longer mirrors that logic.
        batch_items:       list[dict] = []
        batch_metadata:    list[dict] = []   # parallel array: {sha, is_new_file}
        uploaded     = 0
        corrections  = 0
        skipped_unchanged = 0

        # Per-file (mtime, size) → (sha, valid) cache. Crops are content-
        # addressed and effectively immutable once written, so cache hit-rate
        # on subsequent syncs is ~100% — skips both the PNG decode in
        # _validate_crop and the full-file SHA hash.
        file_meta = self._load_file_meta_cache()
        file_meta_dirty = False

        self.progress.emit(10, "Preparing files…")
        for item in confirmed:
            if daily_count + uploaded >= MAX_DAILY_UPLOADS:
                break

            err = _validate_annotation(item)
            if err:
                logger.warning(f"Sync: skipping invalid annotation ({err}): {item}")
                continue

            crop_path = Path(item["path"])
            stat_key = _stat_key(crop_path)
            if stat_key is None:
                _slog.warning(f'HF Sync: crop file missing, skipping: {crop_path.name}')
                continue

            cached = file_meta.get(str(crop_path))
            if cached and cached.get('mtime_ns') == stat_key[0] and cached.get('size') == stat_key[1]:
                if not cached.get('valid', False):
                    continue
                sha = cached['sha']
            else:
                err = _validate_crop(crop_path)
                if err:
                    file_meta[str(crop_path)] = {'mtime_ns': stat_key[0], 'size': stat_key[1], 'valid': False}
                    file_meta_dirty = True
                    logger.warning(f"Sync: skipping invalid crop ({err}): {crop_path.name}")
                    continue
                sha = self._file_sha256(crop_path)
                file_meta[str(crop_path)] = {'mtime_ns': stat_key[0], 'size': stat_key[1], 'sha': sha, 'valid': True}
                file_meta_dirty = True

            file_already_on_hf = sha in existing_hashes
            current_label = f'{item["slot"]}|{item["name"]}'
            cached_label  = uploaded_labels.get(sha)
            label_changed = cached_label != current_label
            if file_already_on_hf and not label_changed:
                skipped_unchanged += 1
                continue

            try:
                png_b64 = base64.b64encode(crop_path.read_bytes()).decode('ascii')
            except Exception as e:
                _slog.warning(f'HF Sync: crop read failed for {crop_path.name}: {e}')
                continue

            payload_item = {
                'slot':         item['slot'],
                'name':         item['name'],
                'crop_png_b64': png_b64,
            }
            if item.get('ml_name') is not None:
                payload_item['ml_name'] = item['ml_name']
            batch_items.append(payload_item)
            batch_metadata.append({'sha': sha, 'is_new_file': not file_already_on_hf})

            if not file_already_on_hf:
                uploaded += 1
            else:
                corrections += 1

        total_to_send = len(batch_items)
        _slog.info(
            f'HF Sync: queued {uploaded} new crops + {corrections} label corrections '
            f'(skipped {skipped_unchanged} already on HF with unchanged labels)')

        if not batch_items:
            _slog.info('HF Sync: nothing new to upload')
            self.progress.emit(100, 'Nothing new to upload (all already on HF).')
            return

        # POST in batches of BULK_CROPS_BATCH; each batch is one HF commit
        # server-side. On a per-batch HTTP failure, log + break (rate limit /
        # outage would just repeat on every batch).
        sent = 0
        for start in range(0, total_to_send, BULK_CROPS_BATCH):
            sub_items = batch_items[start:start + BULK_CROPS_BATCH]
            sub_meta  = batch_metadata[start:start + BULK_CROPS_BATCH]
            self.progress.emit(
                10 + int(80 * start / max(1, total_to_send)),
                f'Uploading batch {start // BULK_CROPS_BATCH + 1} '
                f'({len(sub_items)} items)…',
            )
            try:
                resp = self._post('/contribute/bulk-crops', {
                    'install_id': install_id,
                    'items':      sub_items,
                })
            except urllib.error.HTTPError as e:
                body = e.read().decode('utf-8', errors='replace')[:300]
                _slog.warning(f'HF Sync: backend rejected batch (HTTP {e.code}): {body}')
                break
            except Exception as e:
                _slog.warning(f'HF Sync: backend POST failed: {e}')
                break

            accepted = int(resp.get('accepted', 0))
            sent    += accepted
            # Backend may have rejected some on poison-label / dim checks.
            # Log the raw response so we can see whatever fields the backend
            # exposes about rejections (count, reasons, per-item status).
            if accepted < len(sub_items):
                _slog.warning(
                    f'HF Sync: backend rejected {len(sub_items) - accepted}/'
                    f'{len(sub_items)} items in batch; response: {resp!r:.500}'
                )
            # Mark sent items in caches (only for accepted ones — backend may
            # have rejected some on poison-label / dim checks).
            for meta, payload_item in zip(sub_meta, sub_items):
                existing_hashes.add(meta['sha'])
                uploaded_labels[meta['sha']] = f'{payload_item["slot"]}|{payload_item["name"]}'

        if sent:
            self._save_uploaded_hashes_cache(existing_hashes)
            self._save_uploaded_labels_cache(uploaded_labels)
            _slog.info(f'HF Sync: backend accepted {sent}/{total_to_send} items')

        if file_meta_dirty:
            self._save_file_meta_cache(file_meta)

        # Update local rate limit counter (file uploads only — corrections are cheap)
        rl[today] = daily_count + uploaded
        try:
            rl_file.write_text(json.dumps(rl))
        except Exception:
            pass

        msg = f'Uploaded {uploaded} new crops + {corrections} corrections.'
        _slog.info(f'HF Sync: done — {uploaded} new crops, {corrections} corrections, total on HF: {len(existing_hashes)}')
        self.progress.emit(100, msg)

    def _load_file_meta_cache(self) -> dict:
        """Load per-file (mtime_ns, size) → (sha, valid) cache."""
        cache_file = self._mgr._dir / '.sync_file_meta.json'
        try:
            return dict(json.loads(cache_file.read_text()))
        except Exception:
            return {}

    def _save_file_meta_cache(self, meta: dict) -> None:
        cache_file = self._mgr._dir / '.sync_file_meta.json'
        try:
            cache_file.write_text(json.dumps(meta))
        except Exception:
            pass

    def _load_uploaded_hashes_cache(self) -> set[str]:
        """Load locally cached set of already-uploaded crop sha256 hashes."""
        cache_file = self._mgr._dir / '.sync_uploaded_hashes.json'
        try:
            return set(json.loads(cache_file.read_text()))
        except Exception:
            return set()

    def _save_uploaded_hashes_cache(self, hashes: set[str]) -> None:
        """Persist the uploaded-hashes set locally so future syncs skip HF listing."""
        cache_file = self._mgr._dir / '.sync_uploaded_hashes.json'
        try:
            cache_file.write_text(json.dumps(sorted(hashes)))
        except Exception:
            pass

    def _load_uploaded_labels_cache(self) -> dict[str, str]:
        """Load {sha: 'slot|name'} for already-uploaded annotation entries.

        Used to skip re-queueing annotation entries whose label matches what
        we previously uploaded — avoids no-op annotations.jsonl rewrites on
        every sync when nothing actually changed.
        """
        cache_file = self._mgr._dir / '.sync_uploaded_labels.json'
        try:
            return dict(json.loads(cache_file.read_text()))
        except Exception:
            return {}

    def _save_uploaded_labels_cache(self, labels: dict[str, str]) -> None:
        cache_file = self._mgr._dir / '.sync_uploaded_labels.json'
        try:
            cache_file.write_text(json.dumps(labels, sort_keys=True))
        except Exception:
            pass

    def _fetch_staging_labels(self, path_in_repo: str) -> dict[str, str]:
        """Bootstrap labels cache: read existing annotations.jsonl from HF and
        return {sha: 'slot|name'} for the most recent entry per sha.

        Anonymous read — the dataset is public.
        """
        try:
            from huggingface_hub import hf_hub_download
            local = hf_hub_download(
                repo_id=HF_DATASET_REPO,
                filename=path_in_repo,
                repo_type=HF_REPO_TYPE,
            )
        except Exception:
            return {}
        labels: dict[str, str] = {}
        try:
            with open(local, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except Exception:
                        continue
                    sha = d.get('crop_sha256')
                    if not sha:
                        continue
                    # Last-wins: later entries in the file override earlier ones.
                    labels[sha] = f'{d.get("slot","")}|{d.get("name","")}'
        except Exception:
            pass
        return labels

    def _fetch_staging_hashes(self, staging_crop_dir: str) -> set[str]:
        """Anonymous listing of already-uploaded staging crops on public dataset."""
        try:
            from huggingface_hub import HfApi
            # token=False: anonymous read by design (see REMOTE_SYNC_AUDIT.md).
            # Suppresses the "unauthenticated requests" warning from huggingface_hub.
            files = HfApi(token=False).list_repo_files(
                repo_id=HF_DATASET_REPO,
                repo_type=HF_REPO_TYPE,
            )
            return {
                Path(f).stem
                for f in files
                if f.startswith(staging_crop_dir) and f.endswith(".png")
            }
        except Exception:
            return set()

    # ---------------------------------------------------------------- anchor grids (P11)

    def _upload_anchors_grid(self):
        """Upload normalized bbox grid entries from local anchors.json via backend (P11)."""
        anchors_path = Path(self._mgr._dir) / 'anchors.json'
        if not anchors_path.exists():
            return
        try:
            data = json.loads(anchors_path.read_text(encoding='utf-8'))
        except Exception:
            return

        learned = data.get('learned', [])
        if not learned:
            return

        install_id = _get_install_id()

        cache_file = Path(self._mgr._dir) / '.sync_uploaded_grids.json'
        try:
            uploaded_grids: set[str] = set(json.loads(cache_file.read_text()))
        except Exception:
            uploaded_grids = set()

        grids_payload: list[dict] = []
        new_hashes:    list[str]  = []

        for entry in learned:
            build_type = entry.get('type', '')
            if not build_type:
                continue
            slots = entry.get('slots', {})
            icon_slots = {k: v for k, v in slots.items() if k not in NON_ICON_SLOTS}
            if len(icon_slots) < 3:
                continue

            grid = {
                'build_type': build_type,
                'aspect':     entry.get('aspect'),
                'resolution': entry.get('res', ''),
                'slots':      icon_slots,
            }
            # sha8 must match backend's canonical form (json.dumps sort_keys=True).
            sha8 = hashlib.sha256(
                json.dumps(grid, sort_keys=True, ensure_ascii=False).encode()
            ).hexdigest()[:8]
            if sha8 in uploaded_grids:
                continue

            grids_payload.append(grid)
            new_hashes.append(sha8)

        if not grids_payload:
            _slog.debug('HF Sync: no new anchor grids to upload')
            return

        sent = 0
        for start in range(0, len(grids_payload), BULK_ANCHORS_BATCH):
            sub      = grids_payload[start:start + BULK_ANCHORS_BATCH]
            sub_sha  = new_hashes[start:start + BULK_ANCHORS_BATCH]
            try:
                resp = self._post('/upload/anchors', {
                    'install_id': install_id,
                    'grids':      sub,
                })
            except urllib.error.HTTPError as e:
                body = e.read().decode('utf-8', errors='replace')[:300]
                _slog.warning(f'HF Sync: anchors backend rejected batch (HTTP {e.code}): {body}')
                break
            except Exception as e:
                _slog.warning(f'HF Sync: anchors backend POST failed: {e}')
                break
            sent += int(resp.get('accepted', 0))
            uploaded_grids.update(sub_sha)

        try:
            cache_file.write_text(json.dumps(sorted(uploaded_grids)))
        except Exception:
            pass
        _slog.info(f'HF Sync: uploaded {sent} anchor grid entries')

    # ---------------------------------------------------------------- screen types

    def _upload_screen_types(self):
        """Upload confirmed screen type screenshots via backend."""
        screen_types_dir = self._mgr._dir / 'screen_types'
        if not screen_types_dir.exists():
            return
        type_dirs = [d for d in screen_types_dir.iterdir() if d.is_dir()]
        if not type_dirs:
            return

        install_id  = _get_install_id()
        staging_dir = f"{STAGING_ROOT}/{install_id}/screen_types"

        screen_cache_file = self._mgr._dir / '.sync_uploaded_screen_hashes.json'
        existing = self._load_screen_hashes_cache(screen_cache_file)
        if existing:
            _slog.info(f'HF Sync: {len(existing)} screen type screenshots in local cache (skipping HF listing)')
        else:
            existing = self._fetch_staging_screen_hashes(staging_dir)
            _slog.info(f'HF Sync: bootstrapped {len(existing)} screen hashes from HF listing')

        total_sent = 0
        for type_dir in sorted(type_dirs):
            stype = type_dir.name
            # Build list of {png_b64} for crops we haven't uploaded yet, in
            # parallel with the sha list so we can update the cache after
            # backend ack.
            payloads: list[dict] = []
            shas:     list[str]  = []
            for png in sorted(type_dir.glob('*.png')):
                sha = self._file_sha256(png)
                if sha in existing:
                    continue
                try:
                    b64 = base64.b64encode(png.read_bytes()).decode('ascii')
                except Exception as e:
                    _slog.warning(f'HF Sync: screen-type read failed for {png.name}: {e}')
                    continue
                payloads.append({'png_b64': b64})
                shas.append(sha)

            if not payloads:
                continue

            # POST in batches of BULK_SCREENS_BATCH per screen_type.
            for start in range(0, len(payloads), BULK_SCREENS_BATCH):
                sub      = payloads[start:start + BULK_SCREENS_BATCH]
                sub_shas = shas[start:start + BULK_SCREENS_BATCH]
                try:
                    resp = self._post('/upload/screen-types', {
                        'install_id':  install_id,
                        'screen_type': stype,
                        'items':       sub,
                    })
                except urllib.error.HTTPError as e:
                    body = e.read().decode('utf-8', errors='replace')[:300]
                    _slog.warning(f'HF Sync: screen-types backend rejected (HTTP {e.code}): {body}')
                    break
                except Exception as e:
                    _slog.warning(f'HF Sync: screen-types backend POST failed: {e}')
                    break
                total_sent += int(resp.get('accepted', 0))
                existing.update(sub_shas)

        if not total_sent:
            _slog.debug('HF Sync: no new screen type screenshots to upload')
            return

        try:
            screen_cache_file.write_text(json.dumps(sorted(existing)))
        except Exception:
            pass
        _slog.info(f'HF Sync: uploaded {total_sent} screen type screenshot(s)')

    @staticmethod
    def _load_screen_hashes_cache(cache_file: Path) -> set[str]:
        try:
            return set(json.loads(cache_file.read_text()))
        except Exception:
            return set()

    def _fetch_staging_screen_hashes(self, staging_screen_dir: str) -> set[str]:
        """Anonymous listing of already-uploaded screen-type crops."""
        try:
            from huggingface_hub import HfApi
            # token=False: anonymous read by design (see REMOTE_SYNC_AUDIT.md).
            files = HfApi(token=False).list_repo_files(
                repo_id=HF_DATASET_REPO,
                repo_type=HF_REPO_TYPE,
            )
            return {
                Path(f).stem
                for f in files
                if f.startswith(staging_screen_dir) and f.endswith('.png')
            }
        except Exception:
            return set()

    # ---------------------------------------------------------------- download

    def _download(self):
        """Download approved annotations from data/ (not staging)."""
        self.progress.emit(10, "Downloading approved annotations…")
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(
                repo_id=HF_DATASET_REPO,
                filename=ANNOTATIONS_FILE,
                repo_type=HF_REPO_TYPE,
            )
            with open(path) as _f:
                count = sum(1 for line in _f if line.strip())
            self.progress.emit(100, f"Downloaded {count} approved annotations.")
        except Exception as e:
            logger.warning(f"Download failed: {e}")
            self.progress.emit(100, "Download failed — dataset may be empty.")

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _file_sha256(path: Path) -> str:
        sha = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha.update(chunk)
        return sha.hexdigest()[:32]


# ---------------------------------------------------------------------------
# SyncManager — WARP-specific background sync task
# ---------------------------------------------------------------------------

class SyncManager(QObject):
    """
    WARP background sync — uploads confirmed crops / screen-types / anchors to HF Hub.

    Designed to work with BackgroundTaskManager (src.background_tasks):
        btm.register(sync_mgr.check_and_upload, interval_ms=10*60*1000, startup_delay_ms=15_000)
        btm.on_stop(sync_mgr.stop)

    check_and_upload() is a single sync cycle; safe to call from any periodic timer.

    `progress(int, str)` re-emits SyncWorker progress and also fires for early-
    return paths (no token / no data / nothing to upload) so the launcher
    status bar can mirror what the syslog records.
    """

    progress = Signal(int, str)

    def __init__(self, sets_app, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._sets_app = sets_app
        self._worker: SyncWorker | None = None

    # ---------------------------------------------------------------- public API

    def check_and_upload(self) -> None:
        """One sync cycle: upload pending crops if any. Called by BackgroundTaskManager.

        No more HF-token gate — uploads flow through the WARP backend, which
        holds the only write-capable HF token. The client just needs a
        training data manager with confirmed crops.
        """
        mgr = self._data_manager()
        if mgr is None:
            _slog.info(
                'SyncManager: skipped — no training data manager available '
                '(WARP CORE never opened and no on-disk training_data/)')
            self.progress.emit(0, 'Upload skipped — no training data')
            return

        confirmed = [c for c in mgr.get_confirmed_crops() if Path(c['path']).exists()]
        if not confirmed:
            _slog.info('SyncManager: skipped — no confirmed crops to upload')
            self.progress.emit(0, 'Upload skipped — no confirmed crops')
            return

        if self._worker and self._worker.isRunning():
            _slog.info('SyncManager: skipped — upload already in progress')
            return

        _slog.info(f'SyncManager: {len(confirmed)} confirmed crops — checking for new uploads…')
        self._worker = SyncWorker(data_manager=mgr, mode='upload')
        self._worker.finished.connect(self._on_finished)
        # Forward worker's per-stage progress to coordinator/launcher.
        self._worker.progress.connect(self.progress)
        self._worker.start()

    def stop(self) -> None:
        """Wait for any in-progress upload to finish. Call on app quit."""
        if self._worker and self._worker.isRunning():
            self._worker.wait(5000)

    # ---------------------------------------------------------------- private

    def _on_finished(self, ok: bool) -> None:
        mgr = self._data_manager()
        remaining = 0
        if ok:
            _slog.info('SyncManager: upload OK')
        else:
            _slog.warning('SyncManager: upload FAILED')

    def _data_manager(self):
        """Return TrainingDataManager — prefer WARP CORE's live instance if window is open."""
        win = getattr(self._sets_app, '_warp_core_window', None)
        if win is not None and hasattr(win, '_data_mgr'):
            return win._data_mgr
        # WARP CORE not open — create a read-only instance from disk
        try:
            from warp.trainer.training_data import TrainingDataManager
            from warp import userdata
            data_dir = userdata.training_data_dir()
            if data_dir.exists():
                return TrainingDataManager(str(data_dir))
        except Exception:
            pass
        return None

    @staticmethod
    def _load_uploaded_hashes(mgr) -> set[str]:
        cache_file = mgr._dir / '.sync_uploaded_hashes.json'
        try:
            return set(json.loads(cache_file.read_text()))
        except Exception:
            return set()


