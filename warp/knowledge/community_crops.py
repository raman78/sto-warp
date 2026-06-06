"""Mirror of the approved-truth crops + labels on HF Hub.

WARP's per-user icon matcher seeds a k-NN session-example pool from
locally confirmed crops so recognition recovers when the trained model
misfires. That worked per-user, which meant User A's confirmations only
helped A — User B (same model, same screenshot) saw a worse hit. To give
every install the same baseline we mirror the maintainer-approved
`data/` folder of `sets-sto/sto-icon-dataset` into the user's XDG cache:

  ~/.cache/warp/community_crops/
      data/annotations.jsonl     (approved labels — last write wins per sha)
      data/crops/<sha>.png       (approved crops; filename ≡ content sha)
      .last_commit.sha           (dataset revision used last time)
      .trash/<YYYYMMDD-HHMMSS>/  (soft-deleted crops, last 3 snapshots kept)

`CommunityCropsClient.fetch()` is idempotent. When the upstream dataset
hasn't moved since the last successful sync, we short-circuit on the
dataset SHA. When it has moved, we diff the upstream file list against
the local mirror and download only the missing crops — content-addressed
filenames let us skip per-file hash verification entirely. The
SyncCoordinator runs it every cycle alongside the knowledge / model
refresh, and the icon matcher seeds itself from
`community_crops_dir() + community_annotations_file()` on warmup.

Cleanup safety (when upstream drops files):
- a percentage guard refuses to remove >30% of the mirror in one cycle
  while the mirror has at least 100 files (small mirrors are exempt to
  avoid spurious false positives);
- removals are soft-deletes into `.trash/<timestamp>/`, with the three
  most recent snapshots kept for manual recovery before being pruned.

This path is intended to retire once the trained model alone is strong
enough — at that point the local k-NN layer goes away.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from warp import userdata
from warp.debug import syslog as log

HF_DATASET_REPO = "sets-sto/sto-icon-dataset"
HF_REPO_TYPE    = "dataset"

_ALLOW_PATTERNS = ['data/annotations.jsonl', 'data/crops/*.png']

_CLEANUP_GUARD_FRACTION  = 0.30
_CLEANUP_GUARD_MIN_LOCAL = 100
_TRASH_KEEP_LAST         = 3
_MAX_DOWNLOAD_WORKERS    = 8


def community_root() -> Path:
    p = userdata.cache_dir() / 'community_crops'
    p.mkdir(parents=True, exist_ok=True)
    return p


def community_crops_dir() -> Path:
    return community_root() / 'data' / 'crops'


def community_annotations_file() -> Path:
    return community_root() / 'data' / 'annotations.jsonl'


def _trash_root() -> Path:
    return community_root() / '.trash'


def _assert_inside_mirror_crops(path: Path) -> None:
    """Refuse to touch anything outside `<cache>/community_crops/data/crops/`."""
    expected_parent = community_crops_dir().resolve()
    resolved = path.resolve()
    if resolved.parent != expected_parent:
        raise RuntimeError(
            f'refusing to mutate path outside mirror crops dir: '
            f'{resolved} (expected parent {expected_parent})'
        )


@dataclass
class CommunityCropsSnapshot:
    annotations: int
    crops:       int
    ok:          bool


class CommunityCropsClient:
    """Downloads + caches the approved-truth crops from HF Hub."""

    def fetch(self) -> CommunityCropsSnapshot:
        try:
            from huggingface_hub import HfApi, hf_hub_download
        except Exception as e:
            log.warning(f'community_crops: huggingface_hub unavailable: {e}')
            return CommunityCropsSnapshot(0, 0, ok=False)

        root = community_root()
        revision_file = root / '.last_commit.sha'

        # token=False: anonymous read by design (see REMOTE_SYNC_AUDIT.md).
        # Suppresses the "unauthenticated requests" warning from huggingface_hub.
        api = HfApi(token=False)
        current_sha: str | None = None
        try:
            current_sha = api.dataset_info(HF_DATASET_REPO).sha
        except Exception as e:
            log.warning(f'community_crops: dataset_info failed: {e}')

        if current_sha:
            try:
                cached_sha = revision_file.read_text().strip()
            except Exception:
                cached_sha = ''
            if cached_sha == current_sha:
                log.info(f'community_crops: mirror up-to-date at '
                         f'{current_sha[:8]} — skipping download')
                return self._scan(ok=True)

        if current_sha:
            ok = self._delta_sync(api, hf_hub_download, current_sha)
            if not ok:
                ok = self._fallback_full_snapshot(current_sha)
        else:
            # No upstream sha resolved — can't pin a revision for delta.
            log.warning('community_crops: no upstream sha resolved — '
                        'using full snapshot fallback')
            ok = self._fallback_full_snapshot(None)

        if ok and current_sha:
            try:
                revision_file.write_text(current_sha)
            except Exception:
                pass

        return self._scan(ok=ok)

    def _delta_sync(self, api, hf_hub_download, revision: str) -> bool:
        """Download only the crops upstream has that the local mirror lacks.

        Returns False if listing or any download fails; the caller falls
        back to a full snapshot_download.
        """
        try:
            upstream_files = api.list_repo_files(
                repo_id=HF_DATASET_REPO,
                repo_type=HF_REPO_TYPE,
                revision=revision,
            )
        except Exception as e:
            log.warning(f'community_crops: list_repo_files failed: {e}')
            return False

        upstream_crops: set[str] = {
            Path(p).name for p in upstream_files
            if p.startswith('data/crops/') and p.endswith('.png')
        }
        has_annotations = 'data/annotations.jsonl' in upstream_files

        crops_dir = community_crops_dir()
        crops_dir.mkdir(parents=True, exist_ok=True)
        local_crops: set[str] = {p.name for p in crops_dir.glob('*.png')}

        to_download = upstream_crops - local_crops
        to_remove   = local_crops - upstream_crops

        log.info(
            f'community_crops: delta sync at {revision[:8]} — '
            f'upstream={len(upstream_crops)} local={len(local_crops)} '
            f'+{len(to_download)} -{len(to_remove)}'
        )

        if has_annotations:
            try:
                hf_hub_download(
                    repo_id=HF_DATASET_REPO,
                    repo_type=HF_REPO_TYPE,
                    filename='data/annotations.jsonl',
                    revision=revision,
                    local_dir=str(community_root()),
                )
            except Exception as e:
                log.warning(f'community_crops: annotations.jsonl download failed: {e}')
                return False

        if to_download:
            failures = 0
            t0 = time.monotonic()
            with ThreadPoolExecutor(max_workers=_MAX_DOWNLOAD_WORKERS) as pool:
                futs = {
                    pool.submit(
                        hf_hub_download,
                        repo_id=HF_DATASET_REPO,
                        repo_type=HF_REPO_TYPE,
                        filename=f'data/crops/{name}',
                        revision=revision,
                        local_dir=str(community_root()),
                    ): name
                    for name in to_download
                }
                for fut in as_completed(futs):
                    try:
                        fut.result()
                    except Exception as e:
                        failures += 1
                        if failures <= 3:
                            log.warning(
                                f'community_crops: download {futs[fut]} failed: {e}'
                            )
            dt = time.monotonic() - t0
            log.info(
                f'community_crops: downloaded {len(to_download) - failures}'
                f'/{len(to_download)} crops in {dt:.1f}s'
            )
            if failures:
                # Don't pin the sha — next call retries the misses.
                return False

        if to_remove:
            self._soft_delete(to_remove)

        return True

    def _soft_delete(self, names: set[str]) -> None:
        """Move stale crops to `.trash/<timestamp>/`.

        Refuses to act when removals exceed _CLEANUP_GUARD_FRACTION of the
        current mirror, unless the mirror is below _CLEANUP_GUARD_MIN_LOCAL
        (small mirrors are exempt to avoid false positives). Recoverable
        for _TRASH_KEEP_LAST cycles before auto-prune.
        """
        crops_dir = community_crops_dir()
        local_count = sum(1 for _ in crops_dir.glob('*.png'))

        if local_count >= _CLEANUP_GUARD_MIN_LOCAL:
            frac = len(names) / max(local_count, 1)
            if frac > _CLEANUP_GUARD_FRACTION:
                log.warning(
                    f'community_crops: cleanup guard tripped — '
                    f'would remove {len(names)}/{local_count} '
                    f'({frac:.0%} > {_CLEANUP_GUARD_FRACTION:.0%}); '
                    f'keeping mirror intact'
                )
                return

        trash_dir = _trash_root() / time.strftime('%Y%m%d-%H%M%S')
        trash_dir.mkdir(parents=True, exist_ok=True)
        moved = 0
        for name in names:
            src = crops_dir / name
            if not src.exists():
                continue
            _assert_inside_mirror_crops(src)
            try:
                src.rename(trash_dir / name)
                moved += 1
            except Exception as e:
                log.warning(f'community_crops: soft-delete {name} failed: {e}')
        log.info(f'community_crops: soft-deleted {moved} stale crops → {trash_dir.name}')
        self._prune_trash()

    def _prune_trash(self) -> None:
        """Keep only the most recent _TRASH_KEEP_LAST `.trash/<ts>/` snapshots."""
        trash = _trash_root()
        if not trash.exists():
            return
        snapshots = sorted(
            (p for p in trash.iterdir() if p.is_dir()),
            key=lambda p: p.name,
            reverse=True,
        )
        for old in snapshots[_TRASH_KEEP_LAST:]:
            try:
                for f in old.iterdir():
                    f.unlink(missing_ok=True)
                old.rmdir()
            except Exception as e:
                log.warning(f'community_crops: prune trash {old.name} failed: {e}')

    def _fallback_full_snapshot(self, revision: str | None) -> bool:
        try:
            from huggingface_hub import snapshot_download
        except Exception as e:
            log.warning(f'community_crops: huggingface_hub unavailable: {e}')
            return False

        log.info(f'community_crops: snapshot_download {HF_DATASET_REPO} '
                 f'(fallback) → {community_root()}')
        try:
            snapshot_download(
                repo_id=HF_DATASET_REPO,
                repo_type=HF_REPO_TYPE,
                local_dir=str(community_root()),
                allow_patterns=_ALLOW_PATTERNS,
                revision=revision,
            )
            return True
        except Exception as e:
            log.warning(f'community_crops: snapshot download failed: {e}')
            return False

    def _scan(self, ok: bool) -> CommunityCropsSnapshot:
        ann = community_annotations_file()
        n_ann = 0
        if ann.exists():
            try:
                with open(ann, encoding='utf-8') as f:
                    n_ann = sum(1 for line in f if line.strip())
            except Exception:
                pass
        crops_dir = community_crops_dir()
        n_crops = sum(1 for _ in crops_dir.glob('*.png')) if crops_dir.exists() else 0
        verb = 'ready' if ok else 'stale (download failed)'
        log.info(f'community_crops: mirror {verb} — '
                 f'{n_ann} annotations, {n_crops} crops')
        return CommunityCropsSnapshot(n_ann, n_crops, ok=ok)
