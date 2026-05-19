"""Mirror of the approved-truth crops + labels on HF Hub.

WARP's per-user icon matcher seeds a k-NN session-example pool from
locally confirmed crops so recognition recovers when the trained model
misfires. That worked per-user, which meant User A's confirmations only
helped A — User B (same model, same screenshot) saw a worse hit. To give
every install the same baseline we mirror the maintainer-approved
`data/` folder of `sets-sto/sto-icon-dataset` into the user's XDG cache:

  ~/.cache/warp/community_crops/
      data/annotations.jsonl     (approved labels — last write wins per sha)
      data/crops/<sha>.png       (approved crops)

`CommunityCropsClient.fetch()` is idempotent: huggingface_hub's
`snapshot_download` content-addresses files, so the second call only
pulls deltas. The SyncCoordinator runs it every cycle alongside the
knowledge / model refresh, and the icon matcher seeds itself from
`community_crops_dir() + community_annotations_file()` on warmup.

This path is intended to retire once the trained model alone is strong
enough — at that point the local k-NN layer goes away.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from warp import userdata
from warp.debug import syslog as log

HF_DATASET_REPO = "sets-sto/sto-icon-dataset"
HF_REPO_TYPE    = "dataset"

_ALLOW_PATTERNS = ['data/annotations.jsonl', 'data/crops/*.png']


def community_root() -> Path:
    p = userdata.cache_dir() / 'community_crops'
    p.mkdir(parents=True, exist_ok=True)
    return p


def community_crops_dir() -> Path:
    return community_root() / 'data' / 'crops'


def community_annotations_file() -> Path:
    return community_root() / 'data' / 'annotations.jsonl'


@dataclass
class CommunityCropsSnapshot:
    annotations: int
    crops:       int
    ok:          bool


class CommunityCropsClient:
    """Downloads + caches the approved-truth crops from HF Hub."""

    def fetch(self) -> CommunityCropsSnapshot:
        try:
            from huggingface_hub import snapshot_download
        except Exception as e:
            log.warning(f'community_crops: huggingface_hub unavailable: {e}')
            return CommunityCropsSnapshot(0, 0, ok=False)

        root = community_root()
        log.info(f'community_crops: snapshot_download {HF_DATASET_REPO} → {root}')
        try:
            snapshot_download(
                repo_id=HF_DATASET_REPO,
                repo_type=HF_REPO_TYPE,
                local_dir=str(root),
                allow_patterns=_ALLOW_PATTERNS,
            )
        except Exception as e:
            log.warning(f'community_crops: snapshot download failed: {e}')
            return self._scan(ok=False)

        return self._scan(ok=True)

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
