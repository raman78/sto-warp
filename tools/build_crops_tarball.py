"""Build & upload the community-crops tarball for sto-warp clients.

Runs in the dataset-tooling repo's GitHub Action. Pulls the current state
of `sets-sto/sto-icon-dataset`, packs `data/crops/*.png` +
`data/annotations.jsonl` into a single `crops.tar`, computes sha256,
writes a manifest, and uploads both back to the dataset repo.

Idempotent: if the dataset hasn't moved since the existing manifest's
`dataset_sha_at_build`, exits 0 without rebuilding.

Required env: HF_TOKEN (write scope on the dataset repo).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tarfile
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO          = 'sets-sto/sto-icon-dataset'
REPO_TYPE     = 'dataset'
TARBALL_FILE  = 'crops.tar'
MANIFEST_FILE = 'crops_manifest.json'


def main() -> int:
    token = os.environ.get('HF_TOKEN')
    if not token:
        print('error: HF_TOKEN not set', file=sys.stderr)
        return 2

    api = HfApi(token=token)
    current_sha = api.dataset_info(REPO).sha
    print(f'dataset SHA: {current_sha}')

    # Skip rebuild when the existing tarball already matches the dataset.
    # Explicit token=token everywhere — implicit env pickup in
    # huggingface_hub is inconsistent across call sites; without it the
    # parallel downloads run anonymous and trip the HF rate-limit at
    # ~2000 files (job times out after the 60-minute stall).
    try:
        existing_path = hf_hub_download(
            repo_id=REPO, repo_type=REPO_TYPE, filename=MANIFEST_FILE,
            token=token,
        )
        existing = json.loads(Path(existing_path).read_text())
        if existing.get('dataset_sha_at_build') == current_sha:
            print(f'tarball already current at {current_sha[:8]} — nothing to do')
            return 0
        print(f'existing manifest at {existing.get("dataset_sha_at_build", "?")[:8]} — '
              f'rebuilding')
    except Exception as e:
        print(f'no existing manifest (or unreadable): {e}')

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # Enumerate matching files first — gives progress visibility and
        # avoids snapshot_download's opaque stalls on large repos.
        all_files = list(api.list_repo_files(
            REPO, repo_type=REPO_TYPE, revision=current_sha,
        ))
        to_download = sorted(
            f for f in all_files
            if (f.startswith('data/crops/') and f.endswith('.png'))
            or f == 'data/annotations.jsonl'
        )
        total = len(to_download)
        print(f'files to download: {total}', flush=True)

        def _dl(fname: str) -> str:
            hf_hub_download(
                repo_id=REPO, repo_type=REPO_TYPE,
                filename=fname, revision=current_sha,
                local_dir=str(tmp), token=token,
            )
            return fname

        done = 0
        report_every = max(1, total // 10)
        with ThreadPoolExecutor(max_workers=4) as pool:
            for fut in as_completed(
                [pool.submit(_dl, f) for f in to_download]
            ):
                fut.result()
                done += 1
                if done % report_every == 0 or done == total:
                    print(f'  {done}/{total} downloaded', flush=True)

        crops_dir = tmp / 'data' / 'crops'
        ann_file  = tmp / 'data' / 'annotations.jsonl'
        tar_path  = tmp / TARBALL_FILE

        # Deterministic ordering + a stable mtime so the same dataset state
        # produces a byte-identical tarball on every run. Lets us short-
        # circuit when the dataset SHA hasn't moved but only the workflow
        # was triggered manually.
        crop_count = 0
        with tarfile.open(tar_path, 'w') as tar:
            if ann_file.exists():
                ti = tar.gettarinfo(str(ann_file), arcname='data/annotations.jsonl')
                ti.mtime = 0
                with open(ann_file, 'rb') as f:
                    tar.addfile(ti, f)
            for png in sorted(crops_dir.glob('*.png')):
                ti = tar.gettarinfo(str(png), arcname=f'data/crops/{png.name}')
                ti.mtime = 0
                with open(png, 'rb') as f:
                    tar.addfile(ti, f)
                crop_count += 1

        # sha256 + size
        h = hashlib.sha256()
        with open(tar_path, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                h.update(chunk)
        tar_sha   = h.hexdigest()
        tar_bytes = tar_path.stat().st_size
        ann_lines = sum(1 for _ in open(ann_file)) if ann_file.exists() else 0
        print(f'tarball: {tar_bytes/1e6:.1f} MB, {crop_count} crops, '
              f'sha256={tar_sha[:12]}…')

        manifest = {
            'tarball_file':         TARBALL_FILE,
            'tarball_sha256':       tar_sha,
            'tarball_bytes':        tar_bytes,
            'dataset_sha_at_build': current_sha,
            'crop_count':           crop_count,
            'annotations_lines':    ann_lines,
            'built_at':             time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        }
        manifest_path = tmp / MANIFEST_FILE
        manifest_path.write_text(json.dumps(manifest, indent=2))

        commit_msg = f'Rebuild crops tarball @ {current_sha[:8]}'
        api.upload_file(
            repo_id=REPO, repo_type=REPO_TYPE,
            path_or_fileobj=str(tar_path),
            path_in_repo=TARBALL_FILE,
            commit_message=commit_msg,
        )
        api.upload_file(
            repo_id=REPO, repo_type=REPO_TYPE,
            path_or_fileobj=str(manifest_path),
            path_in_repo=MANIFEST_FILE,
            commit_message=commit_msg,
        )
        print(f'uploaded {TARBALL_FILE} + {MANIFEST_FILE} at {current_sha[:8]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
