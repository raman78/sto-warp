# sto-icon-dataset tooling

Scaffolding for a separate GitHub repo that publishes a prebuilt tarball
+ manifest next to the `sets-sto/sto-icon-dataset` HuggingFace dataset.
The sto-warp client downloads the tarball in a single HTTP GET at cold
start instead of 8000+ per-file `hf_hub_download` calls (which trip the
HF anonymous rate-limit and turn first-run into 10+ minutes of 429 retry
waits).

## Layout

```
.github/workflows/build-crops-tarball.yml   # cron + manual trigger
scripts/build_crops_tarball.py              # snapshot → tar → upload
```

## How to set up

1. **Create a new GitHub repo** `sets-sto/sto-icon-dataset-tooling` (or
   any name — does not need to match the dataset repo).
2. **Copy this directory** (`tools/hf_dataset_workflow/`) into the root
   of the new repo, preserving the `.github/workflows/` layout.
3. **Add the HF write token** as a repo secret named
   `HF_DATASET_WRITE_TOKEN`. The token needs *write* scope on
   `sets-sto/sto-icon-dataset`.
4. **Trigger the workflow manually** once via the Actions tab → "Build
   community-crops tarball" → "Run workflow". This produces the first
   `crops.tar` + `crops_manifest.json` on the dataset repo.
5. After the initial run, the cron schedule (Mondays 04:00 UTC) keeps
   the tarball in sync. Manual `workflow_dispatch` works any time.

## Behaviour

- The script reads the dataset's current commit SHA, compares it to
  `crops_manifest.json.dataset_sha_at_build`, and exits 0 without
  rebuilding when they match. So a manual or scheduled run on an
  unchanged dataset costs only one `dataset_info` API call.
- When the dataset has moved, the script downloads the full snapshot,
  packs `data/crops/*.png` + `data/annotations.jsonl` into a tar with
  deterministic ordering and zero mtime, uploads the tar, then uploads
  the new manifest.
- No compression: PNGs don't compress meaningfully and stdlib `tarfile`
  ships with Python (no extra client dependency).

## Client-side hook

The sto-warp client (`warp.knowledge.community_crops.CommunityCropsClient`)
tries the tarball first on cold start. When the manifest is missing
(HTTP 404), the client falls back to `snapshot_download` — so it's safe
to ship the client update before this workflow is set up; users just see
the existing slower path until the tarball is published.
