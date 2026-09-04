# sto-warp documentation

**Start here:** [`WARP_GUIDE.md`](WARP_GUIDE.md) — the user guide for both
programs, from first run to exporting a build. Everything else on this page is
written for people who read or change the code.

## Recognition pipeline

| Doc | Scope |
|---|---|
| [`ML_PIPELINE.md`](ML_PIPELINE.md) | End-to-end ML: local capture, community training, model delivery |
| [`EQ_DETECTION.md`](EQ_DETECTION.md) | Equipment panel — geometry, row labelling, slot profile, tier recovery |
| [`BOFF_DETECTION.md`](BOFF_DETECTION.md) | BOFF panel — profession markers, seat layout |
| [`TRAIT_DETECTION.md`](TRAIT_DETECTION.md) | Trait panels — grid localisation, section assignment |
| [`SHIP_INFO_DETECTION.md`](SHIP_INFO_DETECTION.md) | Ship name / class / tier from the top band, and their bboxes |
| [`EMPTY_AND_INACTIVE_SLOTS.md`](EMPTY_AND_INACTIVE_SLOTS.md) | Recognising that a slot holds nothing: the fixed rule, the models, and what the labels are for |
| [`sto_slots_rules.md`](sto_slots_rules.md) | Which slots exist per screen, and the game rules behind them |

## Data and sync

| Doc | Scope |
|---|---|
| [`DATA_LIFECYCLE.md`](DATA_LIFECYCLE.md) | One confirmation's journey: client → staging → merge → model → client |
| [`SYNC_ARCHITECTURE.md`](SYNC_ARCHITECTURE.md) | The seven refresh phases, their TTLs and freshness rules |
| [`CARGO_DATA_PLAN.md`](CARGO_DATA_PLAN.md) | Where item/ship reference data comes from, how it is cached, and how a row maps back to its wiki page |
| [`SETS_FORMAT_CONTRACT.md`](SETS_FORMAT_CONTRACT.md) | The build-JSON format SETS reads: how it is frozen, validated on export, and watched for upstream drift |

## Workflows

| Doc | Scope |
|---|---|
| [`FAST_CORRECTION_MODE.md`](FAST_CORRECTION_MODE.md) | The throwaway workspace: staging, snapshot/restore, lifecycle |
| [`REVIEW_MERGE.md`](REVIEW_MERGE.md) | How the review panel merges confirmed annotations with fresh detection |
| [`gpu_setup.md`](gpu_setup.md) | Optional GPU acceleration — for trainers, not for recognition |
| [`RELEASE_HOWTO.md`](RELEASE_HOWTO.md) | Cutting a release (in Polish) |
| [`MAINTAINER_TOOLS.md`](MAINTAINER_TOOLS.md) | Which diagnostic or repair tool answers which question, in both repos |

## Audits and planning

| Doc | Scope |
|---|---|
| [`data_source_audit.md`](data_source_audit.md) | Data-flow audit log, invariants Z1–Z6, decisions D-A…D-H (in Polish) |
| [`client_user_view_filter.md`](client_user_view_filter.md) | Where `__empty__` / `__inactive__` are stopped before they reach a user or a build — Z5 closure from that audit |
| [`REMOTE_SYNC_AUDIT.md`](REMOTE_SYNC_AUDIT.md) | Capacity check on the upload/download paths before wider release |
| [`warp_ml_roadmap.md`](warp_ml_roadmap.md) | Layout + content recognition roadmap, P0–P11 |

Backend-side documentation (ingestion, mergers, training jobs) lives in the
`sets-warp-backend` repository.
