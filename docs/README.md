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
| [`sto_slots_rules.md`](sto_slots_rules.md) | Which slots exist per screen, and the game rules behind them |

## Data and sync

| Doc | Scope |
|---|---|
| [`DATA_LIFECYCLE.md`](DATA_LIFECYCLE.md) | One confirmation's journey: client → staging → merge → model → client |
| [`SYNC_ARCHITECTURE.md`](SYNC_ARCHITECTURE.md) | The seven refresh phases, their TTLs and freshness rules |
| [`CARGO_DATA_PLAN.md`](CARGO_DATA_PLAN.md) | Where item/ship reference data comes from and how it is cached |

## Workflows

| Doc | Scope |
|---|---|
| [`FAST_CORRECTION_MODE.md`](FAST_CORRECTION_MODE.md) | The throwaway workspace: staging, snapshot/restore, lifecycle |
| [`gpu_setup.md`](gpu_setup.md) | Optional GPU acceleration — for trainers, not for recognition |
| [`RELEASE_HOWTO.md`](RELEASE_HOWTO.md) | Cutting a release (in Polish) |

## Audits and planning

| Doc | Scope |
|---|---|
| [`data_source_audit.md`](data_source_audit.md) | Data-flow audit log, invariants Z1–Z6, decisions D-A…D-H (in Polish) |
| [`client_user_view_filter.md`](client_user_view_filter.md) | Z5 closure follow-up from that audit (in Polish) |
| [`REMOTE_SYNC_AUDIT.md`](REMOTE_SYNC_AUDIT.md) | Capacity check on the upload/download paths before wider release |
| [`warp_ml_roadmap.md`](warp_ml_roadmap.md) | Layout + content recognition roadmap, P0–P11 |

Backend-side documentation (ingestion, mergers, training jobs) lives in the
`sets-warp-backend` repository.
