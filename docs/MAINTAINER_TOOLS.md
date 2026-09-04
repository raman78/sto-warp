# Maintainer tools — what to reach for when something looks wrong

Addressed to whoever maintains the dataset and the models. Nothing here is
secret; both repos are public and the dataset is publicly readable. What makes
these maintainer tools is that they act on the shared dataset or interpret it,
which a user neither can nor should.

The user-facing half is one number: the **N not yet shared** counter in WARP
CORE's status bar, described in [`WARP_GUIDE.md`](WARP_GUIDE.md). It says that
something is pending, never why. Everything below answers *why*.

For cutting a release see [`RELEASE_HOWTO.md`](RELEASE_HOWTO.md); for how the
pipeline moves at all, [`DATA_LIFECYCLE.md`](DATA_LIFECYCLE.md).

---

## Start here: which question are you asking

| The question | Reach for |
|---|---|
| Did my machine's confirmations reach the dataset? | `warp.tools.reconcile_uploads` |
| Is the pipeline moving at all? | `admin_audit_pipeline_movement.py` |
| Is there junk in `data/`? | `admin_reject_crops.py`, `admin_console.py` |
| Is staging accumulating? | `admin_audit_staging.py` |
| Which items does SETS refuse on import? | `admin_sets_gaps_report.py` |
| Is my local training store poisoned? | `warp.tools.scrub_training_data` |
| Is the offline cargo snapshot stale? | `warp.tools.make_baseline --check` |

The first is in **sto-warp** and needs no token. The rest live in
**sets-warp-backend** and need `HF_TOKEN` (from its `.env`), because they
write to the dataset or read repository history.

---

## Diagnosis — read-only, safe to run any time

### Did my contributions arrive?

```
python -m warp.tools.reconcile_uploads              # both domains
python -m warp.tools.reconcile_uploads --domain screens
python -m warp.tools.reconcile_uploads --json
```

Compares this install's training store against the published dataset and
splits the difference **by cause**, which is the whole point:

| | Meaning |
|---|---|
| `unsent` | confirmed here, never submitted from here. A transport fault. |
| `outvoted` | submitted, and the tally settled on something else. Not a fault. |
| `absent` | in the dataset, not in this store. Usually a maintainer rejection. |

Exit status is 1 only for `unsent`. Being outvoted is the mechanism working,
and scoring it would make the exit status meaningless.

The split comes from the client's own upload cache, which records the label
each item was last *sent* under. Without it the two are indistinguishable —
and reporting them alike amounts to arguing the consensus should be corrected
to match one machine.

Reads only public files, so no token. Point it at another store with
`--store`.

### Is the pipeline moving?

```
python admin_audit_pipeline_movement.py
```

Runs daily at 07:00 UTC and fails the workflow on a breach, so an unread
inbox is the normal way to learn about this one. It reports a breach when
staging holds files no run can settle, or when crops are waiting and nothing
has promoted them.

### Is staging accumulating, or is `data/` poisoned?

```
python admin_audit_staging.py           # staging vs data/, read-only
python admin_audit_virtual_poison.py    # mislabelled virtual crops
```

Both run monthly. `admin_audit_virtual_poison` looks for the damaging
direction specifically: a real icon filed as `__empty__` or `__inactive__`.

### What is SETS refusing to import?

```
python admin_sets_gaps_report.py                    # grouped by cause
python admin_sets_gaps_report.py --min-installs 3
```

Counts **installs**, not exports, so one enthusiastic user cannot look like
demand. Groups by which project the gap belongs to — the wiki (no cargo row)
or SETS (a row its loader passes over). Ledgers not refreshed in 90 days stop
counting, so an uninstalled copy cannot keep voting.

---

## Repair — these write

### Fix or drop a bad label

```
python admin_reject_crops.py                        # scan, writes a TSV
# edit the TSV: REJECT / KEEP / RELABEL <name>
python admin_reject_crops.py --apply
```

`--direction` chooses what to hunt: `virtual` (a colourful icon filed as
blank), `real` (a blank cell filed under an item's name), or `tail` (the
weakest verdicts, ranked). `admin_console.py` is the same review with
pictures, which for a judgement about what a crop shows is usually the
faster tool.

A `RELABEL` is recorded in `data/reviewed_virtual.jsonl` and **pinned**: the
merge will not let a later client vote overwrite it, and a past overwrite is
undone on the next run. That is the one place a maintainer's decision outranks
the tally, and it is deliberate — a human looked at the picture.

### Repair the local training store

```
python -m warp.tools.scrub_training_data
```

Local only. Removes crops whose label the local evidence contradicts, before
they are ever uploaded.

### Refresh the offline cargo snapshot

```
python -m warp.tools.make_baseline --check     # compare sizes, change nothing
python -m warp.tools.make_baseline             # write it
```

Decide per release, and read [`CARGO_DATA_PLAN.md`](CARGO_DATA_PLAN.md)
first. The tool refuses a file that came back more than 5% smaller and exits
3; treat that as a signal to find out what upstream lost, not as something to
override.

---

## What runs without being asked

| Workflow | When | Fails the job on |
|---|---|---|
| `merge_staging.yml` | every 2 h | any merger crashing |
| `train_central_model.yml` | hourly | training error, or a collapsed model |
| `train_metric_model.yml` | daily 00:45 UTC | as above |
| `audit_pipeline_movement.yml` | daily 07:00 UTC | a movement breach |
| `audit_staging_health.yml` | monthly | staging health breach |
| `audit_virtual_poison.yml` | monthly | poison found |

A red run mails the repository owner, and that is the only push notification
in the system. Two things follow, both learned the hard way:

- **Green is not proof.** A merge failed for seven weeks behind a green tick
  because `| tee` swallowed the exit code. Every workflow step now sets
  `set -o pipefail`, and a step that pipes without it is caught by
  `tests/test_workflow_exit_codes.py`.
- **Red is only useful if read.** After the HF token was rotated, deploy and
  merge both failed correctly for hours and nobody noticed. If the mail is
  not watched, the daily audit is the backstop, not the alert.

### The publication guard

Both trainers refuse to publish a model that collapsed: below 90% of the last
release's class count, or more than ten points of accuracy lost. The refusal
fails the workflow, so it arrives through the same mail.

The thresholds are loose deliberately. This is a collapse detector, not a
quality gate — a model that trains a little worse is still a model. It exists
because one carrying 1592 of roughly 3000 classes was once published and
served, and nothing compared it against what it replaced.
