# 0010 — DVC for `experiments/` only

**Status:** accepted, 2026-10-24

## Context

Three directories hold large files: `data/raw/` (111MB), `experiments/` (71MB)
and `models/` (68MB). All are gitignored. The reflex is to version all of them;
the useful question is which of them you would actually lose something by
losing.

## Decision

DVC tracks **`experiments/` and nothing else.**

That directory is 71MB of `oof_*.npy` and `test_*.npy` — blend material,
declared immutable by `CLAUDE.md` ("Never clean them up"), and until now
protected by nothing but one laptop's disk. A deleted `.npy` is a model you have
to retrain. Versioning gives that invariant an actual mechanism instead of a
comment. 112 bytes in git, 65MB in the remote.

**`data/raw/` is deliberately not tracked.** `make data` re-downloads it from
Kaggle. DVC would duplicate 111MB to guard something one command restores, and
Kaggle is a more durable store than a directory on this machine.

**Model bundles are not either.** They ship as Release assets
([0006](0006-ghcr-and-releases.md)). Two channels for one artifact is how they
diverge.

A **directory-level** `dvc add` puts the pointer at `experiments.dvc` in the
repo root, which falls outside the `forbid-secrets-and-data` hook's
`^data/(raw|interim|processed)/` pattern. Per-file adds under `data/raw/` would
have produced `data/raw/train.csv.dvc` and forced widening a safety hook —
worth knowing, because that is the shape most DVC tutorials teach.

## Azure equivalent

**Azure Blob Storage** as the DVC remote (`dvc remote add -d azure
azure://container/path`) — the same `.dvc` pointers, a durable backend.
**Azure ML Data Assets** is the higher-level equivalent: versioned, named data
with lineage into runs, which is closer to what DVC is *for* than raw blob
versioning is.

## Rejected

- **Tracking everything.** 180MB duplicated to guard two things that already have
  better homes.
- **git-lfs for `experiments/`.** Billed bandwidth, and every clone pays.
- **A `dvc.yaml` pipeline.** Training already goes through `src.train.run`,
  which logs to `experiments.csv` and MLflow. A second orchestrator describing
  the same DAG is a thing to keep in sync, not a capability.

## Consequences

- **A training run writes into `experiments/`, so `dvc status` goes dirty until
  `make dvc-save`.** Documented in CHEATSHEET, because the surprise is worse than
  the chore.
- `experiments/.gitkeep` is untracked so DVC can own the directory;
  `src/config.py` recreates it at import, so a fresh clone is unaffected.
- The remote is a local directory, so this is versioning, not off-site backup.
- **Revisit when:** a second person needs the OOF files, at which point the
  remote should become Blob or S3 — a one-line config change.
