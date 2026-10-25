# 0001 — MLflow as a second sink, not a replacement

**Status:** accepted, 2026-10-24

## Context

The repo already had two experiment records: `experiments.csv`, seven columns
appended by `utils.log_experiment`, and `experiments.md`, the reasoning. The
discipline works — by run nine it is the most valuable file in the repo. What
it cannot do is store an artifact, answer a query, or show two runs side by
side. And `src/log_lb.py` exists only because the leaderboard score arrives
after the run and has to be stitched back in by hand.

## Decision

Add MLflow **alongside** both, as a richer second sink. `utils.mlflow_run` is a
context manager beside `log_experiment` — no new module, because `utils.py`
already owns experiment logging.

`log_experiment` is untouched and **no column was added to `experiments.csv`**.
That is not caution, it is a bug: `log_experiment` writes the header only when
the file does not exist, so an eighth field would append 8-value rows under a
7-column header, and `blend.py:132` reads that file to check fold fingerprints.
The link runs the other way — the repo's `run_id` becomes an MLflow tag, so a
run and a CSV row are joinable by eye and by query.

Backend is SQLite (`mlruns/mlflow.db`) rather than the default file store, because the
model registry needs a database-backed store and the file store cannot register
models. A `.db` file in the repo is still zero infrastructure.

## Azure equivalent

**Azure Machine Learning.** This is most of why MLflow won: Azure ML speaks
MLflow natively, so the migration is one environment variable —

```bash
export MLFLOW_TRACKING_URI="azureml://<region>.api.azureml.ms/mlflow/v1.0/..."
```

— and **none of this code changes**. `src/config.py` already reads that variable.
Azure ML's own SDK would have been a lock-in; MLflow against Azure ML is not.

## Rejected

- **Replacing `experiments.csv`.** `blend.py` reads it, the fingerprint check
  depends on it, and a plain-text log survives a tool you stop paying for.
- **Weights & Biases / Neptune / Comet.** Better UIs, hosted, not open source at
  the core, and none is the native protocol of a major cloud's ML platform.
- **A `src/tracking.py` module.** One context manager is not a subsystem.

## Consequences

- A run now writes ~30MB of models into `mlruns/` as well. At ~2 min/run that is
  fine; it is worth knowing before a hyperparameter sweep.
- `import mlflow` cannot be at module scope: the README documents
  `from src.train import run` inside a Kaggle kernel, and
  `kernel-metadata.json` sets `enable_internet: false`. Hence the `_NoMlflow`
  shim, which exists for that one reason and is tested.
- **A secondary sink must not be able to break the primary one.** The first
  version let a runtime MLflow error -- a locked SQLite file, a full disk, a
  failed artifact upload -- abort `run()` after the arrays were saved but before
  `log_experiment` wrote the CSV row. Now a store that cannot start a run means
  untracked training, and every later call goes through `_SafeMlflow`, which
  reports the failure and carries on. Only training errors propagate.
- **The store is one directory.** The database sits beside its artifacts in
  `mlruns/`, and docker-compose mounts that directory into the MLflow container
  *at the same absolute path*, because MLflow records artifact locations as
  absolute host paths. The first compose file pointed the server at a different
  database entirely, so the UI never showed a host run. Its host port is 5001:
  macOS AirPlay Receiver holds 5000.
- **Artifacts follow the database, not the working directory.** MLflow's
  default puts a new experiment's artifacts in `./mlruns` relative to wherever
  the process runs. Every test fixture pointed the tracking URI at a tmp
  database, but the artifacts still landed in the repo -- about 1,100 test runs
  and 226MB in a day. `utils.use_experiment` now creates experiments with their
  artifact location derived from the tracking URI, so moving the database moves
  everything. Fixing it once at the source, rather than in each fixture, is what
  keeps the next fixture from reintroducing it.
- **Revisit when:** more than one person logs runs, at which point SQLite on a
  bind mount stops being adequate and the server needs Postgres.
