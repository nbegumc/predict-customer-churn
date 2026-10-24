# Competition repo

Local-first Kaggle setup. Git is the source of truth; Kaggle is a compute and
submission target.

**Commands live in [`CHEATSHEET.md`](CHEATSHEET.md).** This file is the *why*:
the decisions that determine where you place, and the conventions that keep a
week of experiments comparable.

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/) — `uv sync`
builds `.venv` from `uv.lock`, so everyone on the team runs identical versions.
Prefix commands with `uv run`; there is nothing to activate. Commit `uv.lock`.
`requirements.txt` is generated (`make requirements`) for Kaggle kernels only —
never hand-edit it.

Pre-commit hooks guard the two mistakes that are expensive to undo: committing
data or credentials, and letting `uv.lock` drift from `pyproject.toml` so
teammates silently get different package versions. They also strip notebook
outputs and run ruff. `make setup` installs everything, `make hooks` re-installs
the hooks after a fresh clone, and `make lint` runs them all without committing.

## First hour

1. Set `COMPETITION`, `TARGET`, `ID_COL`, `TASK`, `METRIC` in `src/config.py`.
2. `make data`
3. Work through `notebooks/01_eda.ipynb`.
4. **Decide `CV_SCHEME` and never change it again.** Match how the organisers
   built the test set: random → `stratified`; future period → `time`; repeated
   entities → `group`; future period *and* unseen entities → `time_coldstart`.
   Getting this wrong is the usual reason teams drop 40 places when the private
   leaderboard reveals. Write the justification into `experiments.md`.
5. `make baseline` → first submission on the board.

## Daily loop

Train → submit → record the leaderboard score → write down what you concluded.
One change per run, always with `--notes` saying what changed, so the OOF score
difference means something.

Every run writes `experiments/oof_<run_id>.npy` and `experiments/test_<run_id>.npy`.
**Keep every one of them.** On the last day `src.blend` optimises weights over
those saved predictions, and a blend of diverse models is usually worth more
than any single extra idea. A deleted `.npy` is a model you have to retrain.

Two files hold the history, and only one is automatic:

- `experiments.csv` — numbers, appended by `src.train`. `src.log_lb` fills in
  the public LB score after you submit and tells you whether CV and LB agree.
- `experiments.md` — the reasoning. Hypothesis, result, keep or drop, and a
  **Dead ends** table. Nothing writes this for you, and by day three it is the
  most valuable file in the repo.

## What actually moves the score

In rough order of value per hour spent:

1. A CV scheme that mirrors the test split. Everything downstream is measured
   through it, so an error here is an error in every decision you make.
2. Features that encode domain structure — group aggregates, history, ratios.
3. A blend of *structurally different* models. Two boosting libraries on the
   same features correlate ~0.9 and blend to almost nothing; a different
   estimator shape is what earns weight.
4. Hyperparameter tuning. Last, and worth less than you think.

## From competition repo to served model

The competition half of this repo optimises for a leaderboard: prediction
arrays, a frozen CV scheme, one logged row per experiment. Production needs
nearly the opposite — a persisted, versioned model and a deterministic
single-record inference path. Both now exist side by side.

```bash
make serve     # the API on :8000
make up        # api + mlflow + prometheus
make drift     # live traffic vs the training distribution
```

- **[docs/architecture.md](docs/architecture.md)** — how a raw row becomes a
  prediction, and what is guaranteed at each hop.
- **[docs/serving.md](docs/serving.md)** — the API contract and the two tiers of
  validation.
- **[docs/deployment.md](docs/deployment.md)** — what deploying would involve.
  Nothing here is built; local compose is the choice.
- **[docs/adr/](docs/adr/README.md)** — twelve decisions, each with its Azure
  equivalent and what it would cost to change.

The single most important thing to understand before touching feature code is
[ADR 0012](docs/adr/0012-train-serve-parity.md): the training and serving paths
compute features through the same functions in the same order, and a test
asserts they agree **bitwise** on all 254,655 real test rows. Three specific
bugs live in the gap between those two paths, and all three are measured there.

## Working on Kaggle notebooks (when you need their GPU)

`src/` is pushed up as a Kaggle utility dataset and imported in the kernel:

```python
import sys; sys.path.append('/kaggle/input/<your>-utils')
from src.train import run
```

`bash scripts/sync_utils_to_kaggle.sh` pushes a new version. `src/config.py`
detects `/kaggle/input` and redirects the paths, so nothing else changes.
