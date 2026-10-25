# Cheatsheet

Every command runs from the repo root. `uv run` uses `.venv` automatically —
you never need to activate anything.

## 1. Set up (once per machine)

```bash
brew install libomp        # macOS only: LightGBM/XGBoost need OpenMP
make setup                 # uv sync + pre-commit install
```

`pre-commit install` writes to `.git/hooks/`, which is local and never travels
with the repo — so every teammate runs `make hooks` once after cloning, or the
hooks silently don't exist for them.

Kaggle API credentials — download `kaggle.json` from your Kaggle account page
(Settings → API → Create New Token), then:

```bash
mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

## 2. Point the repo at your competition (once per competition)

Edit `src/config.py`: `COMPETITION` (the slug from the comp URL), `TARGET`,
`ID_COL`, `TASK`, `METRIC` — then `CV_SCHEME`, which you decide once and never
change.

```python
COMPETITION = "playground-series-s4e6"
TARGET      = "Response"
ID_COL      = "id"
TASK        = "binary"      # binary | multiclass | regression
METRIC      = "auc"         # auc | logloss | accuracy | f1 | rmse | mae | rmsle
CV_SCHEME   = "stratified"  # kfold | stratified | group | time | time_coldstart
```

## 3. Get the data

```bash
make data      # kaggle competitions download -c $COMPETITION -> data/raw/
```

Needs `~/.kaggle/kaggle.json` (`chmod 600`). Never committed — `.gitignore`
handles it.

## 4. Train

```bash
make baseline                                                  # lgbm, notes "baseline"
uv run python -m src.train --model lgbm --notes "added store aggregates"
uv run python -m src.train --model catboost --notes "blend partner"
uv run python -m src.train --model lgbm --params '{"num_leaves": 127}' --notes "wider trees"
```

Each run prints a per-fold and OOF score, then writes four things:

| Where | What |
|---|---|
| `submissions/sub_<run_id>_<cv>.csv` | the file you upload |
| `experiments/oof_<run_id>.npy` | out-of-fold predictions (blend fuel — keep them all) |
| `experiments/test_<run_id>.npy` | test predictions |
| `experiments.csv` | one log row: run_id, model, cv_score, folds, notes |

`run_id` looks like `0822_143052_lgbm` — month/day, time, model.

## 5. Submit

```bash
bash scripts/submit.sh submissions/sub_0822_143052_lgbm_0.87421.csv "baseline"
```

When Kaggle shows you the score, record it — don't hand-edit the csv:

```bash
uv run python -m src.log_lb 0822_143052_lgbm 0.87104
uv run python -m src.log_lb          # no args: CV-vs-LB table + rank correlation
```

Then write *why* into `experiments.md`. That file is the deliverable the CSV
can't be.

## 6. Blend (the last day, and every day after day two)

```bash
uv run python -m src.blend 0822_143052_lgbm 0823_101500_catboost
```

Prints each member's OOF score, the correlation matrix (low = good blend
material), the optimised weights, and writes a blended submission like any
other run. It refuses to blend runs whose fold fingerprints disagree.

## 7. Kaggle kernels (when you need their GPU)

```bash
make requirements                          # pip-installable export for kernels
bash scripts/sync_utils_to_kaggle.sh       # push src/ as a utility dataset
uv run kaggle kernels push -p kaggle       # push the notebook itself
```

First time only: `uv run kaggle datasets create -p kaggle/utils`.

## 8. Serve, monitor, version

```bash
make serve                 # API on :8000 with reload -- http://localhost:8000/docs
make up                    # api + mlflow + prometheus in docker
make down
make mlflow-ui             # tracking UI on :5001 without docker (5000 is macOS AirPlay)

make reference             # build the drift baseline from data/raw/train.csv (once)
make drift                 # live traffic vs that baseline -> reports/drift_<date>.html
make drift SINCE=7d
```

`experiments/` is versioned with DVC, because it is gitignored, 71MB, declared
immutable in CLAUDE.md, and otherwise protected by nothing but this disk.
**A training run writes new files into it, so `dvc status` goes dirty until:**

```bash
make dvc-save              # dvc add experiments && dvc push
```

`data/raw/` is deliberately *not* DVC-tracked -- `make data` re-downloads it
from Kaggle, so versioning it would duplicate 111MB to guard something one
command restores. Model bundles are not either: they ship as GitHub Release
assets (`make release-bundle RUN_ID=...`), and two channels for one artifact is
how they diverge.

## 9. Hygiene

```bash
make lint          # run every pre-commit hook without committing
make clean         # remove __pycache__ and .ipynb_checkpoints
uv add <pkg>       # add a dependency (updates pyproject.toml + uv.lock)
make hooks-update  # bump pinned hook versions, then verify
```

Never `pip install`. Never delete anything in `experiments/`.
