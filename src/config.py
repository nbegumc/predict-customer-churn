"""Single source of truth for paths, the target, and CV settings.

Fill in the COMPETITION block and the CV block in the first hour, then leave
them alone. Everything else in src/ reads from here, so nothing hardcodes a
path and no notebook disagrees with a script.
"""

import contextlib
import os
from pathlib import Path

# --- competition -----------------------------------------------------------
COMPETITION = "playground-series-s6e3"
TARGET = "churn"  # lower-case: basic_clean lower-cases every header
ID_COL = "id"
TASK = "binary"  # "binary" | "multiclass" | "regression"
METRIC = "auc"  # "auc" | "logloss" | "accuracy" | "f1" | "rmse" | "mae" | "rmsle"

# --- target transform ------------------------------------------------------
# RMSLE is RMSE in log1p space, so with METRIC="rmsle" train on log1p(target)
# and the model optimises the competition metric directly. Inverted with
# expm1 + clip(0) in train.py.
TARGET_TRANSFORM = "none"  # "none" | "log1p"

# --- reproducibility -------------------------------------------------------
SEED = 42

# --- cross-validation ------------------------------------------------------
# Decide this ONCE, on day one, to mirror how the test set was built. This is
# the single highest-leverage decision in the repo: get it wrong and every
# comparison you make for the next five days is measuring the wrong thing.
#   "kfold"          - plain random folds
#   "stratified"     - classification, preserves class balance
#   "group"          - rows share an entity (user/store/patient) across folds
#   "time"           - test set is in the future -> expanding window
#   "time_coldstart" - expanding window in time, AND a slice of entities held
#                      out of training entirely, so validation reproduces the
#                      share of test rows whose entity has no history
# DECIDED 2026-09-02 by notebooks/01_eda.ipynb — FROZEN. The value is unchanged
# from the placeholder, but it is now measured rather than assumed: id is unique
# and contiguous in both frames, there is no date column and id order carries no
# time signal, no feature vector repeats (so there is no entity to group or hold
# out cold), and adversarial validation of train vs. test scores AUC 0.5117 — the
# two frames are the same distribution. The test set is a random 30% slice, so
# stratified folds mirror it. Full numbers in experiments.md -> Validation
# decision. Frozen from here: every oof_*.npy carries the fold fingerprint
# (13feb807) and blend.py refuses on a mismatch.
CV_SCHEME = "stratified"
N_FOLDS = 5
GROUP_COL = ""  # required when CV_SCHEME == "group" | "time_coldstart"
TIME_COL = ""  # required when CV_SCHEME == "time" | "time_coldstart"

# --- time_coldstart parameters ---------------------------------------------
# Measure these against the real test set in the EDA notebook; do not guess.
# VALID_HORIZON should equal the real forecast horizon in days, and
# COLD_FRACTION is tuned so the mean cold share across folds matches the share
# of test rows whose entity never appears in train. Note the fleet usually
# grows on its own, so the cold share is already non-zero at COLD_FRACTION=0 —
# calibrate, don't set it to the target share.
VALID_HORIZON = 90  # validation window length, in days = the real horizon
VALID_STEP = 45  # how far each fold's window slides back
COLD_FRACTION = 0.0  # share of entities hidden from each fold's training part

# --- paths -----------------------------------------------------------------
# On Kaggle, /kaggle/input is read-only, so we detect and redirect.
ON_KAGGLE = Path("/kaggle/input").exists()

ROOT = Path(__file__).resolve().parents[1]

if ON_KAGGLE:
    RAW = Path("/kaggle/input") / COMPETITION
    WORK = Path("/kaggle/working")
else:
    # CHURN_RAW_DIR points training at a different copy of the data without
    # touching data/raw, which CLAUDE.md declares immutable. CI uses it to train
    # a throwaway model on tests/fixtures/ so it can build and smoke-test the
    # image without the 77MB dataset, which is gitignored and never in a clone.
    RAW = Path(os.getenv("CHURN_RAW_DIR") or ROOT / "data" / "raw")
    # CHURN_WORK_DIR redirects everything a run *writes* — experiments.csv,
    # models/, submissions/. Without it, training on fixture data appends a
    # meaningless row to the real experiment log. CI deliberately leaves this
    # unset, because the Dockerfile copies models/<run_id> relative to the
    # build context and the bundle therefore has to land under the repo root.
    WORK = Path(os.getenv("CHURN_WORK_DIR") or ROOT)

INTERIM = WORK / "data" / "interim"
PROCESSED = WORK / "data" / "processed"
SUBMISSIONS = WORK / "submissions"
EXPERIMENTS = WORK / "experiments"
FIGURES = WORK / "reports" / "figures"

TRAIN_FILE = RAW / "train.csv"
TEST_FILE = RAW / "test.csv"
SAMPLE_SUB_FILE = RAW / "sample_submission.csv"

# --- model artifacts, tracking, serving ------------------------------------
# A trained bundle is models/<run_id>/: one booster per fold plus a manifest
# naming the feature order, the category lists and the blend weights. ~30MB,
# so it is gitignored and distributed as a GitHub Release, not committed.
MODEL_DIR = WORK / "models"

# SQLite rather than MLflow's default ./mlruns file store, because the model
# registry needs a database-backed store and a file in the repo is still zero
# infrastructure. Point MLFLOW_TRACKING_URI at an Azure ML workspace and the
# same logging code works unchanged — that env var is the whole migration.
# The database sits inside mlruns/ beside the artifacts it points to, so the
# whole store is one directory. docker-compose mounts that directory into the
# MLflow container at the same absolute path, because MLflow records artifact
# locations as absolute paths -- anything else gives the UI a database whose
# artifact links do not resolve.
MLRUNS = WORK / "mlruns"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{MLRUNS / 'mlflow.db'}")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", COMPETITION)

# Risk bands for the API, as probability thresholds. These are a *business*
# decision, not a model property: they come from the lift curve in
# notebooks/99_final_writeup.ipynb, where the top 10% of the ranked call list
# runs at 3.7x the base rate and the top 30% reaches 82% of churners at 2.7x.
# Measured on what is actually served -- the champion's fold_average scores
# over the 254,655 test rows: p90 = 0.694, p70 = 0.295, which put exactly 10.00%
# and 30.01% of customers in the high and medium-or-high bands. An earlier
# version used OOF quantiles (0.711 / 0.311); OOF is one fold model per row and
# more dispersed than the five-model mean, so only 9.18% reached "high".
# A bundle carrying its own score_quantiles overrides these.
RISK_BAND_HIGH = 0.694
RISK_BAND_MEDIUM = 0.295

# One JSON line per served prediction. The input src.monitor reads to build a
# drift report; append-only, and never on the request's critical path.
PRED_LOG = WORK / "logs" / "predictions.jsonl"

# mkdir(2) takes the filesystem write lock before it checks for EEXIST, so on a
# read-only mount this raises EROFS even when the directory is already there.
# Every module in src/ imports this one, so an unguarded failure here would take
# down the API too. Nothing on the inference path writes to these directories.
for _p in (
    INTERIM,
    PROCESSED,
    SUBMISSIONS,
    EXPERIMENTS,
    FIGURES,
    MODEL_DIR,
    MLRUNS,
    PRED_LOG.parent,
):
    with contextlib.suppress(OSError):
        _p.mkdir(parents=True, exist_ok=True)
