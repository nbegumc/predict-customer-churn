"""Single source of truth for paths, the target, and CV settings.

Fill in the COMPETITION block and the CV block in the first hour, then leave
them alone. Everything else in src/ reads from here, so nothing hardcodes a
path and no notebook disagrees with a script.
"""

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
    RAW = ROOT / "data" / "raw"
    WORK = ROOT

INTERIM = WORK / "data" / "interim"
PROCESSED = WORK / "data" / "processed"
SUBMISSIONS = WORK / "submissions"
EXPERIMENTS = WORK / "experiments"
FIGURES = WORK / "reports" / "figures"

TRAIN_FILE = RAW / "train.csv"
TEST_FILE = RAW / "test.csv"
SAMPLE_SUB_FILE = RAW / "sample_submission.csv"

for _p in (INTERIM, PROCESSED, SUBMISSIONS, EXPERIMENTS, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)
