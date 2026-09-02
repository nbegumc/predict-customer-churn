"""The fold splitter.

Write this on day one, verify it mirrors how the test set was constructed,
then DO NOT TOUCH IT. Every experiment must be scored on identical folds or
your comparisons are meaningless. Your CV is the only signal you can trust —
the public leaderboard is a small sample and the points come from the
*private* one.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, TimeSeriesSplit

from . import config as C


def _time_coldstart_folds(df: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding window in time, with a slice of entities hidden from training.

    Two things can make a test set hard, and a fold has to reproduce both: the
    validation rows are strictly in the future, and a fixed share of them belong
    to entities the model has never seen. Hiding COLD_FRACTION of the entities
    from the training part is what creates the second condition — without it, CV
    flatters any model that leans on entity history, and that flattery does not
    survive the private leaderboard.
    """
    assert C.TIME_COL, "set TIME_COL in config.py"
    assert C.GROUP_COL, "set GROUP_COL in config.py"

    t = pd.to_datetime(df[C.TIME_COL]).to_numpy()
    days = np.unique(t)
    groups = df[C.GROUP_COL].to_numpy()
    entities = np.unique(groups)
    n_cold = round(C.COLD_FRACTION * len(entities))

    folds = []
    for k in range(C.N_FOLDS):
        end = len(days) - (C.N_FOLDS - 1 - k) * C.VALID_STEP
        start = end - C.VALID_HORIZON
        if start < 1:
            raise ValueError(
                f"fold {k} has no training days: need VALID_HORIZON + "
                f"(N_FOLDS-1)*VALID_STEP < {len(days)} days, "
                f"got {C.VALID_HORIZON} + {(C.N_FOLDS - 1) * C.VALID_STEP}"
            )
        va_lo, va_hi = days[start], days[end - 1]

        va = (t >= va_lo) & (t <= va_hi)
        tr = t < va_lo
        # a different slice is hidden each fold, so no entity is always cold
        if n_cold:
            cold = np.random.default_rng(C.SEED + k).choice(entities, size=n_cold, replace=False)
            tr &= ~np.isin(groups, cold)

        folds.append((np.flatnonzero(tr), np.flatnonzero(va)))
    return folds


def get_folds(df: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return a list of (train_idx, valid_idx) according to config.CV_SCHEME."""
    y = df[C.TARGET] if C.TARGET in df.columns else None

    if C.CV_SCHEME == "stratified":
        splitter = StratifiedKFold(n_splits=C.N_FOLDS, shuffle=True, random_state=C.SEED)
        return list(splitter.split(df, y))

    if C.CV_SCHEME == "kfold":
        splitter = KFold(n_splits=C.N_FOLDS, shuffle=True, random_state=C.SEED)
        return list(splitter.split(df))

    if C.CV_SCHEME == "group":
        assert C.GROUP_COL, "set GROUP_COL in config.py"
        splitter = GroupKFold(n_splits=C.N_FOLDS)
        return list(splitter.split(df, y, groups=df[C.GROUP_COL]))

    if C.CV_SCHEME == "time_coldstart":
        return _time_coldstart_folds(df)

    if C.CV_SCHEME == "time":
        assert C.TIME_COL, "set TIME_COL in config.py"
        df_sorted = df.sort_values(C.TIME_COL)
        splitter = TimeSeriesSplit(n_splits=C.N_FOLDS)
        pos = {v: i for i, v in enumerate(df_sorted.index)}
        return [
            (
                np.array([pos[i] for i in df_sorted.index[tr]]),
                np.array([pos[i] for i in df_sorted.index[va]]),
            )
            for tr, va in splitter.split(df_sorted)
        ]

    raise ValueError(f"unknown CV_SCHEME: {C.CV_SCHEME}")


def fold_fingerprint(df: pd.DataFrame) -> str:
    """Short hash identifying the fold scheme a prediction was produced under.

    Hashes the *resulting split* — each fold's train/valid sizes and its edges —
    not just the config values, so a change in the splitter's logic is caught as
    well as a change in a parameter.

    Why this exists: an `oof_*.npy` is a bare float array of length len(train).
    Two of them produced under different fold schemes are indistinguishable by
    inspection, and blending them yields a plausible, meaningless number. The
    fingerprint travels in `experiments.csv` so `blend.py` can refuse.
    """
    parts = [C.CV_SCHEME, str(C.N_FOLDS), str(C.SEED)]
    t = pd.to_datetime(df[C.TIME_COL]).to_numpy() if C.TIME_COL else None
    for tr, va in get_folds(df):
        edge = f":{t[va].min()}:{t[va].max()}" if t is not None else ""
        parts.append(f"{len(tr)}:{len(va)}{edge}")
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:8]


def add_fold_column(df: pd.DataFrame) -> pd.DataFrame:
    """Materialise folds as a column so notebooks and scripts agree exactly.

    Only valid for schemes where the folds partition the rows. "time" and
    "time_coldstart" do not — use get_folds() directly there.
    """
    if C.CV_SCHEME in {"time", "time_coldstart"}:
        raise ValueError(f"{C.CV_SCHEME} folds do not partition the rows; use get_folds(df)")
    df = df.copy()
    df["fold"] = -1
    for k, (_, va) in enumerate(get_folds(df)):
        df.loc[df.index[va], "fold"] = k
    assert (df["fold"] >= 0).all(), "some rows never landed in a validation fold"
    return df


def score(y_true, y_pred) -> float:
    """Competition metric in one place, so nothing drifts."""
    m = C.METRIC.lower()
    if m == "auc":
        return roc_auc_score(y_true, y_pred)
    if m == "logloss":
        return log_loss(y_true, y_pred)
    if m == "rmsle":
        yt = np.clip(np.asarray(y_true, dtype=float), 0, None)
        yp = np.clip(np.asarray(y_pred, dtype=float), 0, None)
        return float(np.sqrt(np.mean((np.log1p(yp) - np.log1p(yt)) ** 2)))
    if m == "rmse":
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))
    if m == "mae":
        return mean_absolute_error(y_true, y_pred)
    if m == "f1":
        return f1_score(y_true, (np.asarray(y_pred) > 0.5).astype(int))
    if m == "accuracy":
        return accuracy_score(y_true, (np.asarray(y_pred) > 0.5).astype(int))
    raise ValueError(f"unknown METRIC: {C.METRIC}")


GREATER_IS_BETTER = C.METRIC.lower() in {"auc", "f1", "accuracy"}
