"""Feature engineering.

Rule of thumb: anything that *learns* from the target must be fitted inside the
fold loop on the training part only. Stateless transforms (ratios, date parts,
string parsing) can be done once on the full frame.

Two entry points:
  build_features(train, test)  -> stateless, run once, returns the base columns
  fold_features(...)           -> fitted per fold, where leakage would happen
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


# --- stateless: safe to run once on train+test ------------------------------
def add_date_parts(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    d = pd.to_datetime(df[col], errors="coerce")
    df[f"{col}_year"] = d.dt.year
    df[f"{col}_month"] = d.dt.month
    df[f"{col}_day"] = d.dt.day
    df[f"{col}_dow"] = d.dt.dayofweek
    df[f"{col}_dayofyear"] = d.dt.dayofyear
    df[f"{col}_is_weekend"] = (d.dt.dayofweek >= 5).astype(np.int8)
    return df


def add_aggregates(df: pd.DataFrame, by: str, cols: list[str]) -> pd.DataFrame:
    """Group statistics — usually the highest-value tabular feature family.

    Target-free: these aggregate *features*, not the target. Aggregating the
    target by a group is target encoding and belongs in fold_features().
    """
    df = df.copy()
    for col in cols:
        g = df.groupby(by)[col]
        df[f"{col}_mean_by_{by}"] = g.transform("mean")
        df[f"{col}_std_by_{by}"] = g.transform("std")
        df[f"{col}_diff_from_{by}_mean"] = df[col] - df[f"{col}_mean_by_{by}"]
    return df


def as_categorical(
    train: pd.DataFrame, test: pd.DataFrame, cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shared category dtype across train and test.

    Categories must span BOTH frames. With train-only categories, a value that
    appears solely in test falls out of the dtype and becomes NaN, so the model
    takes a different branch for it than for the equivalent validation row — and
    CV stops measuring what test will do.
    """
    train, test = train.copy(), test.copy()
    for col in cols:
        cats = pd.Index(pd.unique(pd.concat([train[col], test[col]]).dropna()))
        train[col] = pd.Categorical(train[col], categories=cats)
        test[col] = pd.Categorical(test[col], categories=cats)
    return train, test


# --- competition-specific: telco churn --------------------------------------
def add_telco_features(df: pd.DataFrame) -> pd.DataFrame:
    """Structure the raw columns do not already state.

    Picked in notebooks/02_baseline.ipynb to be non-redundant with `contract`,
    which already carries most of what `tenure` carried on its own. A rescaling
    of an existing column adds nothing to a tree; a ratio between two of them
    can.

    Both ratios are safe to divide: tenure >= 1 and monthlycharges >= 18.25 in
    train and test alike, and there are no missing cells in either frame.
    """
    df = df.copy()
    # what the bill *should* have totalled if the current rate had always applied
    implied = df.tenure * df.monthlycharges
    df["charges_ratio"] = (df.totalcharges / implied).astype(np.float32)
    df["avg_monthly_spend"] = (df.totalcharges / df.tenure).astype(np.float32)
    return df


def build_features(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Stateless features + the list of columns the model is allowed to see.

    Returns (train, test, features). Edit this for the competition at hand; the
    only thing train.py needs is that the third element names real columns
    present in both frames.
    """
    if C.TIME_COL:
        train = add_date_parts(train, C.TIME_COL)
        test = add_date_parts(test, C.TIME_COL)

    train, test = add_telco_features(train), add_telco_features(test)

    drop = {C.TARGET, C.ID_COL, C.TIME_COL} | (set(train.columns) - set(test.columns))
    features = [c for c in test.columns if c not in drop]

    # Anything non-numeric becomes a shared category. Checked by kind rather
    # than against `object`, because pandas infers string columns as the `str`
    # dtype now and an `== object` test silently misses them — LightGBM then
    # raises on the raw strings.
    txt = [
        c
        for c in features
        if not (
            pd.api.types.is_numeric_dtype(train[c])
            or pd.api.types.is_bool_dtype(train[c])
            or pd.api.types.is_datetime64_any_dtype(train[c])
        )
    ]
    if txt:
        train, test = as_categorical(train, test, txt)

    print(f"{len(features)} features")
    return train, test, features


# --- fitted per fold: everything that could leak ----------------------------
def fold_features(train, test, tr_pos, va_pos, features):
    """Hook for features fitted on the training part of one fold only.

    Returns (X_tr, X_va, X_test, features). The default is a pass-through, so
    train.py works out of the box. Fill it in when you need:

    * Target encoding — the mean target per category, fitted on tr_pos alone.
      Fitting it on the full frame is the classic silent leak: CV improves, the
      leaderboard does not.

    * Lag / rolling history in a time-series competition — and if you add them,
      MATCH THE STALENESS. At inference the newest usable data is the last
      training day, so a test row 90 days into the horizon is predicted from
      90-day-old features while the first one is predicted from yesterday's.
      Give training rows the same staleness distribution (a random 1..horizon
      day lag on the origin the features are read at), or the model learns to
      trust a freshness it will not have at test time — the failure mode that
      makes rolling features look brilliant in CV and collapse on the LB.
    """
    X = train[features]
    return X.iloc[tr_pos], X.iloc[va_pos], test[features], features
