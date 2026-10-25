"""The frozen-CV invariant, as a test.

`CLAUDE.md` says `CV_SCHEME`, `N_FOLDS` and `src/cv.py` must not change,
because every saved `oof_*.npy` carries the fold fingerprint and `blend.py`
refuses to average files produced under different folds. Until now that
invariant was protected by a comment. This file makes CI enforce it.

The useful property that makes this cheap: `fold_fingerprint` hashes
`CV_SCHEME|N_FOLDS|SEED` plus each fold's train/valid *sizes*, and with
`TIME_COL` empty there are no date edges in the hash. StratifiedKFold's fold
sizes depend only on the row count and the class counts — so the real
fingerprint reproduces from a frame holding nothing but a label column, and
these tests need zero bytes of the real dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config as C
from src import cv as CV

# The fingerprint every existing oof_*.npy and every experiments.csv row was
# produced under. Decided 2026-09-02 in notebooks/01_eda.ipynb and frozen.
FROZEN_FINGERPRINT = "13feb807"

# Shape of the real training frame, from experiments.md -> Validation decision.
REAL_N_ROWS = 594_194
REAL_N_POSITIVE = 133_817


def label_only_frame(n_rows: int = REAL_N_ROWS, n_positive: int = REAL_N_POSITIVE):
    """A frame with the real row count and class balance, and nothing else."""
    y = np.zeros(n_rows, dtype="int8")
    y[:n_positive] = 1
    return pd.DataFrame({C.TARGET: y})


def test_cv_scheme_is_frozen():
    """The two config values CLAUDE.md names as unchangeable."""
    assert C.CV_SCHEME == "stratified"
    assert C.N_FOLDS == 5
    assert C.SEED == 42
    assert C.GROUP_COL == ""
    assert C.TIME_COL == ""


def test_fold_fingerprint_matches_every_saved_run():
    """The whole point. If this fails, every saved OOF file is unblendable."""
    assert CV.fold_fingerprint(label_only_frame()) == FROZEN_FINGERPRINT


@pytest.mark.realdata
def test_fingerprint_on_real_data_matches_the_synthetic_one(real_train):
    """Proves the label-only shortcut above is not lying."""
    y = (real_train["Churn"] == "Yes").astype("int8")
    assert CV.fold_fingerprint(pd.DataFrame({C.TARGET: y.to_numpy()})) == FROZEN_FINGERPRINT


def test_fingerprint_is_insensitive_to_small_class_shifts():
    """Locks the *scheme*, not the data: a few labels either way must not move it.

    StratifiedKFold's fold sizes barely move with a handful of labels, which is
    what makes this test about the fold scheme rather than about the dataset.
    """
    for delta in (-10, -1, 1, 10):
        got = CV.fold_fingerprint(label_only_frame(n_positive=REAL_N_POSITIVE + delta))
        assert got == FROZEN_FINGERPRINT, f"shifted by {delta} and the hash moved"


def test_fingerprint_does_change_when_the_scheme_does(monkeypatch):
    """The other half: the hash has to be sensitive to what it is guarding."""
    df = label_only_frame()
    monkeypatch.setattr(C, "N_FOLDS", 10)
    assert CV.fold_fingerprint(df) != FROZEN_FINGERPRINT


def test_fingerprint_changes_with_the_seed(monkeypatch):
    df = label_only_frame()
    monkeypatch.setattr(C, "SEED", 1)
    assert CV.fold_fingerprint(df) != FROZEN_FINGERPRINT


def test_folds_partition_every_row(synthetic_raw):
    """`stratified` folds must cover each row exactly once.

    experiments.md -> Consequences depends on this: it is why the OOF array has
    no NaNs and why `cv.add_fold_column` is valid for this competition.
    """
    df = synthetic_raw.rename(columns={"Churn": C.TARGET})
    df[C.TARGET] = (df[C.TARGET] == "Yes").astype("int8")
    folds = CV.get_folds(df)
    assert len(folds) == C.N_FOLDS
    seen = np.concatenate([va for _, va in folds])
    assert sorted(seen) == list(range(len(df)))


def test_score_is_auc_and_greater_is_better():
    """`cv.score` is the one place the competition metric lives."""
    assert C.METRIC == "auc"
    assert CV.GREATER_IS_BETTER is True
    y = np.array([0, 0, 1, 1])
    assert CV.score(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert CV.score(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
