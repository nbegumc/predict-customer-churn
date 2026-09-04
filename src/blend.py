"""Blend saved runs using their OOF predictions.

    uv run python -m src.blend 0612_1430_lgbm 0612_1655_catboost

Weights are optimised on OOF, so this is honest as long as your folds never
changed. Usually worth more than any single extra model.

Pass runs that are *distinct where it counts*, not every saved run. Two runs
that share most of their predictions hand the optimiser duplicate columns: the
weights split arbitrarily between them and the story gets harder to tell for no
score.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import config as C
from . import cv as CV
from .data import load, load_sample_submission
from .utils import LOG_FILE, log_experiment, new_run_id, save_oof


def _fit_weights(y, oofs):
    """Weights on the simplex minimising the competition metric."""
    sign = -1 if CV.GREATER_IS_BETTER else 1

    def loss(w):
        w = np.abs(w) / np.abs(w).sum()
        return sign * CV.score(y, oofs @ w)

    res = minimize(loss, x0=np.ones(oofs.shape[1]) / oofs.shape[1], method="Nelder-Mead")
    return np.abs(res.x) / np.abs(res.x).sum()


def _rank_disagreement(y, oofs, run_ids, n_pairs: int = 200_000) -> pd.DataFrame:
    """% of (positive, negative) pairs that two runs order differently.

    AUC is the fraction of positive/negative pairs put in the right order, so
    the diagnostic that matches it is how often two runs order such a pair
    differently. Prediction correlation does not match it: it is dominated by
    the easy rows where both models agree trivially. On this data LightGBM and
    CatBoost correlated 0.997 and still disagreed on 1.09% of pairs — and that
    ~1% is the entire raw material the blend traded on. See experiments.md,
    "0.997 correlation did not mean 0.997 of the same answers".
    """
    rng = np.random.default_rng(C.SEED)
    i = rng.choice(np.flatnonzero(y == 1), n_pairs)
    j = rng.choice(np.flatnonzero(y == 0), n_pairs)
    correct = oofs[i] > oofs[j]  # (n_pairs, n_runs)
    n = len(run_ids)
    m = [[100 * (correct[:, a] != correct[:, b]).mean() for b in range(n)] for a in range(n)]
    return pd.DataFrame(m, index=run_ids, columns=run_ids)


def _assert_same_folds(run_ids: list[str], train) -> str:
    """Refuse to blend predictions produced under different fold schemes.

    An `oof_*.npy` is a bare float array; one made under the original scheme and
    one made after you "just tweaked" the splitter are the same shape and dtype,
    and averaging them gives a plausible number that means nothing. The
    fingerprint in experiments.csv is the only thing that can tell them apart.
    """
    now = CV.fold_fingerprint(train)
    log = pd.read_csv(LOG_FILE).set_index("run_id")
    stale = {
        r: log.at[r, "folds"]
        for r in run_ids
        if r in log.index and pd.notna(log.at[r, "folds"]) and log.at[r, "folds"] != now
    }
    if stale:
        raise SystemExit(
            f"fold mismatch — current scheme is {now!r} but these runs were scored "
            f"under another: {stale}. Their oof_*.npy are not comparable with "
            "today's. Re-run them, or blend them under the scheme they came from."
        )
    unknown = [r for r in run_ids if r not in log.index or pd.isna(log.at[r, "folds"])]
    if unknown:
        print(f"WARNING: no fold fingerprint recorded for {unknown} — cannot verify comparability")
    return now


def blend(run_ids: list[str]):
    train, _ = load()
    _assert_same_folds(run_ids, train)
    y = train[C.TARGET].values

    oofs = np.column_stack([np.load(C.EXPERIMENTS / f"oof_{r}.npy") for r in run_ids])
    tests = np.column_stack([np.load(C.EXPERIMENTS / f"test_{r}.npy") for r in run_ids])

    # Time-based folds leave the earliest rows unvalidated (NaN). Weights are
    # optimised only on rows every run actually predicted.
    mask = ~np.isnan(oofs).any(axis=1)
    if not mask.all():
        print(f"scoring on {mask.sum():,}/{len(mask):,} rows validated by every run")
    y, oofs = y[mask], oofs[mask]

    for r, col in zip(run_ids, oofs.T):
        print(f"{r}: {CV.score(y, col):.6f}")
    print("\ncorrelation of OOF predictions:")
    print(pd.DataFrame(oofs, columns=run_ids).corr().round(3).to_string())
    # For a ranking metric the correlation is the wrong instrument — it reads
    # near-1 for models that still order ~1% of pairs differently, which is
    # exactly what a blend lives on. Print the matching diagnostic too.
    if C.METRIC.lower() == "auc":
        print(
            "\nranking disagreement, % of pairs ordered differently (high = good blend material):"
        )
        print(_rank_disagreement(y, oofs, run_ids).round(2).to_string())

    w = _fit_weights(y, oofs)
    blended = CV.score(y, oofs @ w)
    print("\nweights:")
    for r, wi in zip(run_ids, w):
        print(f"  {wi:.4f}  {r}")
    best = (
        max(CV.score(y, c) for c in oofs.T)
        if CV.GREATER_IS_BETTER
        else min(CV.score(y, c) for c in oofs.T)
    )
    print(f"\nblend {C.METRIC}: {blended:.6f}   (best single: {best:.6f})")

    run_id = new_run_id("blend")
    oof_full = np.full(len(train), np.nan)
    oof_full[mask] = oofs @ w
    test_pred = tests @ w
    save_oof(run_id, oof_full, test_pred, {"members": run_ids, "weights": w.tolist()})

    sub = load_sample_submission()
    # same reason as in train.py: sample_submission is read raw and keeps the
    # competition's casing ("Churn"), while C.TARGET is lower-cased ("churn").
    (sub_target,) = [c for c in sub.columns if c != C.ID_COL]
    sub[sub_target] = test_pred
    sub_path = C.SUBMISSIONS / f"sub_{run_id}_{blended:.5f}.csv"
    sub.to_csv(sub_path, index=False)
    print(f"submission -> {sub_path}")

    log_experiment(
        run_id,
        "blend",
        blended,
        notes="blend of " + ", ".join(run_ids),
        folds=CV.fold_fingerprint(train),
    )
    return run_id, blended


if __name__ == "__main__":
    ids = sys.argv[1:]
    if len(ids) < 2:
        sys.exit("usage: python -m src.blend <run_id> <run_id> [...]")
    blend(ids)
