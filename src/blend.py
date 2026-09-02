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
    print("\ncorrelation of OOF predictions (low = good blend material):")
    print(pd.DataFrame(oofs, columns=run_ids).corr().round(3).to_string())

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
    sub[C.TARGET] = test_pred
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
