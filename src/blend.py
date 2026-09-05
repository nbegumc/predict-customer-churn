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


N_PAIRS = 2_000_000


def _correct_by_pair(y, oofs, n_pairs: int = N_PAIRS) -> np.ndarray:
    """(n_pairs, n_runs) bool: did each run order this (positive, negative) pair right?

    One sample shared by both diagnostics below, so the disagreement matrix and
    the dissent table can never quote slightly different numbers for the same
    comparison.
    """
    rng = np.random.default_rng(C.SEED)
    i = rng.choice(np.flatnonzero(y == 1), n_pairs)
    j = rng.choice(np.flatnonzero(y == 0), n_pairs)
    return oofs[i] > oofs[j]


def _rank_disagreement(correct, run_ids) -> pd.DataFrame:
    """% of (positive, negative) pairs that two runs order differently.

    AUC is the fraction of positive/negative pairs put in the right order, so
    the diagnostic that matches it is how often two runs order such a pair
    differently. Prediction correlation does not match it: it is dominated by
    the easy rows where both models agree trivially. On this data LightGBM and
    CatBoost correlated 0.997 and still disagreed on 1.10% of pairs — and that
    ~1% is the entire raw material the blend traded on. See experiments.md,
    "0.997 correlation did not mean 0.997 of the same answers".

    Necessary, and on its own badly misleading — see `_dissent_table`.
    """
    n = len(run_ids)
    m = [[100 * (correct[:, a] != correct[:, b]).mean() for b in range(n)] for a in range(n)]
    return pd.DataFrame(m, index=run_ids, columns=run_ids)


def _dissent_table(y, oofs, correct, run_ids) -> tuple[str, pd.DataFrame]:
    """Disagreement with the best run, split by *who is right* when they differ.

    Disagreement volume read alone is a trap: it **rises as a candidate gets
    worse**, so it recommends the weakest model in the list. Measured here:
    CatBoost dissented from LightGBM on 1.10% of pairs and was worth +0.000176,
    while a logistic regression dissented on 3.65% — three times the raw
    material — and was worth +0.00000007.

    What decides whether a blend can use the dissent is how evenly it splits,
    and that split is forced by an identity rather than being an empirical
    tendency. AUC is the probability of ordering a pair correctly, so

        AUC_a - AUC_b = P(a right, b wrong) - P(b right, a wrong)

    and both terms sum to the disagreement d, giving

        P(best right | they disagree) = 1/2 + (AUC_best - AUC_other) / 2d

    A model's solo deficit **is** its net loss on the pairs where it dissents;
    there is nowhere else for a deficit to live. So near 50% means the dissent
    is a coin flip and averaging can recover something from it; well above 50%
    means the candidate is predictably wrong where it differs, and no weight
    will help. A blend member has to be different *and* competitive, and being
    different is the cheap half.
    """
    scores = [CV.score(y, c) for c in oofs.T]
    best = int(np.argmax(scores))
    rows = []
    for k, r in enumerate(run_ids):
        if k == best:
            continue
        best_only = (correct[:, best] & ~correct[:, k]).mean()
        cand_only = (~correct[:, best] & correct[:, k]).mean()
        d = best_only + cand_only
        gap = scores[best] - scores[k]
        rows.append(
            {
                "run": r,
                "solo": scores[k],
                "gap": gap,
                "disagree %": 100 * d,
                "best right %": 100 * best_only / d if d else float("nan"),
                "identity %": 100 * (0.5 + gap / (2 * d)) if d else float("nan"),
            }
        )
    return run_ids[best], pd.DataFrame(rows)


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
        correct = _correct_by_pair(y, oofs)
        print(f"\nranking disagreement, % of {N_PAIRS:,} sampled pairs ordered differently:")
        print(_rank_disagreement(correct, run_ids).round(2).to_string())

        # Volume alone recommends the worst candidate in the list, so never
        # print it without the split that says who is right when they differ.
        best, tbl = _dissent_table(y, oofs, correct, run_ids)
        if not tbl.empty:
            print(f"\nblend candidacy, against the best run ({best}):")
            print(
                tbl.to_string(
                    index=False,
                    formatters={
                        "solo": lambda v: f"{v:.6f}",
                        "gap": lambda v: f"{v:+.6f}",
                        "disagree %": lambda v: f"{v:.2f}",
                        "best right %": lambda v: f"{v:.1f}",
                        "identity %": lambda v: f"{v:.1f}",
                    },
                )
            )
            print(
                "  50% = the dissent is a coin flip, so a blend can use it.\n"
                "  Well above 50% = the candidate is predictably wrong where it differs."
            )

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
