"""Train one model across the fixed folds; write OOF, test preds, submission, log row.

    uv run python -m src.train --model lgbm --notes "baseline, raw features"

Every training run goes through here. A run that isn't logged didn't happen:
by day three you will not remember which of eleven variants produced the .npy
sitting in experiments/.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from . import config as C
from . import cv as CV
from .data import load, load_sample_submission
from .features import build_features, fold_features
from .models.catboost import CatBoostModel
from .models.lgbm import LGBMModel
from .utils import log_experiment, new_run_id, save_oof, seed_everything, timer

MODELS = {
    "lgbm": LGBMModel,
    "catboost": CatBoostModel,
}


def run(model: str = "lgbm", notes: str = "", params: dict | None = None):
    seed_everything()

    train, test = load()
    train, test, features = build_features(train, test)
    folds = CV.get_folds(train)

    y_raw = train[C.TARGET].values
    y = np.log1p(np.clip(y_raw, 0, None)) if C.TARGET_TRANSFORM == "log1p" else y_raw

    def inverse(pred):
        return np.clip(np.expm1(pred), 0, None) if C.TARGET_TRANSFORM == "log1p" else pred

    # NaN marks "never validated" — time-based folds do not cover the earliest
    # rows, and those must be masked out rather than silently scored as zero.
    oof = np.full(len(train), np.nan)
    test_pred = np.zeros(len(test))
    importances = []
    model_cls = MODELS[model]

    for fold, (tr_pos, va_pos) in enumerate(folds):
        X_tr, X_va, X_te, feats = fold_features(train, test, tr_pos, va_pos, features)
        with timer(f"fold {fold}"):
            fitted = model_cls(params).fit(X_tr, y[tr_pos], X_va, y[va_pos], features=feats)
        oof[va_pos] = inverse(fitted.predict(X_va))
        test_pred += inverse(fitted.predict(X_te)) / len(folds)
        if hasattr(fitted, "importance"):
            importances.append(fitted.importance())
        print(f"  fold {fold} {C.METRIC}: {CV.score(y_raw[va_pos], oof[va_pos]):.6f}")

    scored = ~np.isnan(oof)
    cv_score = CV.score(y_raw[scored], oof[scored])
    print(f"\nOOF {C.METRIC}: {cv_score:.6f}  ({scored.sum():,}/{len(oof):,} rows validated)")

    run_id = new_run_id(model)
    save_oof(run_id, oof, test_pred, params)

    sub = load_sample_submission()
    # sample_submission is read raw, so its header keeps the competition's own
    # casing ("Churn") while C.TARGET is lower-cased ("churn"). Assigning by
    # C.TARGET would append a second column and leave the real one all-zeros —
    # a file that uploads cleanly and scores 0.5. Write the column the file has.
    (sub_target,) = [c for c in sub.columns if c != C.ID_COL]
    sub[sub_target] = test_pred
    sub_path = C.SUBMISSIONS / f"sub_{run_id}_{cv_score:.5f}.csv"
    sub.to_csv(sub_path, index=False)
    print(f"submission -> {sub_path}")

    if importances:
        imp = pd.concat(importances).groupby("feature")["gain"].mean().sort_values(ascending=False)
        imp.to_csv(C.EXPERIMENTS / f"importance_{run_id}.csv")
        print(imp.head(20))

    log_experiment(run_id, model, cv_score, notes=notes, folds=CV.fold_fingerprint(train))
    return run_id, cv_score


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lgbm", choices=list(MODELS))
    ap.add_argument("--notes", default="", help="what changed vs. the previous run")
    ap.add_argument("--params", default="", help="JSON overrides, e.g. '{\"num_leaves\": 127}'")
    a = vars(ap.parse_args())
    raw = a.pop("params")
    a["params"] = json.loads(raw) if raw else None
    run(**a)
