"""Train one model across the fixed folds; write OOF, test preds, submission, log row.

    uv run python -m src.train --model lgbm --notes "baseline, raw features"

Every training run goes through here. A run that isn't logged didn't happen:
by day three you will not remember which of eleven variants produced the .npy
sitting in experiments/.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd

from . import config as C
from . import cv as CV
from .data import load, load_sample_submission
from .features import build_features, fold_features
from .models.catboost import CatBoostModel
from .models.lgbm import LGBMModel
from .models.linear import LinearModel
from .models.mlp import MLPModel
from .utils import (
    git_sha,
    log_experiment,
    mlflow_run,
    new_run_id,
    save_oof,
    seed_everything,
    timer,
)

MODELS = {
    "lgbm": LGBMModel,
    "catboost": CatBoostModel,
    "linear": LinearModel,
    "mlp": MLPModel,
}


def _write_manifest(
    bundle, run_id, model, train, features, cv_score, fold_scores, best_iters, resolved, has_refit
) -> None:
    """Everything needed to score one unseen row the way a CV row was scored.

    The model file already carries the feature order and the category lists, so
    this is a cross-check rather than the only source of truth — `src/predict.py`
    asserts the two agree. What lives *only* here is the blend weighting, the
    dtype regime and the per-fold stopping rounds.

    `strategies` maps a name to a weighted member list, so fold-averaging is
    five members at 0.2 and a full refit is one at 1.0 — the same code path at
    inference. Promoting the lgbm+catboost blend later means appending one
    member with its optimised weight, not rewriting the loader.
    """
    cats = {
        c: [str(v) for v in train[c].cat.categories]
        for c in features
        if str(train[c].dtype) == "category"
    }
    n = len(best_iters)
    strategies = {"fold_average": [{"path": f"fold{k}.txt", "weight": 1.0 / n} for k in range(n)]}
    if has_refit:
        strategies["full_refit"] = [{"path": "full.txt", "weight": 1.0}]

    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "model": model,
        "created": f"{datetime.now().astimezone():%Y-%m-%d %H:%M:%S%z}",
        "git_sha": git_sha(),
        "competition": C.COMPETITION,
        "target": C.TARGET,
        "metric": C.METRIC,
        "cv_scheme": C.CV_SCHEME,
        "n_folds": C.N_FOLDS,
        "seed": C.SEED,
        "fold_fingerprint": CV.fold_fingerprint(train),
        "cv_score": round(float(cv_score), 6),
        "fold_scores": [round(float(s), 6) for s in fold_scores],
        "best_iterations": best_iters,
        "features": list(features),
        "categories": cats,
        "dtypes": {c: str(train[c].dtype) for c in features},
        "params": resolved,
        "default_strategy": "fold_average",
        "strategies": strategies,
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(f"bundle -> {bundle}  ({', '.join(strategies)})")


def run(
    model: str = "lgbm",
    notes: str = "",
    params: dict | None = None,
    full_refit: bool = False,
    rounds: int | None = None,
    early_stopping: int | None = None,
):
    seed_everything()
    # Allocated before the fold loop, because each fold's model needs somewhere
    # to go. Side effect: the id's timestamp is now run-start, not run-end.
    run_id = new_run_id(model)
    bundle = C.MODEL_DIR / run_id

    with mlflow_run(run_id, model, notes) as ml:
        train, test = load()
        train, test, features = build_features(train, test)
        folds = CV.get_folds(train)
        # The tag that makes an MLflow run auditable against experiments.csv:
        # both carry the fingerprint, so a run produced under different folds
        # is visible as such without opening the OOF file.
        ml.set_tag("fold_fingerprint", CV.fold_fingerprint(train))
        ml.log_param("n_features", len(features))

        y_raw = train[C.TARGET].values
        y = np.log1p(np.clip(y_raw, 0, None)) if C.TARGET_TRANSFORM == "log1p" else y_raw

        def inverse(pred):
            return np.clip(np.expm1(pred), 0, None) if C.TARGET_TRANSFORM == "log1p" else pred

        # NaN marks "never validated" — time-based folds do not cover the earliest
        # rows, and those must be masked out rather than silently scored as zero.
        oof = np.full(len(train), np.nan)
        test_pred = np.zeros(len(test))
        importances = []
        best_iters: list[int] = []
        fold_scores: list[float] = []
        model_cls = MODELS[model]
        # Only the boosters take a round count; linear and mlp have none to set.
        ctor = {
            k: v
            for k, v in (("num_boost_round", rounds), ("early_stopping", early_stopping))
            if v is not None
        }

        for fold, (tr_pos, va_pos) in enumerate(folds):
            X_tr, X_va, X_te, feats = fold_features(train, test, tr_pos, va_pos, features)
            with timer(f"fold {fold}"):
                fitted = model_cls(params, **ctor).fit(
                    X_tr, y[tr_pos], X_va, y[va_pos], features=feats
                )
            oof[va_pos] = inverse(fitted.predict(X_va))
            test_pred += inverse(fitted.predict(X_te)) / len(folds)
            if hasattr(fitted, "importance"):
                importances.append(fitted.importance())
            # Same hasattr idiom as importance above. Until now every fitted model
            # was dropped here, so nothing the repo trained could ever be served.
            if hasattr(fitted, "save"):
                bundle.mkdir(parents=True, exist_ok=True)
                fitted.save(bundle / f"fold{fold}.txt")
                best_iters.append(int(fitted.best_iteration))
            fold_scores.append(CV.score(y_raw[va_pos], oof[va_pos]))
            print(f"  fold {fold} {C.METRIC}: {fold_scores[-1]:.6f}")
            ml.log_metric(f"fold_{C.METRIC}", fold_scores[-1], step=fold)
            if best_iters:
                ml.log_metric("fold_best_iteration", best_iters[-1], step=fold)

        scored = ~np.isnan(oof)
        cv_score = CV.score(y_raw[scored], oof[scored])
        print(f"\nOOF {C.METRIC}: {cv_score:.6f}  ({scored.sum():,}/{len(oof):,} rows validated)")

        # The *resolved* params, not just the overrides. save_oof only wrote a params
        # file when --params was passed, so the champion's hyperparameters were
        # recoverable only from whatever src/models/lgbm.py happened to say at HEAD.
        resolved = dict(getattr(fitted, "params", params or {}))
        save_oof(run_id, oof, test_pred, resolved)

        ml.log_params(resolved)
        ml.log_metric(f"oof_{C.METRIC}", cv_score)
        ml.log_metric("oof_rows_validated", int(scored.sum()))
        for name in (f"oof_{run_id}.npy", f"test_{run_id}.npy", f"params_{run_id}.json"):
            ml.log_artifact(str(C.EXPERIMENTS / name))

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
        ml.log_artifact(str(sub_path))

        if importances:
            imp = (
                pd.concat(importances)
                .groupby("feature")["gain"]
                .mean()
                .sort_values(ascending=False)
            )
            imp.to_csv(C.EXPERIMENTS / f"importance_{run_id}.csv")
            print(imp.head(20))
            ml.log_artifact(str(C.EXPERIMENTS / f"importance_{run_id}.csv"))

        if full_refit and best_iters:
            n = round(float(np.mean(best_iters)) * C.N_FOLDS / (C.N_FOLDS - 1))
            ml.log_param("refit_rounds", n)
            print(
                f"\nfull refit on all {len(train):,} rows at {n} rounds "
                f"(fold mean {np.mean(best_iters):.0f} x {C.N_FOLDS}/{C.N_FOLDS - 1})"
            )
            X_all = train[features]
            with timer("full refit"):
                # A 100%-of-train fit has no held-out rows, so early stopping is
                # impossible and the round count must be carried over from CV,
                # scaled because this fit sees 1/(k-1) more data than a fold did.
                # X_va is the training frame only to satisfy the signature — with
                # early_stopping=0 no callback watches it.
                refit = model_cls(params, num_boost_round=n, early_stopping=0).fit(
                    X_all, y, X_all, y, features=features
                )
            refit.save(bundle / "full.txt")

        if bundle.exists():
            _write_manifest(
                bundle,
                run_id,
                model,
                train,
                features,
                cv_score,
                fold_scores,
                best_iters,
                resolved,
                full_refit and bool(best_iters),
            )
            # The whole bundle, so an MLflow run is self-sufficient: the
            # boosters plus the manifest that says how to feed them.
            ml.log_artifacts(str(bundle), artifact_path="bundle")

        log_experiment(run_id, model, cv_score, notes=notes, folds=CV.fold_fingerprint(train))
        return run_id, cv_score


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lgbm", choices=list(MODELS))
    ap.add_argument("--notes", default="", help="what changed vs. the previous run")
    ap.add_argument("--params", default="", help="JSON overrides, e.g. '{\"num_leaves\": 127}'")
    ap.add_argument(
        "--full-refit",
        action="store_true",
        help="also fit once on 100%% of train and add it to the bundle as a second strategy",
    )
    ap.add_argument("--rounds", type=int, default=None, help="boosters only: fixed num_boost_round")
    ap.add_argument(
        "--no-early-stopping",
        action="store_const",
        const=0,
        dest="early_stopping",
        default=None,
        help="boosters only: train the full --rounds without watching the validation fold",
    )
    a = vars(ap.parse_args())
    raw = a.pop("params")
    a["params"] = json.loads(raw) if raw else None
    run(**a)
