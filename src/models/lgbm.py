"""LightGBM wrapper with the interface train.py expects: fit / predict / importance."""

from __future__ import annotations

import numpy as np

from .. import config as C

_OBJECTIVE = {
    "binary": {"objective": "binary", "metric": "auc"},
    "multiclass": {"objective": "multiclass", "metric": "multi_logloss"},
    "regression": {"objective": "regression", "metric": "rmse"},
}[C.TASK]

DEFAULT_PARAMS = {
    **_OBJECTIVE,
    "learning_rate": 0.03,
    "num_leaves": 63,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_data_in_leaf": 50,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "seed": C.SEED,
    "n_jobs": -1,
}


def _norm_w(w, n):
    """Weights normalised to mean 1, or None. Keeps `min_sum_hessian_in_leaf`
    and the printed metric on the same scale as an unweighted fit."""
    if w is None:
        return None
    w = np.asarray(w, dtype=np.float64)
    assert len(w) == n, f"weight vector is {len(w)} long for {n} rows"
    m = float(w.mean())
    return w / m if m > 0 else None


class LGBMModel:
    name = "lgbm"

    def __init__(
        self,
        params: dict | None = None,
        num_boost_round: int = 5000,
        early_stopping: int = 200,
    ):
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.num_boost_round = num_boost_round
        self.early_stopping = early_stopping
        self.booster = None

    def fit(self, X_tr, y_tr, X_va, y_va, features=None, w_tr=None):
        """`w_tr` re-weights the fitting rows; the validation half stays uniform.

        Early stopping has to watch the population the model will be asked to
        predict. Weighting both halves lets a weighting scheme choose its own
        scoreboard, and then it always looks like an improvement.
        """
        import lightgbm as lgb

        features = list(features or X_tr.columns)
        cats = [c for c in features if str(X_tr[c].dtype) == "category"]
        w_tr = _norm_w(w_tr, len(y_tr))
        dtr = lgb.Dataset(X_tr, y_tr, weight=w_tr, feature_name=features, categorical_feature=cats)
        dva = lgb.Dataset(
            X_va, y_va, feature_name=features, categorical_feature=cats, reference=dtr
        )
        self.booster = lgb.train(
            self.params,
            dtr,
            num_boost_round=self.num_boost_round,
            valid_sets=[dva],
            callbacks=[
                *(
                    [lgb.early_stopping(self.early_stopping, verbose=False)]
                    if self.early_stopping
                    else []
                ),
                lgb.log_evaluation(200),
            ],
        )
        return self

    @property
    def best_iteration(self) -> int:
        """Trees actually kept. `best_iteration` is 0 when early stopping is off."""
        return self.booster.best_iteration or self.booster.num_trees()

    def predict(self, X) -> np.ndarray:
        return self.booster.predict(X, num_iteration=self.booster.best_iteration)

    def importance(self):
        import pandas as pd

        return pd.DataFrame(
            {
                "feature": self.booster.feature_name(),
                "gain": self.booster.feature_importance("gain"),
            }
        ).sort_values("gain", ascending=False)
