"""CatBoost with the interface train.py expects: fit / predict / importance.

Wanted as a decorrelated blend partner. Temper the expectation: a different
boosting library on identical features usually correlates 0.9+ with LightGBM,
and what earns real blend weight is being *structurally* different. Ordered
boosting and CatBoost's own categorical handling are the two places it can
genuinely diverge.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as C
from .lgbm import _norm_w

_LOSS = {
    "binary": {"loss_function": "Logloss", "eval_metric": "AUC"},
    "multiclass": {"loss_function": "MultiClass", "eval_metric": "MultiClass"},
    "regression": {"loss_function": "RMSE", "eval_metric": "RMSE"},
}[C.TASK]

DEFAULT_PARAMS = {
    **_LOSS,
    "learning_rate": 0.05,
    "depth": 8,
    "l2_leaf_reg": 3.0,
    "random_seed": C.SEED,
    "allow_writing_files": False,
    "verbose": False,
}

# CatBoost refuses NaN in a categorical outright, where LightGBM takes it as a
# NaN category. Missingness is often meaningful, so it gets its own level rather
# than an imputation.
NA_LEVEL = "__NA__"


class CatBoostModel:
    name = "catboost"

    def __init__(
        self,
        params: dict | None = None,
        num_boost_round: int = 3000,
        early_stopping: int = 200,
    ):
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.num_boost_round = num_boost_round
        self.early_stopping = early_stopping
        self.model = None
        self.features: list[str] = []
        self.cats: list[str] = []

    def _prep(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X[self.features].copy()
        for c in self.cats:
            X[c] = X[c].astype(object).where(X[c].notna(), NA_LEVEL).astype(str)
        return X

    def fit(self, X_tr, y_tr, X_va, y_va, features=None, w_tr=None):
        """`w_tr` re-weights the fitting rows; validation stays uniform, same
        contract as `models/lgbm.py`."""
        from catboost import CatBoostClassifier, CatBoostRegressor, Pool

        self.features = list(features or X_tr.columns)
        self.cats = [c for c in self.features if str(X_tr[c].dtype) == "category"]

        dtr = Pool(self._prep(X_tr), y_tr, weight=_norm_w(w_tr, len(y_tr)), cat_features=self.cats)
        dva = Pool(self._prep(X_va), y_va, cat_features=self.cats)
        cls = CatBoostRegressor if C.TASK == "regression" else CatBoostClassifier
        self.model = cls(
            iterations=self.num_boost_round,
            early_stopping_rounds=self.early_stopping,
            **self.params,
        )
        self.model.fit(dtr, eval_set=dva, use_best_model=True, verbose=200)
        return self

    def predict(self, X) -> np.ndarray:
        Xp = self._prep(X)
        if C.TASK == "binary":
            return self.model.predict_proba(Xp)[:, 1]
        return np.asarray(self.model.predict(Xp)).ravel()

    def importance(self):
        return pd.DataFrame(
            {"feature": self.features, "gain": self.model.get_feature_importance()}
        ).sort_values("gain", ascending=False)
