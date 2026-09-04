"""Regularised logistic regression on one-hot features.

The README's idea 3 asks for a *structurally different* model, and this is the
cheapest thing that qualifies. A booster is an interaction machine; a linear
model in the one-hot basis is purely additive — it cannot express a single
interaction the trees live on. Two boosters on the same 21 features disagreed
on 1.09% of ranked pairs (`experiments.md`); the question this run exists to
answer is whether a different *shape* disagrees on materially more.

Solo score is not the yardstick. This is expected to lose to LightGBM outright
and still be worth keeping, on the same reasoning that kept CatBoost.

Preprocessing lives here rather than in `features.py` because it is
model-specific: LightGBM wants native categoricals and a one-hot expansion
would hurt it, exactly as CatBoost's NA level is CatBoost's own business. It is
fitted inside `fit` on the training half only — the scaler's mean and scale are
fitted on data, and the repo's rule is that anything fitted on data is fitted
inside the fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as C
from .lgbm import _norm_w

# L2 is sklearn's default (`l1_ratio=0.0`) and `penalty=` is deprecated from
# 1.8, so the penalty is left implicit. Verified bit-identical coefficients
# either way before the explicit argument was dropped, so the OOF from
# 0905_015118_linear stands.
DEFAULT_PARAMS = {
    "C": 1.0,
    "solver": "lbfgs",
    "max_iter": 1000,
}


class LinearModel:
    name = "linear"

    def __init__(self, params: dict | None = None):
        if C.TASK != "binary":
            raise NotImplementedError(
                f"LinearModel is logistic regression and only covers TASK='binary'; "
                f"config says {C.TASK!r}. Ridge/softmax would be a different model, "
                "not a parameter of this one."
            )
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.pipe = None
        self.features: list[str] = []
        self.cats: list[str] = []
        self.nums: list[str] = []

    def fit(self, X_tr, y_tr, X_va, y_va, features=None, w_tr=None):
        """`w_tr` re-weights the fitting rows, same contract as the boosters.

        `X_va`/`y_va` are taken for interface parity and deliberately unused.
        There is nothing to early-stop, and choosing `C` against the validation
        fold would tune on the very rows the OOF score is computed from — the
        fitted-on-what-you-score-on trap that section is at pains to avoid.
        Pass `--params '{"C": ...}'` to change it, one value per run.
        """
        from sklearn.compose import ColumnTransformer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        self.features = list(features or X_tr.columns)
        self.cats = [c for c in self.features if str(X_tr[c].dtype) == "category"]
        self.nums = [c for c in self.features if c not in self.cats]

        # Every level is kept (no drop="first"): with L2 the shrinkage is then
        # symmetric across a column's levels instead of measuring all of them
        # against whichever one happened to sort first. handle_unknown="ignore"
        # is a belt — the EDA found no test level missing from train.
        pre = ColumnTransformer(
            [
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32),
                    self.cats,
                ),
                ("num", StandardScaler(), self.nums),
            ]
        )
        self.pipe = Pipeline([("pre", pre), ("clf", LogisticRegression(**self.params))])
        self.pipe.fit(X_tr[self.features], y_tr, clf__sample_weight=_norm_w(w_tr, len(y_tr)))
        return self

    def predict(self, X) -> np.ndarray:
        return self.pipe.predict_proba(X[self.features])[:, 1]

    def importance(self):
        """|coefficient| on standardised, one-hot inputs — comparable across
        columns because they share a scale. Called `gain` because that is the
        column train.py aggregates; it is not a tree gain, and unlike one it is
        signed information thrown away, so read it as magnitude only.
        """
        names = [n.split("__", 1)[-1] for n in self.pipe.named_steps["pre"].get_feature_names_out()]
        return pd.DataFrame(
            {"feature": names, "gain": np.abs(self.pipe.named_steps["clf"].coef_.ravel())}
        ).sort_values("gain", ascending=False)
