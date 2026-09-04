"""A small MLP — the other half of README idea 3.

`0905_015118_linear` established the constraint this model exists to satisfy.
A blend member has to be *different* and *competitive*: the balance identity in
experiments.md forces a model's dissent to be wrong by exactly its solo deficit,
so a structurally different estimator that trails by 0.008 has predictably wrong
dissent and earns no weight. The logistic regression was different enough and
nowhere near competitive.

An MLP is the natural repair. It keeps the structural contrast — dense
distributed representations rather than axis-aligned splits, so it is wrong in
different places than a tree is — while learning the interactions the linear
model could not express, which is where the whole 0.008 went.

`MLPClassifier` rather than torch: scikit-learn is already a dependency, this is
a 594k x ~50 tabular problem that fits comfortably on CPU, and a new dependency
is not worth buying before the idea has shown it works.
"""

from __future__ import annotations

import numpy as np

from .. import config as C
from .linear import preprocessor

DEFAULT_PARAMS = {
    "hidden_layer_sizes": (128, 64),
    "activation": "relu",
    "alpha": 1e-4,
    # sklearn's default of 200 means ~2,400 minibatches per epoch at this row
    # count, which is most of the runtime for no benefit.
    "batch_size": 1024,
    "learning_rate_init": 1e-3,
    "max_iter": 200,
    "early_stopping": True,
    "validation_fraction": 0.1,
    # sklearn stops on `.score()`, which for a classifier is *accuracy* — a poor
    # proxy for AUC at a 22.5% base rate, since it only reads the 0.5 threshold
    # and is flat wherever a ranking change does not cross it. There is no hook
    # to score AUC instead, so the patience is raised well above sklearn's 5 to
    # keep a noisy accuracy plateau from cutting training short. Chosen on that
    # argument, not by comparing fold scores — picking it on a validation fold
    # would tune on the rows the OOF is computed from.
    "n_iter_no_change": 15,
    "random_state": C.SEED,
}


class MLPModel:
    name = "mlp"

    def __init__(self, params: dict | None = None):
        if C.TASK != "binary":
            raise NotImplementedError(
                f"MLPModel returns P(class 1) and only covers TASK='binary'; config says {C.TASK!r}."
            )
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.pipe = None
        self.features: list[str] = []

    def fit(self, X_tr, y_tr, X_va, y_va, features=None, w_tr=None):
        """`X_va`/`y_va` are unused, as in `models/linear.py`.

        Early stopping watches `validation_fraction` of the *training* rows,
        carved out by sklearn itself — deliberately not this fold's validation
        half, which would be stopping on the rows the OOF score is computed
        from.

        `w_tr` is refused rather than ignored: `MLPClassifier.fit` takes no
        `sample_weight`, and silently dropping a weighting scheme would make a
        weighted run look identical to an unweighted one.
        """
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import Pipeline

        if w_tr is not None:
            raise NotImplementedError("MLPClassifier.fit takes no sample_weight")

        self.features = list(features or X_tr.columns)
        cats = [c for c in self.features if str(X_tr[c].dtype) == "category"]
        nums = [c for c in self.features if c not in cats]

        self.pipe = Pipeline(
            [("pre", preprocessor(cats, nums)), ("clf", MLPClassifier(**self.params))]
        )
        self.pipe.fit(X_tr[self.features], y_tr)
        clf = self.pipe.named_steps["clf"]
        print(f"    epochs {clf.n_iter_}, best internal val score {clf.best_validation_score_:.5f}")
        return self

    def predict(self, X) -> np.ndarray:
        return self.pipe.predict_proba(X[self.features])[:, 1]
