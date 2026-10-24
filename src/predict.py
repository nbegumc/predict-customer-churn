"""The one inference path: a saved bundle plus raw rows in, probabilities out.

    from src.predict import load_bundle, predict_raw
    bundle = load_bundle("1024_141318_lgbm")
    predict_raw(bundle, raw_rows)        # -> np.ndarray of probabilities

Used by `src/api.py`, by the parity test, and by the strategy comparison, so
there is exactly one definition of how a row becomes a prediction.

Three things here are easy to get wrong, and all three are measured rather than
assumed. They are why this module exists instead of the API calling
`build_features` directly.

**1. The dtype regime has to come first.** `utils.reduce_memory` casts every
float column to float32, and `data.load` applies it *before*
`features.add_telco_features` computes `charges_ratio` and `avg_monthly_spend`.
Compute those ratios from float64 operands instead and 31.9% of
`charges_ratio` values differ by one ulp (`60.1` is stored as
`60.099998474121094`). LightGBM's split thresholds were learned on the
float32-rounded values, so near-threshold rows can flip: measured at 3 in
50,000 predictions, max |delta p| 0.0025. Small, but free to prevent, and the
parity test asserts bitwise equality so it stays prevented.

**2. `data.basic_clean` must not be called here.** It ends in
`drop_duplicates()`, so three identical customers in one batch collapse to a
single row and the response comes back misaligned with the request. Only the
header normalisation is wanted, which is why `data.normalise_columns` exists.

**3. `features.build_features` must not be called here either.** It calls
`as_categorical`, which rebuilds each column's category list from the frame it
is given — so a request would get codes derived from the request. The lists are
first-appearance order over the original concat(train, test) and are an artifact
of raw row order; they travel in the manifest and are applied verbatim below.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .data import normalise_columns
from .features import DERIVED, add_telco_features
from .models.lgbm import LGBMModel
from .utils import reduce_memory

# Loading a bundle reads ~30MB off disk and parses five boosters, so it happens
# once per process rather than once per request.
_CACHE: dict[tuple[str, str], Bundle] = {}

LOADERS = {"lgbm": LGBMModel}


@dataclass(frozen=True)
class Bundle:
    """A loaded, ready-to-score model bundle."""

    manifest: dict
    members: tuple[tuple[object, float], ...]  # (model, weight)
    strategy: str

    @property
    def features(self) -> list[str]:
        return self.manifest["features"]

    @property
    def run_id(self) -> str:
        return self.manifest["run_id"]

    def describe(self) -> dict:
        """The manifest without the hyperparameters — for a /model endpoint."""
        return {k: v for k, v in self.manifest.items() if k != "params"} | {
            "strategy": self.strategy,
            "n_members": len(self.members),
        }


def bundle_path(run_id_or_path: str | Path) -> Path:
    """Accept either a run_id under MODEL_DIR or a path to a bundle directory."""
    candidate = Path(run_id_or_path)
    return candidate if candidate.is_dir() else C.MODEL_DIR / str(run_id_or_path)


def load_bundle(run_id_or_path: str | Path, strategy: str | None = None) -> Bundle:
    """Read a bundle's manifest and its members. Cached per (path, strategy).

    `strategy` names a weighted member list in the manifest — `fold_average`
    (five boosters at 0.2) or `full_refit` (one at 1.0). Both go through the
    same weighted sum in `predict_raw`, which is also how a future
    lgbm+catboost blend would be served: one more member with its weight.
    """
    path = bundle_path(run_id_or_path)
    manifest_file = path / "manifest.json"
    if not manifest_file.exists():
        raise FileNotFoundError(
            f"no manifest at {manifest_file}. Train with `uv run python -m src.train` "
            f"to produce a bundle, or set MODEL_RUN_ID to an existing one in {C.MODEL_DIR}."
        )

    manifest = json.loads(manifest_file.read_text())
    strategy = strategy or manifest["default_strategy"]
    if strategy not in manifest["strategies"]:
        raise KeyError(
            f"strategy {strategy!r} not in bundle {path.name}; "
            f"available: {sorted(manifest['strategies'])}"
        )

    key = (str(path), strategy)
    if key in _CACHE:
        return _CACHE[key]

    model_type = manifest["model"]
    if model_type not in LOADERS:
        raise NotImplementedError(
            f"bundle was trained with {model_type!r}, which has no loader. "
            f"Add one to src/predict.py LOADERS alongside {sorted(LOADERS)}."
        )
    loader = LOADERS[model_type]

    members = tuple(
        (loader.load(path / m["path"]), float(m["weight"]))
        for m in manifest["strategies"][strategy]
    )

    # The model file independently records the feature order it was trained on,
    # so the manifest can be checked rather than trusted. A mismatch means the
    # bundle was assembled wrongly and every prediction would be subtly wrong.
    trained_on = members[0][0].booster.feature_name()
    if trained_on != manifest["features"]:
        raise ValueError(
            f"bundle {path.name} is inconsistent: manifest lists {len(manifest['features'])} "
            f"features but the booster was trained on {len(trained_on)}. "
            f"First divergence at position "
            f"{next(i for i, (a, b) in enumerate(zip(trained_on, manifest['features'])) if a != b)}."
        )

    bundle = Bundle(manifest=manifest, members=members, strategy=strategy)
    _CACHE[key] = bundle
    return bundle


def build_serving_frame(bundle: Bundle, df_raw: pd.DataFrame) -> pd.DataFrame:
    """Raw rows -> the exact design matrix the folds were trained on.

    Kept separate from `predict_raw` so the parity test can compare this frame
    against the training frame column by column.
    """
    df = normalise_columns(df_raw.copy())

    # Prefer the list pinned at train time; derive it for bundles written
    # before the manifest carried one.
    required = bundle.manifest.get("raw_columns") or [
        c for c in bundle.features if c not in DERIVED
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    # Order matters: downcast first, then derive. See point 1 in the module
    # docstring — the ratios must be computed from float32 operands, as they
    # were in training, or they differ by an ulp from what the trees expect.
    reduce_memory(df, verbose=False)
    df = add_telco_features(df)

    X = df[bundle.features].copy()
    for col, levels in bundle.manifest["categories"].items():
        # Applied verbatim, never recomputed: LightGBM encodes categoricals as
        # integer codes, so a different level order means different codes. An
        # unseen level becomes NaN and takes the tree's missing branch, which
        # is why src/api.py rejects unseen levels at the schema boundary.
        # Unseen levels are masked to NaN explicitly rather than left for the
        # cast to drop: pandas deprecated the implicit behaviour, and being
        # explicit says what is meant — an unknown level is missing data, and
        # takes the tree's missing branch.
        known = X[col].where(X[col].isin(levels))
        X[col] = known.astype(pd.CategoricalDtype(categories=levels))
    return X


def predict_raw(bundle: Bundle, df_raw: pd.DataFrame) -> np.ndarray:
    """Churn probability per row, in the order the rows came in."""
    X = build_serving_frame(bundle, df_raw)
    preds = [model.predict(X) for model, _ in bundle.members]
    weights = [w for _, w in bundle.members]

    if len(preds) > 1 and len(set(weights)) == 1:
        # Uniform weights are the fold-average case, and `train.py` builds its
        # saved test predictions as `test_pred += p_k / len(folds)`. Dividing by
        # n is NOT bitwise equal to multiplying by 1/n, because 1/5 has no exact
        # binary representation — measured at 4.4e-16 on 79,690 of 254,655 real
        # rows. Twelve orders of magnitude below anything that matters, but
        # dividing the same way makes serving reproduce the leaderboard artifact
        # exactly, which keeps the parity test maximally sensitive.
        return sum(p / len(preds) for p in preds)
    return sum(w * p for w, p in zip(weights, preds))


def clear_cache() -> None:
    """Drop loaded bundles. For tests that rebuild a bundle in a tmpdir."""
    _CACHE.clear()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Score a CSV through a saved bundle.")
    ap.add_argument("--run-id", required=True, help=f"a bundle directory under {C.MODEL_DIR}")
    ap.add_argument("--strategy", default=None, help="fold_average (default) | full_refit")
    ap.add_argument("--csv", default=str(C.TEST_FILE), help="raw CSV to score")
    ap.add_argument("--out", default=None, help="write probabilities to this .npy")
    a = ap.parse_args()

    loaded = load_bundle(a.run_id, a.strategy)
    frame = pd.read_csv(a.csv)
    preds = predict_raw(loaded, frame)
    print(f"{loaded.run_id} [{loaded.strategy}, {len(loaded.members)} member(s)]")
    print(f"scored {len(preds):,} rows  mean={preds.mean():.6f}  std={preds.std():.6f}")
    if a.out:
        np.save(a.out, preds)
        print(f"-> {a.out}")
