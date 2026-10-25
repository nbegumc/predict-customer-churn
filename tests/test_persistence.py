"""Model persistence: the bundle has to be enough to score a row later.

Until this PR, `src/train.py` fitted five models per run and dropped all five,
keeping only prediction arrays — so nothing the repo had ever trained could be
served. These tests pin the properties the serving path will rely on.

All of them run on `tests/fixtures/synthetic_raw.csv`, so CI needs no data.
"""

from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from src import config as C
from src import cv as CV
from src.data import basic_clean, encode_target
from src.features import build_features
from src.models.lgbm import LGBMModel
from src.train import _write_manifest
from src.utils import reduce_memory


@pytest.fixture(scope="module")
def fitted(synthetic_raw):
    """A small booster fitted through the real src/ path, plus its frame."""
    df = encode_target(reduce_memory(basic_clean(synthetic_raw.copy()), False))
    train, _, features = build_features(df, df.drop(columns=[C.TARGET]))
    y = train[C.TARGET].to_numpy()
    model = LGBMModel({"learning_rate": 0.1}, num_boost_round=60, early_stopping=20).fit(
        train[features].iloc[:150],
        y[:150],
        train[features].iloc[150:],
        y[150:],
        features=features,
    )
    return model, train, features


def test_save_load_is_bitwise_identical(fitted, tmp_path):
    """A reloaded booster must predict exactly, not approximately.

    `save` truncates to the early-stopped round, so a reload needs no iteration
    count — and the whole parity story downstream depends on this being exact
    rather than close.
    """
    model, train, features = fitted
    path = tmp_path / "fold0.txt"
    model.save(path)
    before = model.predict(train[features])
    after = LGBMModel.load(path).predict(train[features])
    assert np.array_equal(before, after)


def test_saved_model_carries_its_own_feature_contract(fitted, tmp_path):
    """The model file is a cross-check on the manifest, not just weights."""
    model, _, features = fitted
    path = tmp_path / "fold0.txt"
    model.save(path)
    booster = lgb.Booster(model_file=str(path))
    assert booster.feature_name() == list(features)
    # One list per categorical column, so the codes survive a round trip.
    assert len(booster.pandas_categorical) == 15


def test_shuffled_columns_are_rejected_not_silently_scored(fitted, tmp_path):
    """The reason `predict` passes validate_features=True.

    LightGBM's default checks only the column *count*, so a frame of the right
    width with reordered columns scores silently wrong. Serving builds its frame
    from a persisted feature list; this is the assertion that it was honoured.
    """
    model, train, features = fitted
    path = tmp_path / "fold0.txt"
    model.save(path)
    loaded = LGBMModel.load(path)
    with pytest.raises(lgb.basic.LightGBMError):
        loaded.predict(train[list(features)[::-1]])


def test_string_categoricals_are_rejected(fitted):
    """A payload built without the persisted dtypes must fail loudly."""
    model, train, features = fitted
    as_str = train[features].copy()
    for col in as_str.columns:
        if str(as_str[col].dtype) == "category":
            as_str[col] = as_str[col].astype(str)
    with pytest.raises((ValueError, lgb.basic.LightGBMError)):
        model.predict(as_str)


def test_manifest_describes_the_bundle(fitted, tmp_path):
    """Everything the serving path reads has to actually be in there."""
    model, train, features = fitted
    model.save(tmp_path / "fold0.txt")
    _write_manifest(
        tmp_path,
        run_id="test_run_lgbm",
        model="lgbm",
        train=train,
        features=features,
        cv_score=0.9,
        fold_scores=[0.9],
        best_iters=[model.best_iteration],
        resolved=model.params,
        has_refit=False,
    )
    mf = json.loads((tmp_path / "manifest.json").read_text())

    assert mf["features"] == list(features)
    assert len(mf["features"]) == 21
    assert len(mf["categories"]) == 15
    # The fingerprint of the frame this bundle was built from — here the
    # fixture, not the real table. `13feb807` is locked in test_cv.py instead.
    assert mf["fold_fingerprint"] == CV.fold_fingerprint(train)
    assert mf["cv_scheme"] == "stratified" and mf["n_folds"] == 5
    # Resolved params, not just overrides: the gap that made the champion's
    # hyperparameters recoverable only from whatever lgbm.py said at HEAD.
    assert mf["params"]["num_leaves"] == 63
    assert mf["params"]["learning_rate"] == 0.1
    assert mf["best_iterations"] == [model.best_iteration]

    # The manifest must agree with the model file it ships beside.
    booster = lgb.Booster(model_file=str(tmp_path / "fold0.txt"))
    assert mf["features"] == booster.feature_name()


def test_strategies_are_weighted_member_lists(fitted, tmp_path):
    """One code path for fold-averaging, a full refit, or a future blend."""
    model, train, features = fitted
    _write_manifest(
        tmp_path,
        run_id="test_run_lgbm",
        model="lgbm",
        train=train,
        features=features,
        cv_score=0.9,
        fold_scores=[0.9] * 5,
        best_iters=[10] * 5,
        resolved=model.params,
        has_refit=True,
    )
    strategies = json.loads((tmp_path / "manifest.json").read_text())["strategies"]

    assert set(strategies) == {"fold_average", "full_refit"}
    assert [m["path"] for m in strategies["fold_average"]] == [f"fold{k}.txt" for k in range(5)]
    for members in strategies.values():
        assert sum(m["weight"] for m in members) == pytest.approx(1.0)


def test_categories_are_recorded_in_dtype_order(fitted, tmp_path):
    """Order matters: LightGBM encodes categoricals as integer codes.

    `as_categorical` builds the lists as first-appearance order over
    concat(train, test) — an artifact of raw row order that was previously
    written down nowhere.
    """
    _, train, features = fitted
    _write_manifest(
        tmp_path,
        run_id="r",
        model="lgbm",
        train=train,
        features=features,
        cv_score=0.9,
        fold_scores=[0.9],
        best_iters=[10],
        resolved={},
        has_refit=False,
    )
    cats = json.loads((tmp_path / "manifest.json").read_text())["categories"]
    for col, levels in cats.items():
        assert levels == [str(v) for v in train[col].cat.categories]
        assert len(levels) == len(set(levels))


@pytest.mark.realdata
def test_champion_bundle_reproduces_the_retained_predictions():
    """The headline check: is the bundle the thing that scored on the board?

    `test_*.npy` is the mean of the five fold models, so if a fresh run of the
    same config reproduces it bitwise, the saved fold bundle *is* the artifact
    behind public LB 0.91393.
    """
    log = pd.read_csv(C.ROOT / "experiments.csv").set_index("run_id")
    repack = [r for r in log.index if "repackage" in str(log.loc[r, "notes"])]
    reference = C.EXPERIMENTS / "test_0904_202729_lgbm.npy"
    # experiments.csv is committed but experiments/*.npy is gitignored, so a
    # logged run is not evidence that its arrays are on this machine.
    if not repack or not reference.exists():
        pytest.skip("champion arrays not present (experiments/ is gitignored)")
    ref = np.load(reference)
    new = np.load(C.EXPERIMENTS / f"test_{repack[-1]}.npy")
    assert np.array_equal(ref, new)


@pytest.mark.parametrize("model", ["linear", "mlp"])
def test_round_flags_are_rejected_for_non_boosters_before_loading(model, monkeypatch):
    """`--rounds` used to reach LinearModel/MLPModel and raise TypeError in fold 0,
    after a 594k-row load. It must fail first, and loudly."""
    from src import train as T

    def must_not_load():
        raise AssertionError("training data was loaded before the flags were validated")

    monkeypatch.setattr(T, "load", must_not_load)
    with pytest.raises(ValueError, match="boosters|catboost"):
        T.run(model=model, rounds=10)
    with pytest.raises(ValueError):
        T.run(model=model, early_stopping=0)
