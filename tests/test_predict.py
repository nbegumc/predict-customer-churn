"""Train/serve parity, asserted bitwise.

The claim this file defends: a row scored through `src/predict.py` gets exactly
the number it would have got inside the fold loop. Not approximately — bitwise.
That is achievable because float32 arithmetic is elementwise, so batch size
cannot change a value, and because the serving path reuses the training
functions rather than reimplementing the feature maths.

It is worth asserting rather than assuming. The ordering bug this guards
against is silent and small: computing the two ratios from float64 operands
instead of the float32 ones training used changes 31.9% of `charges_ratio`
values by one ulp, which moves 3 in 50,000 predictions. Nothing would ever
fail; the numbers would just be slightly wrong forever.
"""

from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from src import config as C
from src import predict as P
from src import train as T
from src import utils as U
from src.data import basic_clean, encode_target
from src.features import build_features


@pytest.fixture
def bundle(synthetic_raw, tmp_path, monkeypatch):
    """A real bundle, trained on the fixture through the real `src.train.run`."""
    raw = tmp_path / "raw"
    raw.mkdir()
    synthetic_raw.to_csv(raw / "train.csv", index=False)
    synthetic_raw.drop(columns=["Churn"]).to_csv(raw / "test.csv", index=False)
    pd.DataFrame({"id": synthetic_raw["id"], "Churn": 0.0}).to_csv(
        raw / "sample_submission.csv", index=False
    )
    for name, value in {
        "TRAIN_FILE": raw / "train.csv",
        "TEST_FILE": raw / "test.csv",
        "SAMPLE_SUB_FILE": raw / "sample_submission.csv",
        "EXPERIMENTS": tmp_path / "experiments",
        "SUBMISSIONS": tmp_path / "submissions",
        "MODEL_DIR": tmp_path / "models",
        "MLFLOW_TRACKING_URI": f"sqlite:///{tmp_path / 'mlflow.db'}",
        "MLFLOW_EXPERIMENT": "test-predict",
    }.items():
        monkeypatch.setattr(C, name, value)
    for d in (C.EXPERIMENTS, C.SUBMISSIONS, C.MODEL_DIR):
        d.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(U, "LOG_FILE", tmp_path / "experiments.csv")

    P.clear_cache()
    run_id, _ = T.run(
        model="lgbm",
        notes="parity fixture",
        params={"learning_rate": 0.1},
        rounds=40,
        early_stopping=15,
        full_refit=True,
    )
    yield P.load_bundle(run_id)
    P.clear_cache()


def test_serving_frame_is_bitwise_identical_to_the_training_frame(bundle, synthetic_raw):
    """The core parity assertion, column by column.

    Compares the design matrix `src/predict.py` builds from raw rows against
    the one `build_features` builds in training. Any divergence in dtype,
    category order or the two derived ratios shows up here.
    """
    cleaned = encode_target(U.reduce_memory(basic_clean(synthetic_raw.copy()), False))
    train, _, features = build_features(cleaned, cleaned.drop(columns=[C.TARGET]))
    expected = train[features]
    got = P.build_serving_frame(bundle, synthetic_raw)

    assert list(got.columns) == list(expected.columns)
    for col in expected.columns:
        assert got[col].dtype == expected[col].dtype, f"{col} dtype drifted"
        if str(expected[col].dtype) == "category":
            assert list(got[col].cat.categories) == list(expected[col].cat.categories)
            assert np.array_equal(got[col].cat.codes.to_numpy(), expected[col].cat.codes.to_numpy())
        else:
            assert np.array_equal(got[col].to_numpy(), expected[col].to_numpy()), (
                f"{col} diverged: this is the float32-ordering bug if it is "
                f"charges_ratio or avg_monthly_spend"
            )


def test_the_dtype_ordering_actually_matters(synthetic_raw):
    """Proves the bug the ordering prevents is real, not theoretical.

    Downcast-then-derive (training) against derive-then-downcast (the naive
    serving path). If this test ever shows zero differences the guard has become
    unnecessary — on the real 594k-row table it is 31.9% of values.
    """
    df = basic_clean(synthetic_raw.copy())

    training = df.copy()
    U.reduce_memory(training, False)
    train_ratio = (training.totalcharges / (training.tenure * training.monthlycharges)).astype(
        np.float32
    )

    naive = df.copy()  # float64 operands, cast only at the end
    naive_ratio = (naive.totalcharges / (naive.tenure * naive.monthlycharges)).astype(np.float32)

    assert not np.array_equal(train_ratio.to_numpy(), naive_ratio.to_numpy()), (
        "float32 downcasting no longer changes the ratios; the ordering guard "
        "in src/predict.py may be removable"
    )


def test_single_row_equals_batch_of_one(bundle, synthetic_raw):
    """A per-request endpoint must not disagree with a batch job."""
    batch = P.predict_raw(bundle, synthetic_raw)
    for i in (0, 7, len(synthetic_raw) - 1):
        one = P.predict_raw(bundle, synthetic_raw.iloc[[i]])
        assert one[0] == batch[i], f"row {i}: single {one[0]!r} != batch {batch[i]!r}"


def test_duplicate_rows_are_all_returned(bundle, synthetic_raw):
    """Regression guard: `basic_clean` would have collapsed these.

    Its `drop_duplicates()` is right for a training frame and wrong for a
    request — the response would come back shorter than and misaligned with the
    batch that produced it.
    """
    three = pd.concat([synthetic_raw.iloc[[0]]] * 3, ignore_index=True)
    out = P.predict_raw(bundle, three)
    assert len(out) == 3
    assert out[0] == out[1] == out[2]


def test_row_order_is_preserved(bundle, synthetic_raw):
    """A reversed batch must come back reversed, not re-sorted."""
    head = synthetic_raw.iloc[:20]
    forward = P.predict_raw(bundle, head)
    backward = P.predict_raw(bundle, head.iloc[::-1])
    assert np.array_equal(forward, backward[::-1])


def test_caller_frame_is_not_mutated(bundle, synthetic_raw):
    """`reduce_memory` downcasts in place, so predict_raw has to copy."""
    before = synthetic_raw.dtypes.to_dict()
    P.predict_raw(bundle, synthetic_raw)
    assert synthetic_raw.dtypes.to_dict() == before
    assert "charges_ratio" not in synthetic_raw.columns


def test_extra_columns_are_ignored(bundle, synthetic_raw):
    """A payload carrying id, the target, or junk must still score."""
    noisy = synthetic_raw.copy()
    noisy["some_upstream_column"] = "whatever"
    assert np.array_equal(P.predict_raw(bundle, noisy), P.predict_raw(bundle, synthetic_raw))


def test_missing_required_column_is_rejected_by_name(bundle, synthetic_raw):
    short = synthetic_raw.drop(columns=["Contract"])
    with pytest.raises(ValueError, match="contract"):
        P.predict_raw(bundle, short)


def test_unseen_category_level_does_not_crash(bundle, synthetic_raw):
    """It becomes NaN and takes the tree's missing branch.

    Out of domain, which is why `src/api.py` rejects it with a 422 rather than
    letting it reach here — but this layer must degrade, not explode.
    """
    odd = synthetic_raw.iloc[[0]].copy()
    odd["PaymentMethod"] = "Bitcoin"
    out = P.predict_raw(bundle, odd)
    assert out.shape == (1,) and np.isfinite(out[0])


def test_both_strategies_load_and_score(bundle, synthetic_raw):
    """fold_average is 5 members at 0.2; full_refit is 1 at 1.0."""
    fold_avg = P.load_bundle(bundle.run_id, "fold_average")
    refit = P.load_bundle(bundle.run_id, "full_refit")

    assert len(fold_avg.members) == 5
    assert len(refit.members) == 1
    assert sum(w for _, w in fold_avg.members) == pytest.approx(1.0)

    a = P.predict_raw(fold_avg, synthetic_raw)
    b = P.predict_raw(refit, synthetic_raw)
    assert a.shape == b.shape == (len(synthetic_raw),)
    assert not np.array_equal(a, b), "two different estimators should not agree bitwise"


def test_fold_average_is_the_mean_of_its_members(bundle, synthetic_raw):
    """What `train.py` wrote into test_*.npy, reproduced at serving time."""
    fold_avg = P.load_bundle(bundle.run_id, "fold_average")
    X = P.build_serving_frame(fold_avg, synthetic_raw)
    by_hand = np.mean([m.predict(X) for m, _ in fold_avg.members], axis=0)
    assert np.allclose(P.predict_raw(fold_avg, synthetic_raw), by_hand, rtol=0, atol=1e-12)


def test_unknown_strategy_lists_what_is_available(bundle):
    with pytest.raises(KeyError, match="fold_average"):
        P.load_bundle(bundle.run_id, "no_such_strategy")


def test_missing_bundle_says_how_to_make_one(tmp_path):
    with pytest.raises(FileNotFoundError, match="src.train"):
        P.load_bundle(tmp_path / "not-a-bundle")


def test_manifest_and_booster_must_agree(bundle, tmp_path):
    """A bundle whose manifest disagrees with its model is refused.

    Serving builds the frame from the manifest, so a stale feature list would
    make every prediction subtly wrong rather than failing.
    """
    path = P.bundle_path(bundle.run_id)
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["features"] = manifest["features"][::-1]
    (path / "manifest.json").write_text(json.dumps(manifest))
    P.clear_cache()
    with pytest.raises(ValueError, match="inconsistent"):
        P.load_bundle(bundle.run_id)


@pytest.mark.realdata
def test_champion_bundle_reproduces_its_saved_test_predictions():
    """End-to-end on the real data: raw CSV -> the leaderboard artifact.

    `test_*.npy` is the mean of the five fold models, so scoring
    `data/raw/test.csv` through the fold_average strategy must reproduce it.
    Bitwise equality here means the serving path and the training path agree on
    all 254,655 rows of real, unseen input.
    """
    log = pd.read_csv(C.ROOT / "experiments.csv").set_index("run_id")
    repack = [r for r in log.index if "repackage" in str(log.loc[r, "notes"])]
    if not repack or not (C.MODEL_DIR / repack[-1] / "manifest.json").exists():
        pytest.skip("no packaged bundle on this machine")

    P.clear_cache()
    loaded = P.load_bundle(repack[-1], "fold_average")
    got = P.predict_raw(loaded, pd.read_csv(C.TEST_FILE))
    expected = np.load(C.EXPERIMENTS / f"test_{repack[-1]}.npy")
    assert np.array_equal(got, expected), (
        f"serving diverged from training on real data: "
        f"{int((got != expected).sum()):,}/{len(got):,} rows, "
        f"max |delta| {np.abs(got - expected).max():.3e}"
    )


def test_lightgbm_is_the_only_heavy_import_inference_needs():
    """Keeps src/predict.py out of the sklearn/catboost dependency tree."""
    import subprocess
    import sys

    snippet = (
        "import src.predict, sys; "
        "print(','.join(k for k in ('sklearn', 'catboost', 'mlflow', 'matplotlib') "
        "if k in sys.modules))"
    )
    done = subprocess.run(
        [sys.executable, "-c", snippet],
        capture_output=True,
        text=True,
        cwd=C.ROOT,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == ""


def test_booster_rejects_a_frame_predict_raw_would_never_build(bundle, synthetic_raw):
    """Belt to build_serving_frame's braces: validate_features is on."""
    X = P.build_serving_frame(bundle, synthetic_raw)
    model = bundle.members[0][0]
    with pytest.raises(lgb.basic.LightGBMError):
        model.predict(X[list(X.columns)[::-1]])


def test_feature_lists_differing_only_in_length_raise_valueerror(bundle):
    """Regression: a manifest that is a prefix-extension of the booster's
    feature list used to raise StopIteration from inside the error message."""
    path = P.bundle_path(bundle.run_id)
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["features"] = manifest["features"] + ["extra_a", "extra_b"]
    (path / "manifest.json").write_text(json.dumps(manifest))
    P.clear_cache()
    with pytest.raises(ValueError, match="First divergence at position 21"):
        P.load_bundle(bundle.run_id)


def test_combine_members_reproduces_the_historical_accumulation():
    """train.py and predict_raw share one combine function. It must still give
    exactly what `test_pred += p / n` gave, or the saved arrays stop matching."""
    import numpy as np

    from src.utils import combine_members

    rng = np.random.default_rng(0)
    preds = [rng.random(10_000) for _ in range(5)]
    historical = np.zeros(10_000)
    for p in preds:
        historical += p / len(preds)
    assert np.array_equal(combine_members(preds, [0.2] * 5), historical)
    # A single member at weight 1.0 (full_refit) comes back untouched.
    assert np.array_equal(combine_members(preds[:1], [1.0]), preds[0])
    # Unequal weights are a plain weighted sum (a future blend).
    assert np.allclose(combine_members(preds[:2], [0.7, 0.3]), 0.7 * preds[0] + 0.3 * preds[1])
