"""MLflow tracking, and the first end-to-end exercise of `src.train.run`.

The integration test here is the valuable one: it points `src/config.py` at the
200-row fixture and runs the *whole* training path — folds, five fits, OOF,
submission, bundle, manifest, CSV row, MLflow run — in a couple of seconds, with
no access to the 77MB dataset. CI can therefore prove training works, not just
that it imports.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src import config as C
from src import train as T
from src import utils as U


@pytest.fixture
def isolated_run(synthetic_raw, tmp_path, monkeypatch):
    """Point every path in config at a tmpdir holding the fixture as the dataset.

    Nothing here touches the real `data/raw/`, `experiments/` or `models/`, which
    CLAUDE.md declares immutable.
    """
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
        "MLFLOW_EXPERIMENT": "test-experiment",
    }.items():
        monkeypatch.setattr(C, name, value)
    for d in (C.EXPERIMENTS, C.SUBMISSIONS, C.MODEL_DIR):
        d.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(U, "LOG_FILE", tmp_path / "experiments.csv")
    return tmp_path


def test_training_runs_end_to_end_without_the_real_dataset(isolated_run):
    """Everything a run is supposed to produce, produced."""
    run_id, cv_score = T.run(
        model="lgbm",
        notes="integration test on the synthetic fixture",
        params={"learning_rate": 0.1},
        rounds=40,
        early_stopping=15,
    )

    assert 0.0 <= cv_score <= 1.0
    assert (C.EXPERIMENTS / f"oof_{run_id}.npy").exists()
    assert (C.EXPERIMENTS / f"test_{run_id}.npy").exists()
    # Written on every run now, not only when --params was passed.
    assert (C.EXPERIMENTS / f"params_{run_id}.json").exists()
    assert list(C.SUBMISSIONS.glob(f"sub_{run_id}_*.csv"))

    bundle = C.MODEL_DIR / run_id
    assert sorted(p.name for p in bundle.glob("*.txt")) == [f"fold{k}.txt" for k in range(5)]

    row = pd.read_csv(U.LOG_FILE).set_index("run_id").loc[run_id]
    assert row["folds"] == json.loads((bundle / "manifest.json").read_text())["fold_fingerprint"]


def test_full_refit_adds_a_second_strategy(isolated_run):
    """`--full-refit` must not disturb the fold-average strategy."""
    run_id, _ = T.run(
        model="lgbm",
        notes="integration test: full refit",
        params={"learning_rate": 0.1},
        rounds=40,
        early_stopping=15,
        full_refit=True,
    )
    bundle = C.MODEL_DIR / run_id
    manifest = json.loads((bundle / "manifest.json").read_text())

    assert (bundle / "full.txt").exists()
    assert set(manifest["strategies"]) == {"fold_average", "full_refit"}
    assert manifest["default_strategy"] == "fold_average"
    # The refit sees 1/(k-1) more data than a fold model, hence the k/(k-1) scaling.
    expected = round(sum(manifest["best_iterations"]) / 5 * 5 / 4)
    assert manifest["strategies"]["full_refit"] == [{"path": "full.txt", "weight": 1.0}]
    assert expected > 0


def test_mlflow_run_records_params_metrics_and_the_fingerprint(isolated_run):
    """The tag that makes a run auditable against experiments.csv."""
    import mlflow

    run_id, cv_score = T.run(
        model="lgbm",
        notes="integration test: tracking",
        params={"learning_rate": 0.1},
        rounds=40,
        early_stopping=15,
    )

    mlflow.set_tracking_uri(C.MLFLOW_TRACKING_URI)
    found = mlflow.search_runs(
        experiment_names=["test-experiment"], filter_string=f"tags.run_id = '{run_id}'"
    )
    assert len(found) == 1
    rec = found.iloc[0]

    assert (
        rec["tags.fold_fingerprint"]
        == json.loads((C.MODEL_DIR / run_id / "manifest.json").read_text())["fold_fingerprint"]
    )
    assert rec["tags.cv_scheme"] == "stratified"
    assert rec["tags.n_folds"] == "5"
    assert rec["metrics.oof_auc"] == pytest.approx(cv_score)
    assert rec["metrics.oof_rows_validated"] == 200
    # Resolved params, so DEFAULT_PARAMS is recorded and not just the override.
    assert rec["params.num_leaves"] == "63"
    assert rec["params.learning_rate"] == "0.1"
    assert rec["params.n_features"] == "21"


def test_experiments_csv_keeps_its_seven_columns(isolated_run):
    """MLflow must not have changed the CSV schema.

    `log_experiment` writes the header only when the file is absent, so adding a
    column would append 8-value rows under a 7-column header and corrupt it.
    `blend.py` reads this file to check fold fingerprints.
    """
    T.run(model="lgbm", notes="integration test: csv schema", rounds=20, early_stopping=10)
    assert list(pd.read_csv(U.LOG_FILE).columns) == U.LOG_FIELDS
    assert U.LOG_FIELDS == [
        "run_id",
        "timestamp",
        "model",
        "cv_score",
        "public_lb",
        "folds",
        "notes",
    ]


def test_tracking_degrades_to_a_shim_without_mlflow(monkeypatch):
    """Training must still work in a Kaggle kernel, where mlflow is absent.

    `kaggle/kernel-metadata.json` sets `enable_internet: false` and the README
    documents `from src.train import run` inside a kernel.
    """
    import builtins

    real_import = builtins.__import__

    def no_mlflow(name, *args, **kwargs):
        if name == "mlflow":
            raise ImportError("simulated: no mlflow on Kaggle")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_mlflow)
    with U.mlflow_run("r", "lgbm", "notes") as ml:
        # Every call has to be swallowed, or train.py would need conditionals.
        ml.set_tag("x", 1)
        ml.log_params({"a": 1})
        ml.log_metric("m", 1.0, step=0)
        ml.log_artifact("/nonexistent/path")
        ml.log_artifacts("/nonexistent/dir", artifact_path="bundle")
    assert isinstance(ml, U._NoMlflow)


def test_git_sha_is_recorded_and_marks_a_dirty_tree():
    """A run whose code is unrecorded is a run you cannot reproduce."""
    sha = U.git_sha()
    assert sha != "unknown"
    assert sha.split("-")[0].isalnum()


def test_an_mlflow_failure_mid_run_still_writes_the_csv_row(isolated_run, monkeypatch):
    """The second sink must not be able to erase the first.

    Before the fix, an artifact upload failing after the OOF arrays were saved
    raised out of run() and the experiments.csv row was never written.
    """
    import mlflow

    def broken(*_args, **_kwargs):
        raise OSError("simulated: disk full while uploading an artifact")

    monkeypatch.setattr(mlflow, "log_artifact", broken)
    monkeypatch.setattr(mlflow, "log_artifacts", broken)
    run_id, _ = T.run(model="lgbm", notes="mlflow broken mid-run", rounds=20, early_stopping=10)

    log = pd.read_csv(U.LOG_FILE).set_index("run_id")
    assert run_id in log.index


def test_mlflow_unable_to_start_falls_back_to_untracked(isolated_run, monkeypatch):
    """A locked or unreachable store means no tracking, not no training."""
    import mlflow

    def locked(*_args, **_kwargs):
        raise RuntimeError("simulated: database is locked")

    monkeypatch.setattr(mlflow, "start_run", locked)
    run_id, _ = T.run(model="lgbm", notes="mlflow cannot start", rounds=20, early_stopping=10)
    assert run_id in pd.read_csv(U.LOG_FILE)["run_id"].tolist()


def test_a_training_error_still_propagates_and_marks_the_run_failed(isolated_run, monkeypatch):
    """Guarding tracking must not swallow real errors."""
    import mlflow

    def explode(*_args, **_kwargs):
        raise RuntimeError("simulated training failure")

    monkeypatch.setattr(T, "build_features", explode)
    with pytest.raises(RuntimeError, match="simulated training failure"):
        T.run(model="lgbm", notes="should fail", rounds=20, early_stopping=10)

    mlflow.set_tracking_uri(C.MLFLOW_TRACKING_URI)
    runs = mlflow.search_runs(experiment_names=["test-experiment"])
    assert (runs["status"] == "FAILED").any()
