"""Metrics and drift reporting.

The drift tests matter more than they look. A monitor that never fires is
indistinguishable from a monitor that is broken, so both directions are
asserted: silent on data drawn from the reference, loud on a deliberately
skewed slice.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src import api as A
from src import config as C
from src import monitor as M
from src import predict as P
from src import train as T
from src import utils as U
from src.api import EXAMPLE, app


@pytest.fixture
def client(synthetic_raw, tmp_path, monkeypatch):
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
        "PROCESSED": tmp_path / "processed",
        "PRED_LOG": tmp_path / "logs" / "predictions.jsonl",
        "MLFLOW_TRACKING_URI": f"sqlite:///{tmp_path / 'mlflow.db'}",
        "MLFLOW_EXPERIMENT": "test-monitoring",
    }.items():
        monkeypatch.setattr(C, name, value)
    for d in (C.EXPERIMENTS, C.SUBMISSIONS, C.MODEL_DIR, C.PROCESSED, C.PRED_LOG.parent):
        d.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(U, "LOG_FILE", tmp_path / "experiments.csv")
    monkeypatch.setattr(M, "REFERENCE", tmp_path / "processed" / "reference.parquet")

    P.clear_cache()
    run_id, _ = T.run(
        model="lgbm",
        notes="monitoring fixture",
        params={"learning_rate": 0.1},
        rounds=40,
        early_stopping=15,
    )
    monkeypatch.setenv("MODEL_RUN_ID", run_id)
    with TestClient(app) as c:
        yield c
    P.clear_cache()


def test_metrics_endpoint_exposes_all_five(client):
    client.post("/predict", json=EXAMPLE)
    body = client.get("/metrics/").text
    for metric in (
        "churn_predictions_total",
        "churn_prediction_latency_seconds",
        "churn_predicted_probability",
        "churn_model_info",
    ):
        assert metric in body, f"{metric} missing from /metrics"


def test_out_of_range_counter_fires_without_rejecting(client):
    """Tier two: unusual is served and counted, not refused."""
    before = A._out_of_range_counts().get("monthlycharges", 0)
    assert client.post("/predict", json={**EXAMPLE, "monthlycharges": 130.0}).status_code == 200
    assert A._out_of_range_counts()["monthlycharges"] > before
    assert 'churn_out_of_range_inputs_total{field="monthlycharges"}' in client.get("/metrics/").text


def test_predictions_are_logged_one_json_line_per_row(client):
    """The input src.monitor reads."""
    client.post("/predict/batch", json={"customers": [EXAMPLE] * 3})
    lines = [json.loads(line) for line in C.PRED_LOG.read_text().splitlines() if line.strip()]
    assert len(lines) == 3
    assert {"churn_probability", "ts", "contract", "tenure"} <= set(lines[0])


def test_logging_failure_does_not_break_scoring(client, monkeypatch):
    """A full disk must cost you the log, not the service."""
    monkeypatch.setattr(C, "PRED_LOG", C.PRED_LOG.parent / "nope" / "deeper" / "x.jsonl")
    assert client.post("/predict", json=EXAMPLE).status_code == 200


def test_reference_is_built_from_training_data(client):
    reference = M.build_reference(n_rows=150)
    assert M.REFERENCE.exists()
    assert len(reference) == 150
    # The target and the id must not be in a drift reference: neither is an input.
    assert C.TARGET not in reference.columns
    assert C.ID_COL not in reference.columns


def _serving_rows(synthetic_raw):
    """The fixture in the shape a request carries: lower-cased, no target, no id."""
    return synthetic_raw.rename(columns={c: c.strip().lower() for c in synthetic_raw.columns}).drop(
        columns=["churn", "id"]
    )


def test_no_drift_when_the_traffic_is_the_reference(client, synthetic_raw):
    """The null case, made unambiguous by scoring the reference rows themselves.

    Drift statistics need samples; 60 rows against 200 flags everything whatever
    the data is, which would make this a test of sample size rather than of the
    monitor. Comparing a population against itself cannot drift, so a failure
    here is a real defect.
    """
    M.build_reference(n_rows=200)
    rows = _serving_rows(synthetic_raw)
    with TestClient(app) as c:
        for row in rows.to_dict("records"):
            c.post("/predict", json=row)
    assert M.report()["drifted"] == []


def test_drift_is_detected_on_a_shifted_population(client, synthetic_raw):
    """The other half. A monitor that never fires is indistinguishable from a
    broken one.

    Shifted rather than made constant: a degenerate single-valued column is a
    different failure mode, and what production actually looks like is a
    distribution that moved.
    """
    M.build_reference(n_rows=200)
    rows = _serving_rows(synthetic_raw)
    shifted = rows.assign(
        monthlycharges=(rows.monthlycharges * 0.25 + 15).clip(18.3, 118.7),
        tenure=(rows.tenure // 8 + 1),
    )
    with TestClient(app) as c:
        for row in shifted.to_dict("records"):
            c.post("/predict", json=row)
    drifted = M.report()["drifted"]
    assert drifted, "a population with both numerics shifted should drift"
    assert {"monthlycharges", "tenure"} & set(drifted)


def test_since_window_parsing():
    assert M._since(None) is None
    assert M._since("7d") is not None
    assert M._since("12h") is not None
    with pytest.raises(SystemExit):
        M._since("next tuesday")


def test_evidently_is_not_in_the_serving_import_path():
    """It pulls plotly and nltk; drift is a batch question, not a per-request one."""
    import subprocess
    import sys

    snippet = (
        "import src.api, sys; "
        "print(','.join(k for k in ('evidently', 'plotly', 'nltk') if k in sys.modules))"
    )
    done = subprocess.run(
        [sys.executable, "-c", snippet], capture_output=True, text=True, cwd=C.ROOT, check=False
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == ""


def test_a_drift_report_on_a_fresh_store_keeps_its_artifacts_beside_it(tmp_path, monkeypatch):
    """The monitor creates its experiment through the same helper as training,
    so a drift report against a fresh store cannot leak into the repo's mlruns/."""
    db_dir = tmp_path / "store"
    db_dir.mkdir()
    monkeypatch.setattr(C, "MLFLOW_TRACKING_URI", f"sqlite:///{db_dir / 'mlflow.db'}")
    monkeypatch.setattr(C, "MLFLOW_EXPERIMENT", "drift-isolation")
    html = tmp_path / "report.html"
    html.write_text("<html></html>")

    repo_mlruns = C.ROOT / "mlruns"
    before = set(repo_mlruns.glob("*/*")) if repo_mlruns.exists() else set()
    M._log_to_mlflow({"drifted_columns": 0, "current_rows": 1}, html)
    after = set(repo_mlruns.glob("*/*")) if repo_mlruns.exists() else set()

    assert after == before
    assert list((db_dir / "drift-isolation").glob("*/artifacts/report.html"))


def test_a_drift_report_is_not_written_into_the_real_reports_dir(client, synthetic_raw):
    """Regression: monitor.report wrote to ROOT/"reports" -- always the real
    repo -- so every drift test left an HTML file in reports/ (19 of them)."""
    real_reports = C.ROOT / "reports"
    before = set(real_reports.glob("drift_*.html"))

    M.build_reference(n_rows=200)
    rows = synthetic_raw.rename(columns={c: c.lower() for c in synthetic_raw.columns})
    with TestClient(app) as c:
        for row in rows.drop(columns=["churn", "id"]).head(30).to_dict("records"):
            c.post("/predict", json=row)
    summary = M.report()

    assert set(real_reports.glob("drift_*.html")) == before
    assert C.REPORTS in Path(summary["report"]).parents
