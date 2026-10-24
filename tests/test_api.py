"""The service contract.

Most of these are about the API refusing things rather than computing things.
That is the point of putting a schema in front of a model: the failure modes
this repo has actually hit were silent, not loud — a submission column written
under the wrong name that uploaded cleanly and scored 0.5, a `drop_duplicates`
that returns fewer rows than it was given. A 422 is the cheap version of
finding out.
"""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src import config as C
from src import predict as P
from src import train as T
from src import utils as U
from src.api import EXAMPLE, PLAUSIBLE, app

KAGGLE_CASED = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.7,
    "TotalCharges": 151.65,
}


@pytest.fixture
def client(synthetic_raw, tmp_path, monkeypatch):
    """A live app backed by a bundle trained on the fixture."""
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
        "MLFLOW_EXPERIMENT": "test-api",
    }.items():
        monkeypatch.setattr(C, name, value)
    for d in (C.EXPERIMENTS, C.SUBMISSIONS, C.MODEL_DIR):
        d.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(U, "LOG_FILE", tmp_path / "experiments.csv")

    P.clear_cache()
    run_id, _ = T.run(
        model="lgbm",
        notes="api fixture",
        params={"learning_rate": 0.1},
        rounds=40,
        early_stopping=15,
        full_refit=True,
    )
    monkeypatch.setenv("MODEL_RUN_ID", run_id)
    monkeypatch.delenv("MODEL_STRATEGY", raising=False)
    with TestClient(app) as c:
        yield c
    P.clear_cache()


def test_health_does_not_need_a_model():
    """Liveness must answer even when nothing is loaded, or an orchestrator
    sees a crash loop with no way to ask what went wrong."""
    with TestClient(app) as c:
        assert c.get("/health").json() == {"status": "ok"}


def test_ready_reports_which_model_is_serving(client):
    body = client.get("/ready").json()
    assert body["status"] == "ready"
    assert body["strategy"] == "fold_average"
    assert body["n_members"] == 5
    assert body["n_features"] == 21
    assert "fold_fingerprint" in body


def test_predict_returns_a_probability_and_a_band(client):
    body = client.post("/predict", json=EXAMPLE).json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["risk_band"] in {"high", "medium", "low"}
    assert body["strategy"] == "fold_average"


def test_both_header_casings_give_the_same_answer(client):
    """Why the serving path can skip `basic_clean`: the frame arrives correct."""
    lower = client.post("/predict", json=EXAMPLE).json()
    kaggle = client.post("/predict", json=KAGGLE_CASED).json()
    assert lower["churn_probability"] == kaggle["churn_probability"]


def test_api_answer_matches_predict_raw_exactly(client):
    """The HTTP layer must not change the number."""
    direct = P.predict_raw(
        P.load_bundle(client.get("/ready").json()["run_id"]), pd.DataFrame([EXAMPLE])
    )
    assert client.post("/predict", json=EXAMPLE).json()["churn_probability"] == direct[0]


@pytest.mark.parametrize("bad", [{"tenure": 0}, {"tenure": -1}, {"monthlycharges": 0}])
def test_division_guards_are_422_not_inf(client, bad):
    """`features.add_telco_features` divides by tenure and tenure*monthlycharges
    with no guard of its own. A zero would produce inf and score silently."""
    assert client.post("/predict", json={**EXAMPLE, **bad}).status_code == 422


@pytest.mark.parametrize(
    "bad",
    [
        {"paymentmethod": "Bitcoin"},
        {"contract": "Three year"},
        {"internetservice": "Satellite"},
        {"gender": "Unspecified"},
        {"onlinesecurity": "Maybe"},
    ],
)
def test_unknown_category_levels_are_rejected(client, bad):
    """Out of domain, not out of range: an unseen level becomes NaN and takes a
    branch no training row ever took."""
    assert client.post("/predict", json={**EXAMPLE, **bad}).status_code == 422


def test_missing_field_is_rejected_by_name(client):
    missing = {k: v for k, v in EXAMPLE.items() if k != "contract"}
    response = client.post("/predict", json=missing)
    assert response.status_code == 422
    assert "contract" in response.text


def test_out_of_range_but_plausible_values_are_served_and_counted(client):
    """Tier two. A real new customer at GBP 130/mo should be scored; a *stream*
    of them is drift, which is a monitoring question, not a validation one."""
    high = {**EXAMPLE, "monthlycharges": 130.0}
    assert PLAUSIBLE["monthlycharges"][1] < 130.0
    assert client.post("/predict", json=high).status_code == 200
    assert client.get("/model").json()["out_of_range_inputs"]["monthlycharges"] >= 1


def test_batch_is_index_aligned_even_with_duplicates(client):
    """The `basic_clean` regression, at the HTTP boundary.

    Its `drop_duplicates()` would return two predictions for three customers,
    and the caller would silently mis-attribute every one of them.
    """
    body = client.post("/predict/batch", json={"customers": [EXAMPLE] * 3}).json()
    assert len(body["predictions"]) == 3
    assert len({p["churn_probability"] for p in body["predictions"]}) == 1


def test_batch_preserves_order(client):
    low_risk = {
        **EXAMPLE,
        "contract": "Two year",
        "tenure": 70,
        "internetservice": "No",
        "onlinesecurity": "No internet service",
        "onlinebackup": "No internet service",
        "deviceprotection": "No internet service",
        "techsupport": "No internet service",
        "streamingtv": "No internet service",
        "streamingmovies": "No internet service",
    }
    body = client.post("/predict/batch", json={"customers": [EXAMPLE, low_risk]}).json()
    single = [
        client.post("/predict", json=c).json()["churn_probability"] for c in (EXAMPLE, low_risk)
    ]
    assert [p["churn_probability"] for p in body["predictions"]] == single


def test_batch_size_is_capped(client):
    """The cap is the DoS guard; 1001 should be refused before any scoring."""
    assert client.post("/predict/batch", json={"customers": [EXAMPLE] * 1001}).status_code == 422
    assert client.post("/predict/batch", json={"customers": []}).status_code == 422


def test_model_endpoint_describes_the_bundle_without_leaking_params(client):
    body = client.get("/model").json()
    assert "params" not in body
    assert body["features"][:1] == ["gender"]
    assert len(body["categories"]) == 15
    assert body["cv_scheme"] == "stratified"


def test_risk_band_uses_the_bundles_own_quantiles(client):
    """Bands are a threshold on the ranked population, so they need the score
    distribution. A freshly trained bundle pins its own."""
    bundle = P.load_bundle(client.get("/ready").json()["run_id"])
    quantiles = bundle.manifest["score_quantiles"]
    assert quantiles, "a new bundle should pin its score distribution"
    assert bundle.risk_band(float(quantiles["0.9"]) + 1e-9) == "high"
    assert bundle.risk_band(float(quantiles["0.7"]) + 1e-9) == "medium"
    assert bundle.risk_band(0.0) == "low"


def test_strategy_is_switchable_by_env_var(client, monkeypatch):
    """A/B between the two artifacts is a restart, not a second image."""
    monkeypatch.setenv("MODEL_STRATEGY", "full_refit")
    with TestClient(app) as c:
        assert c.get("/ready").json()["strategy"] == "full_refit"
        assert c.get("/ready").json()["n_members"] == 1


def test_schema_drift_against_the_bundle_is_refused(client, monkeypatch):
    """The copy of the category lists in src/api.py must not rot.

    They are written twice on purpose — a hand-written schema stays greppable
    and keeps OpenAPI honest — so something has to notice when they diverge.
    """
    import json

    from src.api import _check_schema_matches_bundle

    bundle = P.load_bundle(client.get("/ready").json()["run_id"])
    drifted = json.loads(json.dumps(bundle.manifest))
    drifted["categories"]["contract"] = ["Month-to-month", "One year", "Three year"]
    with pytest.raises(RuntimeError, match="schema drift on 'contract'"):
        _check_schema_matches_bundle(
            P.Bundle(manifest=drifted, members=bundle.members, strategy=bundle.strategy)
        )


def test_index_page_renders(client):
    page = client.get("/")
    assert page.status_code == 200
    assert "Churn scoring" in page.text
    assert "/predict" in page.text


def test_openapi_documents_the_contract(client):
    spec = client.get("/openapi.json").json()
    assert "/predict" in spec["paths"]
    assert "/predict/batch" in spec["paths"]
    schema = spec["components"]["schemas"]["Customer"]["properties"]
    assert set(schema["contract"]["enum"]) == {"Month-to-month", "One year", "Two year"}
