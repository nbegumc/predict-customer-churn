"""Churn scoring service.

    uv run uvicorn src.api:app --reload
    MODEL_RUN_ID=1024_141318_lgbm MODEL_STRATEGY=fold_average uv run uvicorn src.api:app

One module, matching `python -m src.train` as the repo's other entry point. All
feature work happens in `src/predict.py`; this file is the contract and the
plumbing around it.

**The schema is the feature contract.** The 15 categorical columns are declared
as `Literal`s with exactly the levels the model was trained on, so an unknown
level is a 422 rather than a NaN that quietly takes the tree's missing branch.
`tenure >= 1` and `monthlycharges > 0` are the divide-by-zero guards for the two
ratios in `features.add_telco_features`, which has no guard of its own — a
payload with `tenure = 0` would otherwise produce `inf` and score silently.

**Validation is in two tiers on purpose.** A value outside the *measured
training range* is not invalid, it is unusual: a real new customer paying £130/mo
should be scored, not rejected. Those increment an out-of-range counter and pass
through, because a stream of them is drift and drift is a monitoring question.
Conflating "out of domain" with "out of range" is the standard mistake here.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from prometheus_client import Counter, Histogram, Info, make_asgi_app
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from . import config as C
from . import predict as P

MAX_BATCH = 1000

# Measured over train and test together. Outside these a row is unusual, not
# invalid — see the two-tier note above.
PLAUSIBLE = {
    "tenure": (1, 72),
    "monthlycharges": (18.25, 118.75),
    "totalcharges": (18.80, 8684.80),
}

SERVICE = Literal["No", "Yes", "No internet service"]
YES_NO = Literal["Yes", "No"]


class Customer(BaseModel):
    """One customer, in the raw shape the competition ships.

    Field names are the lower-cased forms the model was trained on, with an
    alias for the original Kaggle casing, so both spellings post. That is what
    lets the serving path skip `data.basic_clean` — whose `drop_duplicates()`
    would silently collapse identical rows in a batch.
    """

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    gender: Literal["Male", "Female"]
    seniorcitizen: int = Field(
        ge=0, le=1, validation_alias=AliasChoices("seniorcitizen", "SeniorCitizen")
    )
    partner: YES_NO = Field(validation_alias=AliasChoices("partner", "Partner"))
    dependents: YES_NO = Field(validation_alias=AliasChoices("dependents", "Dependents"))
    tenure: int = Field(ge=1, le=120, description="months; >=1 guards a division")
    phoneservice: YES_NO = Field(validation_alias=AliasChoices("phoneservice", "PhoneService"))
    multiplelines: Literal["No", "Yes", "No phone service"] = Field(
        validation_alias=AliasChoices("multiplelines", "MultipleLines")
    )
    internetservice: Literal["DSL", "Fiber optic", "No"] = Field(
        validation_alias=AliasChoices("internetservice", "InternetService")
    )
    onlinesecurity: SERVICE = Field(
        validation_alias=AliasChoices("onlinesecurity", "OnlineSecurity")
    )
    onlinebackup: SERVICE = Field(validation_alias=AliasChoices("onlinebackup", "OnlineBackup"))
    deviceprotection: SERVICE = Field(
        validation_alias=AliasChoices("deviceprotection", "DeviceProtection")
    )
    techsupport: SERVICE = Field(validation_alias=AliasChoices("techsupport", "TechSupport"))
    streamingtv: SERVICE = Field(validation_alias=AliasChoices("streamingtv", "StreamingTV"))
    streamingmovies: SERVICE = Field(
        validation_alias=AliasChoices("streamingmovies", "StreamingMovies")
    )
    contract: Literal["Month-to-month", "One year", "Two year"] = Field(
        validation_alias=AliasChoices("contract", "Contract")
    )
    paperlessbilling: YES_NO = Field(
        validation_alias=AliasChoices("paperlessbilling", "PaperlessBilling")
    )
    paymentmethod: Literal[
        "Mailed check",
        "Credit card (automatic)",
        "Electronic check",
        "Bank transfer (automatic)",
    ] = Field(validation_alias=AliasChoices("paymentmethod", "PaymentMethod"))
    monthlycharges: float = Field(
        gt=0,
        validation_alias=AliasChoices("monthlycharges", "MonthlyCharges"),
        description=">0 guards a division",
    )
    totalcharges: float = Field(ge=0, validation_alias=AliasChoices("totalcharges", "TotalCharges"))


class Prediction(BaseModel):
    churn_probability: float = Field(description="P(churn). Not a decision — see the note below.")
    risk_band: Literal["high", "medium", "low"]
    run_id: str
    strategy: str


class BatchRequest(BaseModel):
    customers: list[Customer] = Field(min_length=1, max_length=MAX_BATCH)


class BatchResponse(BaseModel):
    predictions: list[Prediction]


# Five metrics, hand-rolled on prometheus_client rather than pulling in an
# instrumentator package to replace fifteen lines. No per-feature-value labels:
# that is how a metrics endpoint turns into a cardinality incident.
PREDICTIONS = Counter("churn_predictions_total", "Rows scored", ["endpoint"])
LATENCY = Histogram("churn_prediction_latency_seconds", "Scoring latency", ["endpoint"])
# The cheapest drift signal there is: a shifted score distribution is the first
# thing you notice, and it needs no labels and no reference data.
PROBABILITY = Histogram(
    "churn_predicted_probability",
    "Distribution of predicted churn probability",
    buckets=[i / 10 for i in range(11)],
)
OUT_OF_RANGE = Counter(
    "churn_out_of_range_inputs_total", "Values outside the training range", ["field"]
)
MODEL = Info("churn_model", "Which bundle is serving")

STATE: dict = {"bundle": None, "error": None}


def _reset_state() -> None:
    """Every key the request path reads. Called on startup and shutdown, so a
    second app instance in the same process never finds a half-empty dict."""
    STATE.update({"bundle": None, "error": None})


def _check_schema_matches_bundle(bundle: P.Bundle) -> None:
    """Refuse to start if the Literals disagree with the trained categories.

    The levels are written twice — here as `Literal`s and in the bundle
    manifest. That duplication is deliberate: a hand-written schema stays
    greppable and keeps OpenAPI honest, where a dynamically built model would
    not. This is what stops the copy rotting.
    """
    declared = {
        name: set(field.annotation.__args__)
        for name, field in Customer.model_fields.items()
        if getattr(field.annotation, "__args__", None) and name != "seniorcitizen"
    }
    trained = {col: set(levels) for col, levels in bundle.manifest["categories"].items()}
    for col, levels in trained.items():
        if col in declared and declared[col] != levels:
            raise RuntimeError(
                f"schema drift on {col!r}: the API accepts {sorted(declared[col])} "
                f"but the model was trained on {sorted(levels)}. "
                f"Update src/api.py to match the bundle, or serve a different bundle."
            )


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load the bundle once per process; ~30MB and five boosters is not per-request.

    A failure here is recorded rather than raised, so the container still starts
    and /health still answers — otherwise an orchestrator sees a crash loop with
    no way to ask what went wrong. /ready is what reports the truth.
    """
    _reset_state()
    run_id = os.getenv("MODEL_RUN_ID")
    strategy = os.getenv("MODEL_STRATEGY") or None
    try:
        if not run_id:
            candidates = sorted(
                p.name for p in C.MODEL_DIR.glob("*/") if (p / "manifest.json").exists()
            )
            if not candidates:
                raise FileNotFoundError(
                    f"no bundle in {C.MODEL_DIR} and MODEL_RUN_ID is unset. "
                    "Train one with `uv run python -m src.train --full-refit`."
                )
            # Exactly one bundle (the container case) is unambiguous. More than
            # one is not: picking the newest once served a fixed-rounds
            # experiment instead of the champion, and %m%d run ids have no year,
            # so "newest" is not even well defined across a new year.
            if len(candidates) > 1:
                raise LookupError(
                    f"MODEL_RUN_ID is unset and {C.MODEL_DIR} holds {len(candidates)} bundles "
                    f"({', '.join(candidates)}). Refusing to guess which one to serve; "
                    "set MODEL_RUN_ID (see .env.example)."
                )
            run_id = candidates[0]
        bundle = P.load_bundle(run_id, strategy)
        _check_schema_matches_bundle(bundle)
        STATE["bundle"] = bundle
        MODEL.info(
            {
                "run_id": bundle.run_id,
                "strategy": bundle.strategy,
                "cv_score": str(bundle.manifest["cv_score"]),
                "fold_fingerprint": bundle.manifest["fold_fingerprint"],
            }
        )
        print(f"loaded {bundle.run_id} [{bundle.strategy}, {len(bundle.members)} member(s)]")
    except Exception as exc:  # reported via /ready instead of crash-looping
        STATE["error"] = f"{type(exc).__name__}: {exc}"
        print(f"model load FAILED: {STATE['error']}")
    yield
    _reset_state()


app = FastAPI(
    title="Churn scoring",
    description=(
        "Ranks customers by churn risk for a retention call list. "
        "Returns a probability, not a decision: this model was selected on "
        "ROC-AUC and no operating threshold was ever chosen."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def _bundle() -> P.Bundle:
    if STATE["bundle"] is None:
        raise HTTPException(status_code=503, detail=f"model not loaded: {STATE['error']}")
    return STATE["bundle"]


def _count_out_of_range(frame: pd.DataFrame) -> None:
    """Tier two: serve it, but count it. A stream of these is drift."""
    for col, (lo, hi) in PLAUSIBLE.items():
        n = int(((frame[col] < lo) | (frame[col] > hi)).sum())
        if n:
            OUT_OF_RANGE.labels(field=col).inc(n)


def _out_of_range_counts() -> dict[str, int]:
    """Read the counts back from the Prometheus counter, the single source.

    An earlier version kept a parallel dict in STATE, updated by an unlocked
    read-modify-write from threadpool handlers. prometheus_client counters are
    thread-safe, so the duplicate (and the lock it then needed) are gone.
    """
    return {
        sample.labels["field"]: int(sample.value)
        for family in OUT_OF_RANGE.collect()
        for sample in family.samples
        if sample.name.endswith("_total")
    }


def _log_predictions(frame: pd.DataFrame, probabilities) -> None:
    """One JSON line per scored row — the input src.monitor reads.

    Best-effort and append-only: a full disk must not take down scoring. It is
    a synchronous write, which is fine at this scale and is the thing to move
    off the request path first if throughput ever matters.
    """
    try:
        now = f"{datetime.now().astimezone():%Y-%m-%dT%H:%M:%S%z}"
        with C.PRED_LOG.open("a") as handle:
            for row, probability in zip(frame.to_dict("records"), probabilities):
                handle.write(
                    json.dumps({**row, PREDICTION_KEY: float(probability), "ts": now}) + "\n"
                )
    except OSError as exc:
        print(f"prediction log write skipped: {exc}")


def _score(customers: list[Customer], endpoint: str = "/predict") -> list[Prediction]:
    bundle = _bundle()
    frame = pd.DataFrame([c.model_dump() for c in customers])
    _count_out_of_range(frame)

    started = time.perf_counter()
    probabilities = P.predict_raw(bundle, frame)
    LATENCY.labels(endpoint=endpoint).observe(time.perf_counter() - started)
    PREDICTIONS.labels(endpoint=endpoint).inc(len(probabilities))
    for probability in probabilities:
        PROBABILITY.observe(float(probability))
    _log_predictions(frame, probabilities)

    return [
        Prediction(
            churn_probability=float(p),
            risk_band=bundle.risk_band(float(p)),
            run_id=bundle.run_id,
            strategy=bundle.strategy,
        )
        for p in probabilities
    ]


# Prometheus scrapes this. Mounted as a sub-app rather than wrapped in a
# route, so the exposition format is handled by the client library.
app.mount("/metrics", make_asgi_app())

PREDICTION_KEY = "churn_probability"


@app.get("/health", tags=["ops"])
def health() -> dict:
    """Liveness. Deliberately does not touch the model."""
    return {"status": "ok"}


@app.get("/ready", tags=["ops"])
def ready() -> dict:
    """Readiness: is a model actually loaded, and which one?"""
    if STATE["bundle"] is None:
        raise HTTPException(status_code=503, detail=f"model not loaded: {STATE['error']}")
    bundle = STATE["bundle"]
    return {
        "status": "ready",
        "run_id": bundle.run_id,
        "strategy": bundle.strategy,
        "n_members": len(bundle.members),
        "n_features": len(bundle.features),
        "cv_score": bundle.manifest["cv_score"],
        "fold_fingerprint": bundle.manifest["fold_fingerprint"],
    }


@app.get("/model", tags=["ops"])
def model_info() -> dict:
    """The manifest, minus hyperparameters. What is being served, and from what."""
    return _bundle().describe() | {"out_of_range_inputs": _out_of_range_counts()}


@app.post("/predict", response_model=Prediction, tags=["scoring"])
def predict_one(customer: Customer) -> Prediction:
    """Score one customer.

    Returns a probability rather than a 0/1 decision on purpose: this model was
    selected on ROC-AUC, which is a ranking metric, and no operating threshold
    was ever chosen. Picking one is a business decision about retention budget —
    `risk_band` reports where the customer falls in the scored population.
    """
    return _score([customer], "/predict")[0]


@app.post("/predict/batch", response_model=BatchResponse, tags=["scoring"])
def predict_batch(request: BatchRequest) -> BatchResponse:
    """Score up to 1000 customers. Output is index-aligned with the input.

    Alignment is a real guarantee, not an incidental one: `data.basic_clean`
    would have deduplicated the frame and returned fewer rows than were sent.
    """
    return BatchResponse(predictions=_score(request.customers, "/predict/batch"))


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index() -> str:
    """A minimal form, so the compose stack is demonstrable without a client."""
    bundle = STATE["bundle"]
    status = (
        f"{bundle.run_id} · {bundle.strategy} · {len(bundle.members)} member(s) · "
        f"CV {bundle.manifest['cv_score']}"
        if bundle
        else f"no model loaded — {STATE['error']}"
    )
    fields = "".join(
        f'<label>{name}<input name="{name}" value="{default}"></label>'
        for name, default in EXAMPLE.items()
    )
    return f"""<!doctype html><meta charset="utf-8"><title>Churn scoring</title>
<style>
 body{{font:14px/1.5 system-ui,sans-serif;max-width:52rem;margin:2rem auto;padding:0 1rem}}
 label{{display:flex;justify-content:space-between;gap:1rem;padding:.2rem 0}}
 input{{width:18rem}} fieldset{{border:1px solid #ccc;border-radius:6px}}
 #out{{margin-top:1rem;padding:1rem;background:#f5f5f5;border-radius:6px;white-space:pre-wrap}}
 code{{background:#eee;padding:.1rem .3rem;border-radius:3px}}
</style>
<h1>Churn scoring</h1>
<p><small>{status}</small></p>
<p>Ranks customers for a retention call list. Returns a probability, not a
decision — see <a href="/docs">/docs</a> for the full contract.</p>
<form id="f"><fieldset><legend>customer</legend>{fields}</fieldset>
<p><button>Score</button></p></form>
<div id="out">Submit the form to score this customer.</div>
<script>
document.getElementById('f').onsubmit = async (e) => {{
  e.preventDefault();
  const body = {{}};
  for (const [k, v] of new FormData(e.target)) body[k] = isNaN(v) || v === '' ? v : Number(v);
  const r = await fetch('/predict', {{
    method: 'POST', headers: {{'Content-Type': 'application/json'}}, body: JSON.stringify(body)
  }});
  document.getElementById('out').textContent =
    (r.ok ? '' : `HTTP ${{r.status}}\\n`) + JSON.stringify(await r.json(), null, 2);
}};
</script>"""


# A real row, so the form scores on first click and the OpenAPI example is valid.
EXAMPLE = {
    "gender": "Female",
    "seniorcitizen": 0,
    "partner": "Yes",
    "dependents": "No",
    "tenure": 2,
    "phoneservice": "Yes",
    "multiplelines": "No",
    "internetservice": "Fiber optic",
    "onlinesecurity": "No",
    "onlinebackup": "No",
    "deviceprotection": "No",
    "techsupport": "No",
    "streamingtv": "No",
    "streamingmovies": "No",
    "contract": "Month-to-month",
    "paperlessbilling": "Yes",
    "paymentmethod": "Electronic check",
    "monthlycharges": 70.7,
    "totalcharges": 151.65,
}
