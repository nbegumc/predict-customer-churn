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

import os
import threading
from contextlib import asynccontextmanager
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
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


STATE: dict = {"bundle": None, "error": None, "out_of_range": {}}
# FastAPI runs sync `def` endpoints in a threadpool, so the out-of-range
# counter is updated concurrently and needs a lock around its read-modify-write.
_STATE_LOCK = threading.Lock()


def _reset_state() -> None:
    """Every key the request path reads. Called on startup and shutdown, so a
    second app instance in the same process never finds a half-empty dict."""
    STATE.update({"bundle": None, "error": None, "out_of_range": {}})


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
            with _STATE_LOCK:
                STATE["out_of_range"][col] = STATE["out_of_range"].get(col, 0) + n


def _score(customers: list[Customer]) -> list[Prediction]:
    bundle = _bundle()
    frame = pd.DataFrame([c.model_dump() for c in customers])
    _count_out_of_range(frame)
    probabilities = P.predict_raw(bundle, frame)
    return [
        Prediction(
            churn_probability=float(p),
            risk_band=bundle.risk_band(float(p)),
            run_id=bundle.run_id,
            strategy=bundle.strategy,
        )
        for p in probabilities
    ]


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
    with _STATE_LOCK:
        counts = dict(STATE["out_of_range"])
    return _bundle().describe() | {"out_of_range_inputs": counts}


@app.post("/predict", response_model=Prediction, tags=["scoring"])
def predict_one(customer: Customer) -> Prediction:
    """Score one customer.

    Returns a probability rather than a 0/1 decision on purpose: this model was
    selected on ROC-AUC, which is a ranking metric, and no operating threshold
    was ever chosen. Picking one is a business decision about retention budget —
    `risk_band` reports where the customer falls in the scored population.
    """
    return _score([customer])[0]


@app.post("/predict/batch", response_model=BatchResponse, tags=["scoring"])
def predict_batch(request: BatchRequest) -> BatchResponse:
    """Score up to 1000 customers. Output is index-aligned with the input.

    Alignment is a real guarantee, not an incidental one: `data.basic_clean`
    would have deduplicated the frame and returned fewer rows than were sent.
    """
    return BatchResponse(predictions=_score(request.customers))


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
