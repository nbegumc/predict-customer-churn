# Serving

## Run it

```bash
make serve                 # host, with reload -- http://localhost:8000/docs
make up                    # api + mlflow + prometheus in docker
```

Both read `MODEL_RUN_ID` from `.env` (`cp .env.example .env`). When `models/`
holds more than one bundle the API **refuses to guess** which to serve: `/ready`
returns 503 naming the candidates, and `/health` stays up. It used to pick the
newest run id, which once meant serving a fixed-rounds experiment instead of the
champion. Without a bundle at all, train one:

```bash
uv run python -m src.train --model lgbm --full-refit --notes "serving bundle"
```

## The contract

`POST /predict` takes the 19 raw columns the competition ships, named in
lower-case. The original Kaggle casing (`MonthlyCharges`, `Contract`) is
accepted as an alias and gives an identical answer.

```json
{"churn_probability": 0.7097, "risk_band": "high",
 "run_id": "1024_141318_lgbm", "strategy": "fold_average"}
```

It returns a **probability, not a decision**. The model was selected on ROC-AUC,
a ranking metric, and no operating threshold was ever chosen. `risk_band` says
where the customer falls in the scored population — `high` is the top 10%,
which `notebooks/99_final_writeup.ipynb` measured at 3.7× the base churn rate.
Choosing a cutoff is a retention-budget decision, not a model property.

The thresholds are quantiles of what each strategy *actually serves*, pinned in
the bundle per strategy. They were first taken from OOF predictions, which come
from one fold model per row and are more dispersed than the five-model mean
served here — so only 9.18% of customers reached `high`. The customer above
scored 0.7097 and was labelled `medium`; on the served distribution they are in
the top 10%.

| endpoint | |
|---|---|
| `GET /health` | liveness; never touches the model |
| `GET /ready` | 503 until a bundle is loaded; reports which one |
| `POST /predict` | one customer |
| `POST /predict/batch` | ≤1000, **index-aligned** with the request |
| `GET /model` | the manifest, minus hyperparameters |
| `GET /metrics` | Prometheus |
| `GET /` | a small form, for poking at it |

## Validation is two-tiered, deliberately

**422 — out of domain.** An unknown category level (`"paymentmethod":
"Bitcoin"`), or `tenure: 0` / `monthlycharges: 0`. The latter two are the
divide-by-zero guards for the ratios in `features.add_telco_features`, which has
none of its own: a zero would produce `inf` and score silently.

**200, but counted — out of range.** A customer at £130/mo against a training
max of £118.75 is served, and increments
`churn_out_of_range_inputs_total{field="monthlycharges"}`. One of those is a
customer. A *stream* of them is drift, which `make drift` answers.

Conflating the two turns a monitoring signal into a rejected sale.

## Switching the served model

Both artifacts live in one bundle. Switching is a restart, not a rebuild:

```bash
MODEL_STRATEGY=full_refit make serve
```

`fold_average` is the default and [ADR 0002](adr/0002-fold-average-over-refit.md)
explains why. In short: it is the only one of the two that can be verified
against a prediction that actually scored on the leaderboard.

## Monitoring

```bash
make reference             # once -- 10k-row baseline from data/raw/train.csv
make drift                 # live traffic vs baseline -> reports/drift_<date>.html
make drift SINCE=7d
```

`notebooks/01_eda.ipynb` measured adversarial train-vs-test AUC at 0.5117, so
this data has no intrinsic shift — a flag from `make drift` is new information,
not a threshold artefact.
