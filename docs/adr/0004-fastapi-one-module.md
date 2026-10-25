# 0004 — FastAPI in one module; the schema is the feature contract

**Status:** accepted, 2026-10-24

## Context

The model needed an HTTP interface. The interesting question was not the web
framework but where the **input contract** lives: a model takes 19 raw columns
with 15 low-cardinality categoricals, and an unknown level does not fail — it
becomes NaN and takes a branch no training row ever took.

## Decision

FastAPI, in a single `src/api.py`, matching `python -m src.train` as the repo's
other entry point. All feature work stays in `src/predict.py`.

FastAPI because **Pydantic makes the API contract the feature contract**. The 15
categoricals are declared as `Literal`s with exactly the levels the model saw,
so an unknown level is a 422 instead of a silent NaN. `tenure >= 1` and
`monthlycharges > 0` are the divide-by-zero guards for the two ratios in
`features.add_telco_features`, which has no guard of its own.

**Validation is two-tiered.** Out-of-*domain* (unseen level, zero divisor) is a
422. Out-of-*range* — a real customer at £130/mo against a training max of
£118.75 — is served and counted. One of those is a customer; a stream of them is
drift. Conflating the two turns a monitoring signal into a rejected sale.

The `Literal` lists duplicate the manifest's category lists on purpose: a
hand-written schema stays greppable and keeps OpenAPI accurate. Startup asserts
the two agree and refuses to serve on drift, which is what stops the copy
rotting.

## Azure equivalent

**Azure ML Managed Online Endpoints.** Its scoring script has the same
`init()` / `run()` shape as this module's `lifespan` + `_score`, so the port is
mechanical. **Azure Container Apps** runs this exact image unchanged, which is
the lower-lock-in option. API Management would sit in front of either for auth
and rate limiting.

## Rejected

- **Flask / Litestar.** Both fine; neither gives schema-derived OpenAPI for free.
- **`mlflow models serve`.** Free and standard, but `/invocations` takes a
  dataframe split and validates nothing about *levels* — exactly the guarantee
  worth having here.
- **A `schemas/` + `services/` package layout.** 160 lines is not a subsystem.
- **Returning a 0/1 decision.** The model was selected on ROC-AUC and no
  operating threshold was ever chosen. It returns a probability and a risk band;
  picking a cutoff is a budget decision, not a model property.

## Consequences

- Levels are written twice and a startup check enforces agreement.
- Adding a feature means editing the schema, which is the point.
- **Revisit when:** the input grows past ~30 fields, where generating the model
  from the manifest starts to beat hand-writing it.
