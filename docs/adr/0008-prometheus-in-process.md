# 0008 — Prometheus client in-process; five metrics; no Grafana

**Status:** accepted, 2026-10-24

## Context

A served model needs operational visibility, and ML serving needs one thing
ordinary services do not: the *distribution of what it predicts*. A model that
is up, fast and quietly scoring everything at 0.9 is broken in a way no latency
graph shows.

## Decision

`prometheus_client` mounted at `/metrics`, five metrics, about fifteen lines:

| metric | why |
|---|---|
| `churn_predictions_total{endpoint}` | throughput |
| `churn_prediction_latency_seconds{endpoint}` | the usual |
| `churn_predicted_probability` | **the cheapest drift signal there is** — needs no reference data |
| `churn_out_of_range_inputs_total{field}` | tier two of the schema's validation |
| `churn_model_info{run_id,strategy,cv_score,fold_fingerprint}` | which model answered |

No per-feature-*value* labels. That is how a metrics endpoint becomes a
cardinality incident.

**No Grafana.** Five metrics are readable in Prometheus' own expression browser
at `:9090`. A dashboard here would be a screenshot, not a capability — and it is
one compose service away when that stops being true.

## Azure equivalent

**Azure Monitor managed Prometheus** scrapes this endpoint unchanged — the
exposition format is the interface, so the application code is portable.
**Application Insights** covers latency and failures via auto-instrumentation,
but not the prediction-distribution histogram, which you would still define
yourself as a custom metric.

## Rejected

- **`prometheus-fastapi-instrumentator`.** A dependency to replace fifteen lines,
  and it would not produce the probability histogram, which is the one metric
  here that is about the model rather than the server.
- **Grafana in compose.** See above.
- **Logging metrics to MLflow.** MLflow records *training*. Serving telemetry is
  a different question on a different timescale.

## Consequences

- `/metrics` is unauthenticated. Fine bound to localhost; it needs network policy
  or a scrape credential anywhere else.
- Counters reset on restart, which is normal for Prometheus and means rate
  queries, not absolute ones.
- **Revisit when:** more than one replica runs, where per-instance counters need
  aggregation.
