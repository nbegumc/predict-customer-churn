# 0009 — Evidently as a batch job, not middleware

**Status:** accepted, 2026-10-24

## Context

Drift detection needs a reference distribution and a current one. The tempting
design computes it per request; the honest one notices that "has the population
shifted" is not a question a single request can answer.

## Decision

`src/monitor.py`, a CLI shaped like the existing `src/log_lb.py`. The API
appends one JSON line per scored row to `PRED_LOG` — best-effort, so a full disk
costs the log and not the service — and the CLI compares a window of that
against a 10,000-row sample of training data.

Evidently stays in the `mlops` dependency group and **never enters the serving
image**; a test asserts it never reaches the serving import path. It pulls
plotly and nltk, which would roughly double a 540MB image to answer a question
nobody asks per request.

**This dataset comes with its own null hypothesis.** `notebooks/01_eda.ipynb`
measured adversarial train-vs-test AUC at **0.5117** — the two frames are the
same distribution. So "no drift" is not a guessed threshold here, it is a
measured property, and a flag is genuinely new information.

## Azure equivalent

**Azure ML Model Monitoring**, which is also batch, also on a schedule, and also
built on a reference dataset plus collected inference data. The shape transfers
exactly; Azure's Data Collector replaces the JSONL append. Evidently can be run
inside an Azure ML pipeline job if you would rather keep the library.

## Rejected

- **Evidently in the request path.** Doubles the image for a batch question.
- **Evidently's monitoring dashboard service.** More infrastructure than a
  local-only project justifies.
- **Only the Prometheus probability histogram.** It catches *output* drift and
  needs no reference, but it cannot tell you *which input* moved.

## Consequences

- Drift is only as fresh as the last run of the CLI. A cron or a scheduled
  routine is the obvious next step.
- The prediction log grows unboundedly; `--since` bounds the query, not the file.
- **A real trap, found the hard way:** Evidently picks its statistical test by
  sample size. Under ~1000 rows it uses hypothesis tests (`K-S p_value`,
  `Z-test p_value`) where drift means a value **below** the threshold; above
  that it uses distances (`Wasserstein`, `Jensen-Shannon`) where drift means
  **above**. Assuming one direction reports every column as drifted when a
  200-row sample is compared with itself. `src/monitor.py` reads the method out
  of the metric name; `tests/test_monitoring.py` pins both regimes.
- **Revisit when:** a label feedback loop exists, at which point *performance*
  drift beats input drift as the signal worth alerting on.
