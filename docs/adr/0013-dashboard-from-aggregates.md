# 0013 — A business dashboard over fixed aggregates

**Status:** accepted, 2026-10-26

## Context

[`docs/use-case.md`](../use-case.md) prices the retention campaign, but its
money figures rest on three assumptions (£5 a call, a 10% save rate, 12 months
retained) that live in one notebook cell. A retention manager cannot change them
without running Jupyter, and the save rate is the one number nobody knows yet.
Business users needed a way to move those assumptions themselves.

## Decision

A **Claude Artifact dashboard**, the
[Churn Campaign Planner](https://claude.ai/artifact/Um1wxkVSasxiBG2831YXKa),
reading **four aggregate CSVs** that `scripts/build_dashboard_data.py` writes to
`reports/dashboard/`:

- `planner_curve.csv`: cumulative totals down the ranked call list at every 0.1%
  of the base, for both rankings (risk, and risk × monthly bill). Every slider
  setting is a row lookup plus `s × H × Σ(p·m) − c × n`, so the page does no
  modelling.
- `totals.csv`, `calibration.csv`, `segments.csv`: the headline figures, the
  calibration-by-decile check, and churn rate by contract, tenure, internet
  service and payment method.

The script mirrors notebook 04: same inputs, same `int(k × N)` cut-offs, and the
same assertion that the served bundle's OOF is bitwise the champion's. At its
default settings the dashboard therefore shows the numbers in `use-case.md`.
The CSVs are committed (about 130KB) so a diff shows when they change.

## Azure equivalent

**Power BI** over the same CSVs in Blob Storage, with what-if parameters for the
four sliders. Once a live campaign exists, Power BI on the campaign table in
Fabric or Synapse would replace the OOF-based curve entirely.

## Rejected

- **Streamlit or Dash in the repo.** A new dependency group and a server to host,
  against [0011](0011-local-compose-only.md)'s "nothing deployed". A link a
  manager can open was the point.
- **Customer-level data in the dashboard.** 594k rows would be ~30MB, and anyone
  who can open the dashboard can download its files. Aggregates answer every
  question the page asks.
- **Live API traffic as the source.** There is none yet. A dashboard over
  `logs/predictions.jsonl` would show nothing a business user can act on.
- **Model metrics (AUC, LB, run history).** They are for whoever trains the
  model; [`experiments.md`](../../experiments.md) already holds them.

## Consequences

- **The dashboard does not follow the repo.** Rebuilding the CSVs changes
  nothing until they are re-uploaded to the dashboard's sources.
- Rebuilding needs `experiments/oof_1024_141318_lgbm.npy`, which DVC keeps in a
  local remote ([0010](0010-dvc-scoped-to-experiments.md)). On another machine
  the script cannot run until that remote is shared.
- The figures describe historical customers scored out-of-fold, and the save
  rate is an assumption. The page says both.
- The dashboard is private; viewers need it shared with them.
- **Revisit when:** the served model changes (rebuild and re-upload), or a live
  campaign with a holdout produces a measured save rate, at which point that
  data should replace the assumption.
