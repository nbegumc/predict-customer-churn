# Experiment notes

`experiments.csv` holds the numbers (auto-appended by `src/utils.log_experiment`).
This file holds the *reasoning* — what you tried and what you concluded.

---

## Submission budget

Self-directed project — no deadline and no daily submission cap recorded yet.
If this becomes a scored competition, put the close date and the submissions/day
limit here, because the binding constraint is *information per submission*: two
runs whose CV differs by a thousandth are one measurement, not two.

Record every public LB score with `uv run python -m src.log_lb <run_id> <score>`
the same day. An unlogged submission is a slot spent for nothing.

## Validation decision

- CV scheme: **UNDECIDED.** `src/config.py` currently holds `CV_SCHEME =
  "stratified"`, `N_FOLDS = 5` as a *placeholder* — valid for binary
  classification, but chosen with no evidence about the test split.
- Task: binary classification. Metric: **ROC-AUC** between predicted probability
  and observed target (`cv.score` -> `roc_auc_score`, greater is better).
- Why it is open: the test set has not been released, so how it was built is
  unknown. Guessing costs more than waiting — the scheme freezes the moment the
  first run writes an `oof_*.npy`.

### Settle this before the first `make baseline`

Load train and test and **measure**, do not assume:

1. **Row granularity** — is `ID_COL` unique in train? If a customer appears in
   several rows, random folds leak that customer across the split.
2. **Time** — is there a date column? Compare `train[TIME_COL].max()` against
   `test[TIME_COL].min()`. If test starts after train ends, the split is
   temporal and random folds are measuring the wrong thing.
3. **Entity overlap** — the share of test customers that never appear in train:
   `(~test[cust].isin(train[cust])).mean()`. This is the number that separates
   `group` from `time` from `time_coldstart`.

Then set `CV_SCHEME` from what you measured:

| Measurement | `CV_SCHEME` |
|---|---|
| one row per customer, no time signal | `stratified` |
| test is a later time window | `time` (set `TIME_COL`) |
| whole customers held out, several rows each | `group` (set `GROUP_COL`) |
| later window **and** customers unseen in train | `time_coldstart` (set both; calibrate `COLD_FRACTION`) |

Record the actual numbers here — row counts, date ranges, measured overlap —
not the assumption. Then note the consequences: do the folds partition the rows?
Are any rows never validated? Does the OOF over- or under-represent a slice of
test?

- **Frozen once decided.** Changing it invalidates every saved `oof_*.npy`; the
  fingerprint in `experiments.csv` is what enforces this.

## Runs

### `<run_id>` — one-line summary

- **Hypothesis:** what you expected and why.
- **Change:** exactly one thing, vs. which previous run.
- **Result:** CV x.xxxxxx (prev x.xxxxxx), LB x.xxxxx.
- **Read:** what the number actually licenses you to conclude. Is the difference
  bigger than fold-to-fold noise? Do the per-fold signs agree?
- **Keep / drop:** and what it implies for the next run.

## Dead ends

Anything tried that did not work, so it is not tried twice. One row each.

| What | Why it failed | Run |
|---|---|---|
| | | |
