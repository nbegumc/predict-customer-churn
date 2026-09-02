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

- CV scheme: **`stratified`, 5 folds — DECIDED and FROZEN (2026-09-02).**
  `src/config.py` already held this as a placeholder; `notebooks/01_eda.ipynb`
  measured the split and the placeholder turned out to be right. It is now
  evidence-backed, not a default.
- Task: binary classification. Metric: **ROC-AUC** between predicted probability
  and observed target (`cv.score` -> `roc_auc_score`, greater is better).
- `GROUP_COL` and `TIME_COL` stay empty. `COLD_FRACTION` is **not applicable** —
  see below, there is no entity to be cold.

### What was measured

Dataset: `playground-series-s6e3`, a synthetic expansion of the Telco churn
table. train 594,194 x 21, test 254,655 x 20 — test is exactly 30.0% of the
combined rows. 19 features: 4 numeric (`seniorcitizen`, `tenure`,
`monthlycharges`, `totalcharges`), 15 categorical at 2-4 levels each.

1. **Row granularity.** `id` is unique in both frames, contiguous
   (train 0-594,193, test 594,194-848,848), and the two ranges do not overlap.
   Test simply continues the train counter.
2. **Time.** No date column exists, and none can be derived. `id` order is the
   only possible carrier of time, and it does not carry it:
   `corr(id, churn) = -0.013` and `corr(id, tenure) = +0.019` inside train —
   detectable at n=594k, negligible in size — and the tenure trend *fails to
   continue into test* (`corr(id, tenure) = +0.005` there). The first 2-3 id
   ventiles sit slightly high (churn ~0.25 vs ~0.22 for the rest), which is a
   head-of-file artefact, not a temporal ordering.
3. **Entity overlap.** There is no `customerid` column. All 594,194 train
   feature vectors are distinct (0 duplicates), and 0.00% of sampled test
   feature vectors appear in train. There is no repeated entity anywhere, so
   `group` and `time_coldstart` are not merely unnecessary — there is nothing
   to group by.
4. **Adversarial validation** (LightGBM, train vs. test on features alone,
   5-fold): **AUC = 0.5117**. With ~849k rows the SE is ~0.0007, so this is
   measurably above 0.5 and practically indistinguishable from it. No
   categorical marginal differs by more than 0.8pp between the frames; every
   numeric mean is within 0.03 SD.
5. **Unseen levels.** None. Every categorical level present in test also occurs
   in train.
6. **Missingness.** Zero missing cells in either frame. `totalcharges` parsed
   cleanly as float — the classic Telco blank-string-at-tenure-0 trap is absent
   because there are no tenure-0 rows.

### The conclusion in one sentence

The test set is a random 30% slice of one synthetic table: one row per customer,
no time column, no repeated entity, no distribution shift.

Hence `stratified`. Stratified rather than plain `kfold` because the positive
rate is 22.5%; at 5 folds the difference is small, but stratifying removes
fold-to-fold class-balance variation as a noise source at no cost. Realised
folds: 118,838-118,839 rows each, positive rate 0.22520-0.22521 across all five.
Fold fingerprint `13feb807`.

### Consequences

- The folds **partition the rows**: every training row is validated exactly
  once, the OOF array has no NaNs, and `cv.add_fold_column` is valid here.
- OOF and test are drawn from the same distribution, so CV should track the
  public LB closely. A CV-LB gap would indicate a bug, not a shift.
- Noise, measured on fold 0 at AUC ~0.915 (bootstrap, 400 resamples; derivation
  and worked example in `learnings.md`): **+/-0.00085 on a single run's fold
  score**, but only **+/-0.000053 on a paired difference between two runs**, and
  **+/-0.000052 across seeds** for the same config. Runs share identical folds,
  so comparisons are paired and roughly **16x** more precise than either
  absolute score. (Hanley-McNeil predicts +/-0.0012 per fold — conservative by
  ~1.4x; prefer the bootstrap.)
- So the test is not one threshold. Cheap version, free on every run:
  **do the per-fold differences agree in sign?** +0.0003 positive in 5/5 folds
  is real; +0.002 positive in 3/5 is one lucky fold. And anything smaller than
  the ~0.00005 seed spread is not a result — it is the seed.

- **Frozen once decided.** Changing it invalidates every saved `oof_*.npy`; the
  fingerprint in `experiments.csv` is what enforces this.

## Name/dtype mismatches found by the EDA — fixed 2026-09-02

Three of them sat between the EDA and a working submission. All fixed; the folds
are unchanged by the fix (fingerprint still `13feb807`).

1. `config.TARGET` was `"Churn"`, which never matches a column — `basic_clean`
   lower-cases the headers, so `train[C.TARGET]` in `train.py` raised
   `KeyError`. Now `"churn"`.
2. The label was the strings `"Yes"`/`"No"`. `train.py` passed `y_raw` straight
   to LightGBM and to `roc_auc_score`, neither of which takes text. New
   `data.encode_target` maps it to int8 via `TARGET_MAP`, asserts nothing falls
   outside the map, and runs *after* `reduce_memory` — which would otherwise
   widen the int8 back to int32.
3. `load_sample_submission` does not call `basic_clean`, so the frame keeps
   `Churn` while `C.TARGET` is `churn`. `sub[C.TARGET] = test_pred` *added* a
   second column and left `Churn` all-zeros. `train.py` now writes whichever
   column is not `ID_COL`, so the header the competition expects survives
   whatever its casing.

The third is the dangerous one: the other two crash, this one succeeds and
silently spends a submission slot on a 0.5.

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
| Assigning the submission column by `C.TARGET` | `sample_submission.csv` is read raw and keeps `Churn`; `C.TARGET` is lower-cased `churn`, so the assignment appends a column and leaves the real one all-zeros. Uploads cleanly, scores 0.5. | pre-baseline |
