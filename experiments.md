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

### `0904_202324_lgbm` — four engineered features, all at once

- **Hypothesis:** notebook 02 concluded the model needs columns that say
  something the raw ones do not. Four candidates, chosen to be non-redundant
  with `contract`: `charges_ratio` = `totalcharges / (tenure * monthlycharges)`
  (the residual against the bill the current rate implies), `avg_monthly_spend`
  = `totalcharges / tenure` (realised rate vs. current rate), `n_addons` (the
  six optional services as a 0-6 count), and `tenure_bucket` (cut at the
  contract boundaries 12 and 24, left as a *category* so LightGBM can take a
  subset split — as an integer it would only reproduce splits it can already
  make on `tenure`).
- **Change:** 19 -> 23 features, vs. `0902_232045_lgbm`. Model, params and folds
  identical; fingerprint still `13feb807`.
- **Result:** CV 0.916402 (prev 0.916296), **+0.000106**. Positive in 4/5 folds
  (fold 4: -0.000002, a tie). Paired bootstrap on the pooled OOF, 400 resamples:
  mean +0.000105, SD 0.000045, 95% interval [+0.000025, +0.000186], B ahead in
  395/400.
- **Read:** real, and small. The interval excludes zero, but the bootstrap holds
  the fitted models fixed and so misses refit noise; against the ~+/-0.00007
  total (0.000053 evaluation + 0.000052 seed, in quadrature) this is only ~1.5x.
  Four features bundled into one run is one measurement, not four — and the gain
  table put `avg_monthly_spend` 6th and `charges_ratio` 9th while `n_addons` sat
  second from last. That is a hypothesis about *which* of the four earned it, so
  it was worth the 90 seconds to ablate rather than guess.
- **Keep / drop:** superseded by the ablation below. OOF kept as blend material.

### `0904_202729_lgbm` — the same, minus `n_addons` and `tenure_bucket`

- **Hypothesis:** the two ratios carry the whole gain; the count and the bucket
  are redundant with columns the model already has.
- **Change:** dropped `n_addons` and `tenure_bucket`, 23 -> 21 features, vs.
  `0904_202324_lgbm`.
- **Result:** CV 0.916391. Against the **baseline**: **+0.000095, positive in
  5/5 folds**, bootstrap mean +0.000094, 95% interval [+0.000013, +0.000179],
  393/400. Against the **four-feature run**: -0.000010 — i.e. the two dropped
  features were worth +0.000010, 95% interval [-0.000032, +0.000057], ahead in
  only 274/400 resamples.
- **Read:** the two ratios are the result and they pass the cheap test cleanly —
  5/5 same-sign is the version of this gain worth believing, and it is *more*
  convincing than the larger 4/5 number from the bundled run. The other two
  features contribute +0.00001, which is a fifth of the seed spread: not a small
  effect, no effect. Dropping them costs nothing and removes two columns.
- **Keep / drop:** **keep the two ratios** — this is the new reference run.
  `features.py` now builds exactly these. Drop the count and the bucket.
- **Caveat, unresolved:** +0.000095 is still only ~1.4x the +/-0.00007 paired
  noise. 5/5 folds and 393/400 resamples say the sign is right; the magnitude is
  worth about a thousandth of nothing. Do not expect to see it on the public LB
  — it is well inside LB sampling noise. It is banked because it is free and it
  is in the right direction, not because it moves the standings.

## Dead ends

Anything tried that did not work, so it is not tried twice. One row each.

| What | Why it failed | Run |
|---|---|---|
| Assigning the submission column by `C.TARGET` | `sample_submission.csv` is read raw and keeps `Churn`; `C.TARGET` is lower-cased `churn`, so the assignment appends a column and leaves the real one all-zeros. Uploads cleanly, scores 0.5. | pre-baseline |
| `n_addons` — the six optional services as a 0-6 count | +0.000010 on top of the two charges ratios, 95% interval [-0.000032, +0.000057] spanning zero, ahead in 274/400 resamples. Second from last on gain. The six columns are already in the model individually; summing them says nothing they do not. | 0904_202324 vs 0904_202729 |
| `tenure_bucket` — `tenure` cut at 12/24 as a category | Same measurement as above; the two were dropped together. A bucketing of a numeric column the model already holds is the "rescaling of one column" case notebook 02 warned about — the categorical framing buys a subset split the tree cannot otherwise express, and that turned out to be worth nothing. | 0904_202324 vs 0904_202729 |
