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
- **LB, recorded after the fact (2026-09-04):** 0.91393, up +0.00004 on the
  baseline's 0.91389 for a CV move of +0.000095. That is not confirmation —
  +0.00004 is far inside LB sampling noise — it is only the absence of
  contradiction. The number that carries information is the **gap**: -0.00241
  then -0.00246, i.e. steady. A constant CV-LB offset is what "CV tracks LB"
  looks like; it is the gap *moving* that would mean the folds are wrong.

### `0904_210834_catboost` — a second booster, same 21 features

- **Hypothesis:** blend material. The README is explicit that this is the *weak*
  version of idea 3 — two boosting libraries on identical features are supposed
  to correlate ~0.9 and blend to almost nothing, and what earns weight is a
  different estimator *shape*. Run it anyway, because "almost nothing" is a
  prediction and this repo measures predictions.
- **Change:** model lgbm -> catboost (defaults: depth 8, lr 0.05, <=3000 rounds,
  early stopping 200), vs. `0904_202729_lgbm`. Same features, same folds.
- **Result:** CV 0.916274, i.e. **-0.000117 behind LightGBM solo**. Per fold it
  loses on 0, 2, 4 and ties on 1, 3. ~5.5 min/fold against LightGBM's 18s.
- **Read:** solo score is the wrong yardstick here and the notebook said so in
  advance. Recorded, kept, judged below.
- **Keep / drop:** keep as blend material. Not a candidate on its own.

### `0904_210905_blend` — lgbm + catboost

- **Hypothesis:** per the README, near-zero. The correlation came out at
  **0.997** — *higher* than the ~0.9 the README predicted — which made near-zero
  look like a safe bet.
- **Change:** weighted average of `0904_202729_lgbm` and `0904_210834_catboost`,
  weights optimised on OOF by `src.blend`.
- **Result:** CV **0.916567**, +0.000176 over the best single model. Positive in
  **5/5 folds**, paired bootstrap mean +0.000172, SD 0.000023, 95% interval
  [+0.000130, +0.000216], ahead in **400/400** resamples. Nearly **twice** the
  feature-engineering gain, from two models that correlate 0.997.
- **Read — the blend weight is fitted on the same OOF it is scored on**, so the
  headline number had to be checked before it was believed. Three ways, all
  agreeing: weight fitted on all OOF 0.916567; weight fitted on 4 folds and
  applied to the 5th (nested, honest) 0.916565; flat 50/50 with nothing fitted
  at all 0.916563. A spread of 0.000004 — the optimism is nil, because one
  weight on a 2-simplex fitted to 594k rows has nowhere to overfit. The
  per-fold weights are stable too (0.539-0.579). Rank-averaging scores
  identically to probability-averaging (0.916563), as it must at r=0.997.
- **LB (2026-09-04):** **0.91401**, the best on the board so far, and the third
  scored run — so `src.log_lb` now prints a rank correlation:
  spearman(cv, public_lb) = 1.000 over 3 runs. Do not read much into that. With
  three points, perfect rank agreement happens 1 time in 6 under a pure null.
  The real argument for trusting CV is in the section below, and it is stronger.
- **Keep / drop:** keep. Best CV *and* best LB in the repo, and the honest
  estimate is the same as the fitted one.

### The lesson: 0.997 correlation did not mean 0.997 of the same answers

Worth stating separately, because it contradicts the diagnostic the README
recommends and it will come up again on every future blend.

The 0.997 is a Pearson correlation of two prediction *vectors*, and it is
dominated by the easy rows. 54% of the data sits at a mean prediction below 0.1
where both models say "will not churn" and agree trivially. Split by confidence
band, the correlation inside each band is only 0.89-0.96.

The metric is AUC, which is a *ranking* statistic: it is the fraction of
churner/non-churner pairs put in the right order. So the diagnostic that matches
the metric is pairwise ranking disagreement. Measured on 200,000 random pairs:
LightGBM ranks 0.9162 correctly, CatBoost 0.9161 — and **they disagree on 1.09%
of the pairs**. That ~1% is the raw material the blend trades on, and it is
completely invisible in the 0.997.

**Use ranking disagreement, not prediction correlation, to judge blend material
for a ranking metric.** The README's headline number was the wrong instrument;
its *conclusion* — that two boosters on the same features are worth little — is
still right in absolute terms, since +0.00018 AUC will not move the standings.
A structurally different estimator is still the thing to reach for.

### What the public LB can and cannot resolve — measured 2026-09-04

Three scored runs, and the gap widens monotonically:

| run | CV | LB | gap | CV delta | LB delta |
|---|---|---|---|---|---|
| `0902_232045_lgbm` | 0.916296 | 0.91389 | -0.00241 | — | — |
| `0904_202729_lgbm` | 0.916391 | 0.91393 | -0.00246 | +0.000095 | +0.00004 |
| `0904_210905_blend` | 0.916567 | 0.91401 | -0.00256 | +0.000176 | +0.00008 |

Both improvements landed on the LB at roughly **45% of their CV size**, twice
running. That looks like a pattern — CV gains only half-realised on test — and
it would be worrying if it were real, so it was tested rather than eyeballed.

The two submissions in each pair come from the **same fitted models**: the same
booster that wrote `oof_*.npy` wrote `test_*.npy`. So comparing a pair's LB
delta against its CV delta carries no refit or seed term, only evaluation noise
on two disjoint row samples. The OOF paired-bootstrap SD is known for both
comparisons (0.000044 and 0.000023), and evaluation noise scales as
`1/sqrt(n)`, so the LB-side SD follows from the public set's size.

The public fraction is not published, so all three plausible values were checked:

| public = | SD(LB delta), blend vs ratios | z of the shortfall |
|---|---|---|
| 10% of test (25k rows) | 0.000111 | -0.85 |
| 20% of test (51k rows) | 0.000079 | -1.17 |
| 50% of test (127k rows) | 0.000050 | -1.75 |

For the ratios-vs-baseline pair the z runs -0.25 to -0.53. **Nothing here is
significant at any plausible public size.** The "45% carryover" is not an
effect; it is two draws from a distribution wide enough to contain them.

**The operational conclusion, which is the point of this section.** At a 20%
public set the paired noise on an LB delta is ~0.00008, and the changes being
made move the score by 0.00010-0.00018. The instrument is the same size as the
thing being measured, so the LB cannot resolve a change of this magnitude —
one submission is worth roughly one bit, and that bit is mostly noise. The
absolute score is worse still: its SE is ~0.0019, twenty times any delta so far.

So: **select on CV, and stop spending submissions to confirm changes this
small.** That is what `log_lb` advises too, but its reason (a 3-point rank
correlation) is much weaker than this one. Submissions are worth spending on
changes large enough to see above ~0.0002, or on final selection. Revisit this
if a run ever moves CV by enough for the LB to have an opinion.

### `0905_015118_linear` — logistic regression, the first structurally different model

- **Hypothesis:** README idea 3. Two boosters on the same features disagreed on
  only 1.09% of ranked pairs, and the note above concluded that a *structurally
  different* estimator is the thing to reach for. A linear model in the one-hot
  basis is the cheapest thing that qualifies and the sharpest contrast
  available: a booster is an interaction machine, a linear model is purely
  additive and cannot express a single interaction. It should therefore dissent
  on far more than 1.09% of pairs, and that dissent should be worth something.
  The first half of that was right and the second half was wrong.
- **Change:** new model `linear` (`src/models/linear.py`) — L2 logistic
  regression, C=1.0, on one-hot categoricals plus standardised numerics, fitted
  inside the fold. Same 21 features and same folds as `0904_202729_lgbm`;
  fingerprint still `13feb807`. Preprocessing lives in the model rather than in
  `features.py` because it is model-specific — LightGBM wants native
  categoricals and a one-hot expansion would hurt it, the same reason CatBoost
  owns its NA level.
- **Result:** CV **0.907953** (folds 0.907465 / 0.908911 / 0.908136 / 0.909146 /
  0.906143), i.e. **-0.008438 behind LightGBM**, losing on 5/5 folds. ~2s per
  fold against CatBoost's 5.5 minutes.
- **Read:** the solo gap is the whole story of this run, and it is 100x every
  feature-engineering move made so far. As blend material judged the way the
  note above says to judge it, this looked excellent — see the next two entries
  for why it was not.
- **Keep / drop:** keep the code and the OOF. Not a candidate on its own, and —
  unlike CatBoost — not a candidate as a blend member either.

### `0905_015321_blend` — lgbm + catboost + linear

- **Hypothesis:** the linear model dissents ~3x as often as CatBoost does, so on
  the diagnostic this file recommends it is 3x the raw material. Expect a gain
  larger than the +0.000176 the two boosters produced.
- **Change:** added `0905_015118_linear` to the blend, vs `0904_210905_blend`.
- **Result:** CV **0.91656693** against the two-model blend's **0.91656687** —
  **+0.00000005**, positive in 2/5 folds. The optimiser gave the linear model a
  weight of **0.0044** and there is no weight at which it does better: a scan
  from w=0 to 0.2 peaks at w=0.005 for +0.0000000 and then falls monotonically
  (w=0.05 -> 0.916550, w=0.10 -> 0.916495, w=0.20 -> 0.916273).
- **Read:** this is not a small gain, it is no gain. +0.00000005 is a
  *thousandth* of the +/-0.000052 seed spread. The ranking disagreement was
  measured correctly and was real — 3.65% of pairs against CatBoost's 1.10%,
  and that 1.10% reproduces the 1.09% recorded above, which is the check that
  the new implementation in `blend.py` measures what the old ad-hoc one did.
  Three times the disagreement bought exactly nothing.
- **Keep / drop:** drop the linear model from the blend; `0904_210905_blend`
  remains the submission. Kept in `experiments/` as blend material only in the
  sense that everything is kept.

### The correction: disagreement has to be *balanced*, not just large

The lesson two sections up says to judge blend material by ranking disagreement
rather than prediction correlation. That is still right as far as it goes, and
followed on its own it would have predicted the linear model was three times the
blend material CatBoost was. It is worth nothing. The rule was half a rule.

Splitting each disagreement by *who was right* is what finishes it. On 2,000,000
sampled (churner, non-churner) pairs:

| pair | disagree on | A right | B right | blend gain |
|---|---|---|---|---|
| lgbm vs catboost | 1.10% | 50.4% | 49.6% | +0.000176 |
| lgbm vs linear | 3.65% | **61.5%** | **38.5%** | +0.00000005 |
| catboost vs linear | 3.59% | 61.6% | 38.4% | — |

CatBoost's dissent is a coin flip, so averaging recovers information from it.
The linear model's dissent is 62/38 against itself — when it breaks from the
boosters it is usually simply wrong, and a weighted average of a good ranker and
a worse one that disagrees *predictably* has nothing to recover.

**And this is not a coincidence, it is an identity.** AUC is the probability of
ordering a (positive, negative) pair correctly, so for any two models

    AUC_A - AUC_B = P(A right, B wrong) - P(B right, A wrong)

Both terms on the right sum to the disagreement `d`, so the balance is forced:

    P(A right | they disagree) = 1/2 + (AUC_A - AUC_B) / (2d)

Check it: CatBoost, gap 0.000117 and d 0.0110 -> 50.5% (measured 50.4%). Linear,
gap 0.008438 and d 0.0365 -> 61.6% (measured 61.5%). A model's solo deficit is
*exactly* its net loss on the pairs where it dissents; there is nowhere else for
a deficit to live.

**The operational form.** The quantity to look at before blending is the ratio
`gap / (2 x disagreement)` — how far from a coin flip the dissent is. It is
free: the gap is the difference of two CV scores already in `experiments.csv`,
and `src/blend.py` now prints both halves on every run — the disagreement
matrix, and a `blend candidacy` table splitting each candidate's dissent
against the best run by who is right. CatBoost scores 1.1% away from balanced,
the linear model 12%. Large disagreement is
necessary and not sufficient; it has to be large *and* near 50/50, which means
**a blend member must be both different and competitive.** Being different is
the cheap half.

So solo score is not the wrong yardstick for a blend member after all, which is
what the CatBoost entry above assumed. It is the wrong yardstick only within the
band where the models are close enough that the dissent stays balanced. CatBoost
sits 0.0001 behind and qualifies; the linear model sits 0.0084 behind and cannot
be rescued by any weight.

### `0905_020138_mlp` — the same idea, made competitive

- **Hypothesis:** the section above says a blend member must be different *and*
  competitive, and names the linear model's 0.0084 solo deficit as the reason
  its dissent was unusable. An MLP repairs exactly that half: it keeps the
  structural contrast — distributed representations rather than axis-aligned
  splits — while learning the interactions the additive model could not express,
  which is where the deficit came from. If the correction is right, a smaller
  gap should show up directly as more balanced dissent.
- **Change:** new model `mlp` (`src/models/mlp.py`) — `MLPClassifier`, hidden
  (128, 64), relu/adam, alpha 1e-4, batch 1024, on the *same* one-hot +
  standardised design matrix as the linear model (`preprocessor()` is now shared
  between them). Same 21 features, same folds, fingerprint `13feb807`.
  scikit-learn rather than torch: it is already a dependency and this is a
  594k x ~50 CPU problem.
- **Result:** CV **0.912762** (folds 0.912244 / 0.913611 / 0.912947 / 0.914082 /
  0.911562), **-0.003629 behind LightGBM**, still losing 5/5. 17-29s per fold,
  25-43 epochs.
- **Read:** the repair worked on its own terms — the gap more than halved,
  0.008438 -> 0.003629 — and the dissent duly became more balanced, 61.5% ->
  57.3%. Not balanced enough. Note the early-stopping caveat: sklearn stops on
  `.score()`, i.e. *accuracy*, which at a 22.5% base rate barely moves when a
  ranking change does not cross the 0.5 threshold. The patience was raised to 15
  on that argument rather than tuned against a fold, so this is an honest but
  not a squeezed number.
- **Keep / drop:** keep the OOF, drop as a blend member. See below.

### `0905_020201_blend` — lgbm + catboost + mlp

- **Hypothesis:** at 57.3% balance instead of 61.5%, expect a small positive
  gain — less than CatBoost's +0.000176, more than the linear model's zero.
- **Change:** swapped the linear model for the MLP, vs `0905_015321_blend`.
- **Result:** CV **0.916567**, weight on the MLP **0.0000**, and a weight scan
  from 0 to 0.5 in steps of 0.0025 finds **no weight that gains anything at all**
  (best +0.00000000). Identical to the two-model blend.
- **Read:** the prediction was wrong in the direction that matters. "More
  balanced" did not mean "worth something"; 57.3% is still predictably wrong
  dissent, and the optimiser is not being coy — there is genuinely nothing
  there. Two independent methods agree, Nelder-Mead over the full 3-simplex and
  the one-dimensional scan against a fixed two-model base.
- **Keep / drop:** drop. `0904_210905_blend` is still the submission.

### The bar: how competitive a different model has to be — measured 2026-09-05

Three candidates against the same reference (`0904_202729_lgbm`), 2,000,000
sampled pairs, weight scanned 0 to 0.5 against the fixed two-model blend:

| candidate | solo | gap | disagree | balance | identity says | best gain |
|---|---|---|---|---|---|---|
| catboost | 0.916274 | +0.000117 | 1.10% | 50.4% | 50.5% | *(already in)* |
| mlp | 0.912762 | +0.003629 | 2.40% | 57.3% | 57.6% | +0.00000000 |
| linear | 0.907953 | +0.008438 | 3.65% | 61.5% | 61.6% | +0.00000007 |

`balance = 1/2 + gap/(2d)` predicts the measured column to within 0.3pp in every
row, so the relationship is now checked on three models and not just derived.
Note the ordering: **disagreement rises as the model gets worse, and value falls
to zero.** Ranking disagreement read alone points at exactly the wrong candidate,
which is what makes the uncorrected version of the rule actively dangerous.

**Inverting it gives the bar.** For a candidate to dissent as evenly as CatBoost
does (50.5%), its solo score must satisfy `gap <= 0.01 x d`:

- the MLP, at d = 2.40%, would need solo **>= 0.916151** — it has 0.912762, short
  by 0.0034.
- the linear model, at d = 3.65%, would need **>= 0.916026** — short by 0.0081.

That bar is within **0.00024** of LightGBM. So on this dataset a structurally
different model has to be, to a first approximation, *as good as a tuned
booster* before its difference is worth anything, and neither of the two
non-tree shapes available is within 14x of that. Buying it with hyperparameter
tuning is possible in principle and is the thing the README ranks last, for
about this reason.

**Now enforced in the tool.** `src/blend.py` printed only the disagreement
matrix when this section was first written, which is the half that recommends
the worst candidate. It now prints the split as well, off a single shared pair
sample so the two can never quote different numbers for the same comparison,
and the `identity %` column shows `1/2 + gap/2d` beside the measurement so a
future run gets the check for free.

**Verdict on README idea 3: it does not pay here, and the reason is measurable
rather than a matter of taste.** The one blend that ever worked in this repo was
two boosters 0.000117 apart — which the README predicted would be worth almost
nothing, and which is in fact the *only* pair close enough to qualify. Idea 3
still stands as written for problems where a different shape is competitive;
this is a synthetic expansion of a small tabular table, where nothing that is
not a tree comes close. What generalises is the test, not the verdict — and the
test is two numbers that are already free on every run.

## Dead ends

Anything tried that did not work, so it is not tried twice. One row each.

| What | Why it failed | Run |
|---|---|---|
| Assigning the submission column by `C.TARGET` | `sample_submission.csv` is read raw and keeps `Churn`; `C.TARGET` is lower-cased `churn`, so the assignment appends a column and leaves the real one all-zeros. Uploads cleanly, scores 0.5. | pre-baseline |
| `n_addons` — the six optional services as a 0-6 count | +0.000010 on top of the two charges ratios, 95% interval [-0.000032, +0.000057] spanning zero, ahead in 274/400 resamples. Second from last on gain. The six columns are already in the model individually; summing them says nothing they do not. | 0904_202324 vs 0904_202729 |
| `tenure_bucket` — `tenure` cut at 12/24 as a category | Same measurement as above; the two were dropped together. A bucketing of a numeric column the model already holds is the "rescaling of one column" case notebook 02 warned about — the categorical framing buys a subset split the tree cannot otherwise express, and that turned out to be worth nothing. | 0904_202324 vs 0904_202729 |
| Logistic regression (L2, one-hot + standardised numerics) as a blend member | Structurally different as intended — it dissents on 3.65% of ranked pairs against CatBoost's 1.10% — but 0.008438 behind LightGBM solo, which forces the dissent to 62/38 against it. Added to the blend it is worth +0.00000005 and takes weight 0.0044; no weight in [0, 0.2] does better. Different is not enough, it also has to be competitive. | 0905_015118 vs 0905_015321 |
| MLPClassifier (128, 64) on one-hot + standardised numerics as a blend member | Fixed the linear model's problem and still failed. Gap to LightGBM more than halved (0.008438 -> 0.003629) and dissent duly rebalanced (61.5% -> 57.3%), but the blend weight is 0.0000 and no weight in [0, 0.5] gains anything. The bar implied by `balance = 1/2 + gap/(2d)` is a solo score of 0.916151 at its 2.40% disagreement — within 0.00024 of LightGBM — and it is 0.0034 short. | 0905_020138 vs 0905_020201 |
