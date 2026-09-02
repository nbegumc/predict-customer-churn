# Learnings

Concepts worth keeping, written down as they come up. `experiments.md` records
what was tried on *this* competition; this file records what was *understood*,
so it stays useful after the leaderboard closes.

---

## Univariate AUC — reading the "which features carry signal" table

**Where it came up:** the last cell of `notebooks/01_eda.ipynb` section 1
produced this, and the summary sentence was "signal is concentrated in `tenure`
(0.79), `contract` (0.79), `paymentmethod` (0.78) and `internetservice` (0.77);
`gender` and `phoneservice` are at chance."

### The mental model first

Take one feature. Ignore every other column. Ask:

> If I pick one customer who churned and one who didn't, at random, how often
> does this single feature rank the churner as the more likely churner?

That fraction **is** the AUC. It is not a metaphor for AUC or an approximation
of it — the probabilistic interpretation is exactly what ROC-AUC computes.

So the scale reads:

| AUC | Meaning |
|---|---|
| 0.50 | coin flip — the feature orders churners and non-churners no better than chance |
| 0.79 | in 79% of random churner/non-churner pairs, this feature ranks them correctly |
| 1.00 | perfect separation |

`tenure = 0.79` therefore means: pick a churner and a non-churner at random, and
79% of the time the churner has the lower tenure. `gender = 0.504` means gender
tells you essentially nothing — it is 0.4pp away from a coin flip, which at
n=594k is a detectable difference and a useless one.

### Why AUC and not correlation

Two reasons, both practical:

1. **It is the competition metric.** Scoring features in the same units the
   leaderboard uses means a feature's univariate number is directly comparable
   to the model's OOF number. `tenure` alone gets 0.79; the full model reaches
   ~0.91. That gap is what all the other features plus their interactions are
   worth.
2. **It only depends on rank order,** so it is immune to monotone transforms.
   `tenure`, `log(tenure)` and `tenure²` all score identically. Correlation
   would not, and you would waste time chasing a "better" transform that a
   tree-based model cannot even see the difference in.

### The two traps in that table

**Trap 1 — categoricals are scored optimistically.** A raw category like
`contract` has no numeric order, so there is nothing to rank by. The notebook
gives it one by mapping each level to its own churn rate, computed on the same
rows it then scores:

```python
s = train[c].map(y.groupby(train[c]).mean())   # in-sample rate
roc_auc_score(y, s)
```

That is target leakage in miniature — the score is fitted and evaluated on the
same data. With 2–4 levels and ~119k rows per level the level means are
essentially exact, so the bias here is negligible. **On a high-cardinality
column it would not be.** A 5,000-level id column scored this way can post an
AUC near 1.0 while carrying no generalisable signal at all, because each level
is memorising its own handful of rows. If a column like that ever appears, score
it out-of-fold or not at all.

**Trap 2 — univariate AUC is not feature importance.** It is a *marginal*
measure: each feature judged alone. It systematically misreads two situations.

- **Redundancy.** `contract` (0.79), `paymentmethod` (0.78) and
  `internetservice` (0.77) each look strong, but they overlap heavily —
  month-to-month customers on electronic check with fibre are largely the same
  people. Their *combined* contribution is far less than the sum of the three.
  This is exactly why the README warns that two boosting models on the same
  features correlate ~0.9 and blend to nearly nothing: correlated inputs, and
  therefore correlated errors.
- **Interaction.** A feature can score 0.50 alone and still be valuable, if it
  only matters conditional on another feature. Univariate AUC cannot see this by
  construction. So a low score is **weak evidence to drop a feature, never
  proof.** `gender` at 0.504 is a reasonable drop candidate; it is not a
  demonstrated one until a run without it scores the same.

### What it is actually good for

- **A sanity check on the data.** A feature with AUC 0.99 is usually a leak, not
  a gift — a column that encodes the answer. Nothing here is close, which is
  itself a useful negative result.
- **A prior on where to spend feature-engineering effort.** Ratios and buckets
  built on `tenure`, `contract` and the charges columns are more likely to pay
  than anything derived from `gender`.
- **A floor to beat.** If a 5-fold LightGBM cannot clear 0.79, something is
  wrong with the pipeline, not with the model.

It is a cheap first look, not a decision procedure. The decision procedure is a
logged run through `src.train` with one change and a `--notes`.

### Related, worth learning next

1. **Permutation importance** — the multivariate counterpart. Shuffle one column
   in the validation fold and measure how far AUC falls. It accounts for
   redundancy, because shuffling one of two correlated columns barely hurts.
2. **SHAP values** (`shap` is already a dependency) — per-row attributions, so
   you can see *where* a feature matters, not just how much.
3. **Why AUC is threshold-free**, and when that is the wrong property to want:
   AUC only grades ranking. If the business question is "which 5% do we call?"
   then precision@k or a calibrated probability matters more, and a model can
   win on AUC while being badly calibrated.

---

## Adversarial validation — using AUC to measure the train/test difference

**Where it came up:** `notebooks/01_eda.ipynb` section 2.3. It is the
measurement that settled `CV_SCHEME`, so it is worth understanding properly
rather than as a recipe.

### The mental model

You want to know whether the test set is *like* the training set. There is no
direct way to ask that — but there is an indirect one:

> Throw away the real target. Label every train row `0` and every test row `1`.
> Fit a model to tell the two apart. **How well it does is how different they
> are.**

If the organisers split the data at random, no model can beat a coin flip at
this, because there is nothing to find: AUC ≈ 0.5. If they took the last three
months as test, or held out whole customers, or sampled non-uniformly, then some
combination of features carries a fingerprint of that choice and the classifier
finds it: AUC climbs.

This is the univariate AUC idea from the section above, generalised — same
question ("how well does this separate two classes?"), but *all features at
once*, and with a different pair of classes.

### The recipe

```python
X = pd.concat([train[features], test[features]], ignore_index=True)
y = np.r_[np.zeros(len(train)), np.ones(len(test))]     # 0 = train, 1 = test

oof = np.zeros(len(X))
for tr_i, va_i in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y):
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63)
    m.fit(X.iloc[tr_i], y[tr_i])
    oof[va_i] = m.predict_proba(X.iloc[va_i])[:, 1]      # out-of-fold, always

adv_auc = roc_auc_score(y, oof)
```

Three details are not optional:

**Score it out-of-fold.** In-sample predictions from a 300-tree model will
separate anything, including pure noise. The whole quantity of interest is
whether the difference *generalises*, so it has to be measured on rows the
adversarial model did not see. This is the same discipline as the real CV.

**Drop the target.** It does not exist in test. Including it would make the task
trivially separable via its own missingness.

**Drop the row identifier — this one bites.** In this competition `id` is a
contiguous counter: train is `0…594,193`, test is `594,194…848,848`. A single
split at 594,193.5 separates the two files perfectly. Measured:

| Features given to the adversarial model | AUC |
|---|---|
| 19 real features **+ `id`** | **0.999991** |
| 19 real features | **0.511677** |

The first number is not a finding about the data. It is the model reading the
row number off the file — and it looks exactly like catastrophic drift, so it
will send you rebuilding a CV scheme you never needed. Anything that encodes
*which file a row came from* (id, index, filename, a row-order artefact) must
come out first.

### Reading the number

| AUC | Reading |
|---|---|
| ≈ 0.5 | exchangeable — the split is random, ordinary K-fold mirrors it |
| 0.5–0.6 | mild shift; usually ignorable, worth knowing where it lives |
| 0.6–0.8 | real selection mechanism — find it and reproduce it in the folds |
| > 0.9 | either severe drift, or (far more often) you left an identifier in |

**Our result: 0.5117.** With 594,194 train and 254,655 test rows, the
Hanley–McNeil standard error of an AUC at 0.5 is **0.00068**, so 0.5117 sits
about 17 SE above chance. (That formula treats the scorer as fixed, and here it
was fitted, so read it as "clearly above 0.5" rather than as an exact z-score.)

That combination — **statistically unmistakable, practically nothing** — is the
normal state of affairs at n ≈ 850k, and it is the thing to internalise. At this
sample size almost any difference is "significant". The question that matters is
never *is it non-zero*, it is *is it big enough to change a decision*. A
discriminator that is right on 51.2% of pairs cannot meaningfully reweight
anything, so the answer is no, and `stratified` folds stand.

### Where a small difference lives

The adversarial model's feature importances say *which* columns carry the
separation — this is the technique's main practical use, not the headline AUC.
Ours (share of total gain):

```
monthlycharges  28.0%    gender             3.4%
tenure          21.2%    dependents         2.7%
totalcharges    18.6%    paperlessbilling   2.6%
```

Read that with care. The three continuous columns dominate, but partly for a
mechanical reason: a continuous column offers thousands of candidate split
points, so a tree can always carve out thin slivers that happen to lean one way,
while a 3-level categorical offers two. **Gain concentrating on continuous
features is the expected shape of a null result**, not evidence of drift in
those columns. Given AUC 0.5117, the honest summary is "there is no meaningful
shift, and what little the model found is in the continuous columns."

### What you do when the AUC *is* high

1. **Read the importances** to find the responsible column, as above.
2. **Decide whether it is a leak or a real shift.** A column that identifies the
   file (an id, an ingestion timestamp) gets dropped. A column that genuinely
   differs (a price that rose over time) is a shift to reproduce, not remove.
3. **Rebuild the folds to mimic it** — this is the point. High adversarial AUC
   driven by a date column means `CV_SCHEME = "time"`; driven by an entity means
   `group`. The adversarial model tells you *that* test was selected somehow;
   the importances tell you *by what*.
4. **Or weight the training rows** by `p/(1-p)` from the adversarial model, so
   the fit leans toward the training rows that look most like test. Worth
   knowing; rarely worth doing below AUC ≈ 0.7.

### The limitation that matters

Adversarial validation compares **P(X)** — the distribution of the features. It
says nothing about **P(y | X)**, the relationship between features and target,
because it never looks at the target.

So AUC ≈ 0.5 means "the test rows look like the training rows". It does *not*
prove the churn mechanism is unchanged. Concept drift with a stable covariate
distribution is invisible to this test, and it is not a hypothetical: the same
customers, same demographics, new retention policy. Here the split is a random
partition of one synthetic table, so P(y | X) is identical by construction and
the distinction does not bite — but on a real temporal competition, a 0.5
adversarial AUC is not a clean bill of health.

**Why it is still the decisive test.** Every other check in the EDA looks for a
mechanism you already suspected: you check for a date column because you
thought of dates. Adversarial validation is the only one that catches a
selection rule you never thought to look for, because you never have to name it.
That is why it belongs in the first hour of every competition — and it needs no
target, so it runs the moment the data lands.

## Judging whether a CV difference is real

**Where it came up:** deciding what counts as an improvement between two runs.

### The question

You run model A and get OOF AUC 0.9100. You change one thing, run model B, and
get 0.9120. **Is B actually better, or did it get a lucky draw of customers?**

Your score is measured on a finite set of people. Had you drawn a different
594k customers, the number would have landed somewhere slightly different. That
wobble is the noise, and the whole question is whether your +0.002 is bigger
than it.

### Why the obvious rule is wrong

The obvious rule is "measure the noise, ignore anything smaller". That rule
throws away real results here, because it measures the wrong noise.

Both runs are scored on **exactly the same rows in exactly the same folds** —
that is the entire reason `CV_SCHEME` is frozen. So both face the same draw of
customers. If fold 3 happens to hold a clump of unpredictable people, it drags
A down *and* B down. Subtract, and that shared bad luck cancels.

> Two runners. Timed on different days on different tracks, you need a big gap
> before you can say who is faster — wind and track conditions swamp it. Race
> them **side by side** and a small gap means something, because they shared the
> conditions. Frozen folds are racing side by side.

So the *difference between* two runs is measured far more precisely than either
run's own score. Below, 16x more precisely.

### How to calculate it — three ways

**1. A formula, no refitting.** Hanley-McNeil gives the standard error of an AUC
from the score and the class counts alone:

```
SE = sqrt( [ A(1-A) + (n+ - 1)(Q1 - A²) + (n- - 1)(Q2 - A²) ] / (n+ · n-) )
     where  Q1 = A/(2-A)    Q2 = 2A²/(1+A)
```

At fold 0 (A = 0.915, 26,763 positives, 92,076 negatives) this gives **0.00121**.
Cheap, but it assumes a distributional shape and comes out **conservative** —
measurement 2 shows by how much.

**2. Bootstrap the score.** Fit once, then resample the validation rows *with
replacement* 400 times and recompute the AUC on each resample. The spread of
those 400 numbers is the noise, assuming nothing:

```python
rng = np.random.default_rng(0)
boot = []
for _ in range(400):
    i = rng.integers(0, len(y_va), len(y_va))     # resample rows, same size
    boot.append(roc_auc_score(y_va[i], pred[i]))
print(np.std(boot))
```

Measured: **0.00085** — so the formula overstated it by about 1.4x.

**3. Bootstrap the *difference*.** The one that actually governs decisions. Same
400 resamples, but score *both* runs on each one and record `B - A`:

```python
i = rng.integers(0, len(y_va), len(y_va))
d.append(roc_auc_score(y_va[i], pred_b[i]) - roc_auc_score(y_va[i], pred_a[i]))
```

### Worked example on this data

Two real runs on fold 0, LightGBM at 200 rounds, differing only in `num_leaves`:

| Quantity | Value |
|---|---|
| Run A (`num_leaves=63`) | 0.914754 |
| Run B (`num_leaves=127`) | 0.915242 |
| Observed difference B-A | **+0.000488** |
| SD of A's AUC alone (bootstrap) | 0.000849 |
| SD of B's AUC alone (bootstrap) | 0.000846 |
| **SD of the paired difference** | **0.000053** — 16x smaller |
| 95% interval on the difference | [+0.00039, +0.00059] |
| Resamples where B beat A | **400 / 400** |

A gain of +0.0005 — which the naive "must exceed +/-0.0012" rule would have
discarded — is in fact unambiguous.

### The caveat, also measured

That bootstrap holds the two *fitted* models fixed. It answers "would the score
change if I evaluated on different customers?" and says nothing about
**retraining** randomness. So measure that separately: same config, four seeds.

```
0.914921   0.914875   0.914804   0.914828      SD = 0.000052
```

Seed noise (0.000052) turns out to be the same size as the paired evaluation
noise (0.000053). Added in quadrature, a paired comparison carries roughly
**+/-0.00007** in total. The observed +0.00049 is about 7x that, so it is real.

Two lessons in that. A comparison has **two** noise sources, and the one you
forget is the one from refitting. And a difference smaller than the seed spread
is not a result at all — if a change moves the score less than re-running the
same code with a different seed does, you have measured the seed.

### The everyday rule

The bootstrap needs saved predictions and a few minutes. The five per-fold
scores are printed free on every run, so use the cheap version of the same
reasoning:

> **Do the per-fold differences agree in sign?**

+0.0003 that is positive in 5/5 folds is a real, small gain — under a true null
each fold is a coin flip, and five landing the same way happens about 1 time in
16. +0.002 that is positive in 3/5 folds is one lucky fold dragging an average.
A small consistent gain beats a large erratic one.

Caveat on that shortcut: the folds share training data, so they are not fully
independent and 5/5 is not exactly a 1-in-16 p-value. Treat it as a strong hint;
reach for the paired bootstrap when a decision actually turns on it.

This is why the run template in `experiments.md` asks "do the per-fold signs
agree?" instead of naming a threshold.
