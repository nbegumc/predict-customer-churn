# 0002 — Serve the fold average, not a model refit on all of train

**Status:** accepted, 2026-10-24

## Context

`src/train.py --full-refit` writes two usable artifacts into one bundle:
`fold_average` (the five CV fold models, averaged) and `full_refit` (one model
fit on 100% of train at a fixed round count). Industry habit says refit: more
data, one small artifact, faster inference.

## Decision

Serve `fold_average`. `full_refit` stays in the bundle, reachable with one
environment variable.

**The deciding argument is not score, it is that `full_refit` has no score.**
A model fit on 100% of train has no out-of-fold rows, so the artifact cannot be
evaluated honestly. Only its *procedure* can — committing to a fixed round
count instead of early stopping — and that was measured: **−0.000034, negative
in 5/5 folds** (run `1024_143040_lgbm`). By this repo's own standard the sign is
the believable part; the magnitude sits under the ±0.000052 seed spread.

Meanwhile `fold_average` is **verifiable**. `test_*.npy` is the mean of the five
fold models, and `tests/test_predict.py` asserts that scoring
`data/raw/test.csv` through this strategy reproduces that array bitwise. The
served artifact is therefore provably the one behind public LB 0.91393.

The two barely differ in practice: spearman **0.99890** on 254,655 rows, top-10%
call-list overlap **97.2%**, and 0.001% of customers move more than one decile.
So the operational cost of choosing the verifiable one is ~3% of the top decile.

## Azure equivalent

**Azure ML Model Registry** versions either artifact identically, and managed
online endpoints serve either. Nothing about this decision is cloud-specific —
it is about what you can prove, not where you run it.

## Rejected

- **Refit on all data.** Standard, and unfalsifiable here.
- **Spending a submission to decide.** `experiments.md` → "What the public LB can
  and cannot resolve" established paired LB noise at ~0.00008 against changes of
  0.00010–0.00018. The instrument is the same size as the thing measured.
- **Serving the lgbm+catboost blend.** +0.000176 CV for ~100MB of image and a
  second artifact to version, against a leaderboard that cannot resolve it.

## Consequences

- 30MB and five boosters instead of 6MB and one. ~1–2ms per row against ~0.3ms;
  neither matters at this scale.
- The manifest's `strategies` key is a weighted member list, so promoting the
  blend later means appending a member, not rewriting the loader.
- **Revisit when:** cold-start latency or image size becomes a real constraint,
  or when a held-out slice exists that could score a refit honestly.
