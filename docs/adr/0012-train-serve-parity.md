# 0012 — Train/serve dtype parity, asserted bitwise

**Status:** accepted, 2026-10-24

## Context

The highest-risk part of productionising a model is not the serving stack, it is
that the serving path computes features *slightly* differently from training.
Nothing fails. The numbers are just quietly wrong.

This repo had three such traps, all measured:

1. **`utils.reduce_memory` casts every float to float32**, and `data.load`
   applies it *before* `features.add_telco_features` computes the two ratios.
   Computing them from float64 operands instead changes **31.9% of
   `charges_ratio`** values by one ulp — `60.1` is stored as `60.099998474121094`.
2. **`data.basic_clean` ends in `drop_duplicates()`.** Verified: three identical
   customers in, one row out. In a batch endpoint the response comes back
   shorter than and misaligned with the request.
3. **`features.as_categorical` rebuilds category lists from the frame it is
   given.** LightGBM encodes categoricals as integer codes, so a request scored
   through `build_features` would get codes derived from whatever levels that
   batch happened to contain. The real lists are first-appearance order over the
   original `concat(train, test)` — an artifact of raw row order, recorded
   nowhere until the manifest.

## Decision

One inference path, `src/predict.py`, which **reuses the training functions in
the training order** — `normalise_columns` → `reduce_memory(verbose=False)` →
`add_telco_features` → select the manifest's feature order → apply the
manifest's category lists — and calls neither `basic_clean` nor
`build_features`.

Then assert it: `tests/test_predict.py` compares the serving design matrix
against the training one column by column, and scores all 254,655 real test rows
through the bundle expecting `test_*.npy` back **bitwise**.

Bitwise, not `allclose`, on purpose. A tolerance cannot distinguish a real
feature bug from rounding, and bitwise is achievable here because float32
arithmetic is elementwise — batch size cannot change a value.

Holding that line turned up one more thing. `train.py` accumulates
`test_pred += p_k / len(folds)`; `predict_raw` first used `sum(w * p_k)` with
`w = 0.2`. `1/5` has no exact binary representation, so `p/5 != 0.2*p`, and the
two diverged on **79,690 of 254,655 rows at max |Δ| 4.4e-16**. Twelve orders of
magnitude below anything that matters — and exactly the kind of difference that
makes you weaken an assertion instead of understanding it. `predict_raw` now
divides the same way when weights are uniform.

## Azure equivalent

None, and that is the point: this is a practice, not a product. The nearest
*tooling* answers are a feature store (Azure ML Managed Feature Store), which
removes the duplication by making one definition serve both paths, and MLflow
`pyfunc`, which makes preprocessing travel with the model. Both reduce the
surface; neither removes the need for the test.

## Consequences

- Any future edit to `add_telco_features` or `reduce_memory` must keep parity —
  which is why `predict.py` *calls* them rather than copying them, and why the
  test exists.
- `CLAUDE.md` records both invariants so a future session does not reintroduce
  them.
- **How big was the risk really?** At the prediction level the float32 ordering
  bug moves **3 in 50,000 predictions, max |Δp| 0.0025** — LightGBM's 255-bin
  histogram absorbs 1-ulp input differences unless a value straddles a bin
  boundary. Small. The fix is free, so it is a parity-test target rather than a
  hazard; the duplicate-row bug was the genuinely serious one.
