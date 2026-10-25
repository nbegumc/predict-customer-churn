# 0003 — LightGBM native text plus a JSON manifest, not pickle

**Status:** accepted, 2026-10-24

## Context

`src/train.py` fitted five models per run and dropped all five. Nothing this
repo had ever trained could be loaded again. Persisting a model needs a format,
and persisting *enough to score a row the way CV scored it* needs more than the
model.

## Decision

`booster.save_model(..., num_iteration=best_iteration)` — LightGBM's native text
format — plus a `manifest.json` beside it.

Text over pickle because it survives a library upgrade, it diffs, and
critically **it independently records `feature_name()` and
`pandas_categorical`**. That turns the manifest into a cross-check rather than a
thing to trust: `load_bundle` asserts the two agree and refuses a bundle where
they do not.

The manifest carries what the model file cannot: the 21-name feature order, the
15 category lists, the dtype regime, resolved hyperparameters, per-fold stopping
rounds, the fold fingerprint, the OOF score distribution, and the git sha.

`strategies` maps a name to a **weighted member list** — `fold_average` is five
members at 0.2, `full_refit` is one at 1.0 — so both go through one code path at
inference, and a future blend is an appended member.

## Azure equivalent

**MLflow `pyfunc`** registered in Azure ML's Model Registry is the equivalent
packaging: one artifact carrying preprocessing plus model plus signature.
Registering this bundle as a `pyfunc` is the natural next step; the manifest
already holds everything a `pyfunc` wrapper would need.

## Rejected

- **Pickle / joblib.** Breaks across library versions, carries no human-readable
  contract, and executes arbitrary code on load.
- **ONNX.** Portable and fast, but it would discard the categorical handling
  this model depends on and add a conversion step to verify.
- **Manifest as the only source of truth.** The model file already knows its
  feature order; using it as a check costs nothing and catches assembly errors.

## Consequences

- A fold bundle is ~30MB, which is 30× the `check-added-large-files` limit — so
  models can never be committed. See [0006](0006-ghcr-and-releases.md).
- **Revisit when:** a non-LightGBM member joins the bundle, at which point
  `LOADERS` in `src/predict.py` needs a second entry.
