# Architecture

How a row of raw customer data becomes a scored prediction, and what is
guaranteed at each hop.

## The pipeline

```mermaid
flowchart TB
    subgraph T["training — offline, on a laptop"]
        RAW["data/raw/train.csv<br/>594,194 rows · gitignored"]
        CLEAN["data.basic_clean<br/>+ encode_target"]
        RED["utils.reduce_memory<br/>float64 → float32"]
        FEAT["features.build_features<br/>+ 2 engineered ratios → 21 features"]
        CV["cv.get_folds<br/>stratified · 5 · fingerprint 13feb807"]
        FIT["5 × LGBMModel.fit<br/>early stopping on each fold"]
        RAW --> CLEAN --> RED --> FEAT --> CV --> FIT
    end

    subgraph A["artifacts"]
        OOF["experiments/oof_*.npy<br/>test_*.npy — DVC"]
        CSV["experiments.csv<br/>+ experiments.md"]
        ML["MLflow<br/>params · metrics · artifacts"]
        BUNDLE["models/&lt;run_id&gt;/<br/>fold0-4.txt · full.txt<br/>manifest.json"]
        FIT --> OOF & CSV & ML & BUNDLE
    end

    subgraph S["serving — the container"]
        REQ["POST /predict<br/>19 raw fields"]
        SCHEMA["Pydantic Customer<br/>15 Literals · range guards"]
        PRED["predict.predict_raw<br/>reduce_memory → ratios →<br/>manifest feature order<br/>→ manifest categories"]
        OUT["probability + risk band"]
        REQ --> SCHEMA --> PRED --> OUT
        BUNDLE -.loaded once at startup.-> PRED
    end

    subgraph O["operations"]
        METRICS["/metrics<br/>5 Prometheus series"]
        LOG["logs/predictions.jsonl"]
        DRIFT["src.monitor<br/>Evidently vs reference"]
        PRED --> METRICS & LOG --> DRIFT
    end

    BUNDLE ==>|"make release-bundle<br/>+ gh release create"| REL["GitHub Release"]
    REL ==>|CD| GHCR["ghcr.io image"]
```

## The two paths must agree, and are tested to

The training path and the serving path compute the same 21 features through
**the same functions in the same order**. That is not a style preference; three
specific bugs live in the gap between them, and
[ADR 0012](adr/0012-train-serve-parity.md) records all three with measurements.

The guarantee is bitwise: `tests/test_predict.py` scores all 254,655 rows of
`data/raw/test.csv` through the bundle and asserts the result equals
`experiments/test_<run_id>.npy` exactly. Not `allclose` — a tolerance cannot
tell a real feature bug from rounding.

Two things the serving path deliberately does **not** call:

| | why not |
|---|---|
| `data.basic_clean` | ends in `drop_duplicates()` — three identical customers in a batch become one row, and the response misaligns with the request |
| `features.build_features` | calls `as_categorical`, which rebuilds category lists *from the request*; LightGBM encodes categoricals as integer codes, so the codes would be wrong |

## What the manifest carries

The model file independently records its feature order and category lists, so
`manifest.json` is a **cross-check**, not a thing to trust — `load_bundle`
refuses a bundle where the two disagree. What lives only in the manifest:

- `strategies` — a name → weighted member list. `fold_average` is five members
  at 0.2; `full_refit` is one at 1.0. One code path at inference, and a future
  blend is an appended member.
- `dtypes`, `raw_columns`, `best_iterations`, `score_quantiles` (the risk-band
  thresholds), `cv_score`, `fold_fingerprint`, `git_sha`.

## The invariant that holds it together

`CV_SCHEME`, `N_FOLDS` and `src/cv.py` are frozen. Every saved `oof_*.npy`
carries fold fingerprint `13feb807`, and `blend.py` refuses to average files
produced under different folds. `tests/test_cv.py` reproduces that fingerprint
from a label-only frame — **no data required** — so CI enforces it on every PR.

## Where each decision is written down

Twelve ADRs in [`docs/adr/`](adr/README.md), each with its Azure equivalent.
Start with [0012](adr/0012-train-serve-parity.md) if you are going to change
feature code, and [0002](adr/0002-fold-average-over-refit.md) if you want to
know why the served model is the one it is.
