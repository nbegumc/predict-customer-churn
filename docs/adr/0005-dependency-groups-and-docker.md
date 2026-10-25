# 0005 — uv dependency groups and a multi-stage image

**Status:** accepted, 2026-10-24

## Context

`pyproject.toml` had one flat `dependencies` list holding jupyter, ipykernel,
nbstripout, matplotlib, seaborn, shap, optuna, xgboost and the Kaggle CLI. A
container built from it would install the entire notebook and competition
toolchain to score one LightGBM booster. (`xgboost` and `optuna` were not
imported anywhere at all.)

## Decision

Split into groups — `train`, `notebook`, `kaggle`, `serve`, `mlops`, `tuning`,
`dev` — leaving only `pandas`, `numpy`, `lightgbm` as runtime dependencies.

`[tool.uv] default-groups` restores exactly what a bare `uv sync` used to
install, so **the daily workflow is unchanged**. The image opts out with
`--no-default-groups --group serve`, which resolves to **29 packages instead of
287**: no sklearn, no catboost, no mlflow, no notebook stack.

That boundary is real and enforced: `src/cv.py` is what imports sklearn, and
`tests/test_config.py` asserts in a subprocess that the inference path
(`config`, `utils`, `data`, `features`, `predict`, `api`) never pulls it in.

Multi-stage image: `uv` builds the venv, `python:3.11-slim-bookworm` runs it.
`--frozen` fails rather than re-resolving, so the image matches the lockfile the
tests ran against. Non-root uid 10001.

**`libgomp1` is the non-obvious part.** LightGBM's manylinux wheel links OpenMP
dynamically and `python:slim` does not ship it, so without it the image builds
clean and dies at the first `import lightgbm`. It is the same dependency
`CHEATSHEET.md` installs as `brew install libomp` on macOS — the symptom just
moves from build time to run time.

## Azure equivalent

**Azure Container Registry**, with **ACR Tasks** as the build service (the
equivalent of the `docker` CI job). ACR supports multi-stage builds and layer
caching the same way. For base images, Microsoft Artifact Registry mirrors the
Debian slim images, which matters if egress to Docker Hub is restricted.

## Rejected

- **A single-stage image.** Ships uv, the package cache and build tooling.
- **`python:3.11` (non-slim).** ~1GB of base for a `libgomp1` you can add in one
  layer.
- **Alpine.** musl means no manylinux wheels; LightGBM would compile from source.
- **Mounting the model instead of baking it.** An image should never be
  ambiguous about which model it carries.

## Consequences

- `make setup` still works, because `default-groups` preserves the old behaviour.
- Image is 540MB, dominated by numpy/scipy/pandas wheels.
- **Revisit when:** image size matters enough to try a distroless runtime.
