# 0007 — CI builds with a fixture model; CD builds with the real one

**Status:** accepted, 2026-10-24

## Context

CI should prove the container works. But the real bundle is 30MB, gitignored and
30× over the large-files hook, so **no clone — including a CI checkout — has a
model in it**. The obvious workarounds all make CI depend on a secret, a
download, or the 77MB dataset.

## Decision

Split the two concerns rather than reconciling them.

**CI trains its own throwaway model.** `scripts/make_fixture_data.py` expands
the committed 200-row fixture into train/test/sample_submission, and
`CHURN_RAW_DIR` points `src/config.py` at it. Training takes under a second and
exercises the same code path. CI then builds the image with *that* bundle, runs
it, scores a real request and checks `tenure: 0` is still a 422. No data, no
credentials, no secrets.

**CD builds the shippable image** from a bundle attached to a Release
([0006](0006-ghcr-and-releases.md)).

The three CI jobs map to the three ways this repo can break: `lint` (including
`uv-lock`, so a `pyproject.toml` edited without re-locking fails), `test` (the
fold-fingerprint lock turns a `CV_SCHEME` change red), and `build and smoke`.

## Azure equivalent

**Azure Pipelines**, or these same workflows plus `azure/login` with OIDC
federation and `az ml online-deployment update` as the deploy step. The CI /
CD split is identical; only the registry and the deploy target change.

## Rejected

- **Downloading the real model in CI.** Needs a credential and makes every PR
  depend on an external service.
- **Committing a tiny real model.** It would not be the model, so the test would
  not be the test.
- **Skipping the container job.** The Dockerfile is where `libgomp1`-class
  failures live, and they are invisible to unit tests.

## Consequences

- The image CI builds is never shipped. That is explicit in the fixture script's
  docstring and in the run notes it logs.
- `CHURN_RAW_DIR` exists partly for CI — and immediately revealed that training
  on fixture data pollutes the real `experiments.csv`, which is why
  `CHURN_WORK_DIR` exists too. CI sets only `RAW`, because the Dockerfile copies
  `models/<run_id>` relative to the build context.
- **Revisit when:** the build gets slow enough to want a published base image
  with dependencies pre-installed.
