# Deployment

**Nothing here is built.** This project runs locally by choice
([ADR 0011](adr/0011-local-compose-only.md)). CD publishes a versioned image to
GHCR and stops. This file records what running it somewhere else would involve,
because the gap between "it works in compose" and "it is deployed" is where most
of the remaining work in a real system lives.

## Cutting a release

The only path to a shippable image, by design — there is no route to production
that does not produce a version number.

```bash
uv run python -m src.train --model lgbm --full-refit --notes "release candidate"
make release-bundle RUN_ID=<run_id>
gh release create v1.0.0 dist/bundle-<run_id>.tar.gz --notes "OOF 0.916391"
```

`cd.yml` then fetches the bundle, builds, pushes
`ghcr.io/nbegumc/predict-customer-churn:v1.0.0`, pulls the pushed image back and
smoke-tests **that** — not the local build.

## Running the published image

```bash
docker login ghcr.io       # GHCR package visibility is independent of the repo's
docker run -p 8000:8000 ghcr.io/nbegumc/predict-customer-churn:v1.0.0
```

The image carries its model. `docker inspect` reports which:
`ai.model.run_id`, `ai.model.cv_score`, `org.opencontainers.image.revision`.

## Azure Container Apps — the lowest-friction option

Runs this image unchanged, scales to zero, takes the same environment
variables compose does.

```bash
az containerapp create -n churn-api -g <rg> --environment <env> \
  --image ghcr.io/nbegumc/predict-customer-churn:v1.0.0 \
  --target-port 8000 --ingress external \
  --env-vars MODEL_STRATEGY=fold_average \
  --min-replicas 0 --max-replicas 3
```

Then: Key Vault for anything secret, Azure Monitor managed Prometheus scraping
`/metrics` unchanged, and API Management in front for auth and rate limiting.
`/health` and `/ready` are already the right shape for probes — `/ready` returns
503 until the bundle loads, so a rollout waits rather than serving 503s to users.

## Azure ML Managed Online Endpoints — the ML-native option

More work: Azure's scoring-script contract replaces `src/api.py`, with the same
`init()` / `run()` shape as the `lifespan` handler plus `_score`. What it buys
is **traffic splitting between model versions** — which would turn the
`fold_average` → `full_refit` switch from a restart into a canary, and is the
one capability this stack genuinely lacks.

## What is missing either way

- **Authentication.** Nothing in front of `/predict`.
- **Rate limiting.** `MAX_BATCH = 1000` is the only guard.
- **Horizontal scale.** Prometheus counters are per-instance; more than one
  replica needs aggregation.
- **Scheduled drift.** `make drift` is manual. It wants a cron or a routine.
- **A retraining trigger.** Nothing closes the loop from a drift report back to
  a training run — and nothing should, until labels arrive late enough to make
  *performance* drift measurable rather than input drift.
