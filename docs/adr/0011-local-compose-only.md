# 0011 — Local docker-compose only; deployment documented, not built

**Status:** accepted, 2026-10-24

## Context

The project goal was to learn an end-to-end pipeline. Hosting was considered —
Hugging Face Spaces specifically — and deliberately scoped out.

## Decision

`docker-compose.yml` brings up the API, MLflow and Prometheus locally. There is
no deploy workflow. CD publishes a versioned image to GHCR and stops there;
`docs/deployment.md` records what running it elsewhere involves.

The reasoning: the image is host-agnostic, so deployment is the *least*
transferable part of the exercise and the easiest to add later. A published
endpoint also has running costs and an attack surface that a learning project
does not need. Everything a deploy would exercise — a reproducible image, a
health check, readiness, configuration by environment variable, a published
registry artifact — is already built and tested.

One finding worth recording: **Hugging Face's pricing page lists Docker Spaces
under PRO ($9/month)** while the Docker Spaces documentation does not mention a
plan requirement. That ambiguity is unresolved, and it is the kind of thing to
verify before designing around a free tier.

## Azure equivalent

**Azure Container Apps** is the closest fit: it runs this image unchanged,
scales to zero, and takes `MODEL_RUN_ID` / `MODEL_STRATEGY` as environment
variables exactly as compose does. **Azure ML Managed Online Endpoints** is the
ML-native option, adding traffic splitting between model versions — the thing
that would make the `fold_average` vs `full_refit` switch
([0002](0002-fold-average-over-refit.md)) a canary instead of a restart. **AKS**
only if something else already needs a cluster.

## Rejected

- **Hugging Face Spaces.** The plan ambiguity above, plus it is the piece that
  teaches least.
- **A free container host (Render / Koyeb / Cloud Run).** Would work; adds a
  second account and a cold-start story for no learning.
- **Terraform / Bicep for infrastructure.** Infrastructure-as-code is worth
  learning and is a different project.

## Consequences

- No public URL to show. The compose stack plus `GET /` is the demo.
- The deploy step is the one link in the chain not exercised end to end.
- **Revisit when:** someone other than the author needs to call the API.
