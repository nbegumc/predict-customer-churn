# 0006 — GHCR for images, GitHub Releases for model bundles

**Status:** accepted, 2026-10-24

## Context

A fold bundle is ~30MB. The repo's `check-added-large-files` hook caps commits
at 1MB and `models/` is gitignored, so **the model can never live in git**. It
still has to reach the image build reproducibly, from a specific version.

## Decision

Two channels, one artifact each.

**Images → GHCR.** `GITHUB_TOKEN` can already write this repository's packages,
so there is no secret to configure and no second account to manage.

**Model bundles → GitHub Releases.** `make release-bundle RUN_ID=...` tars a
bundle; `gh release create <tag> dist/bundle-<run_id>.tar.gz` attaches it. The
bundle is then versioned, downloadable and reproducible from a tag — which is
precisely the job a model registry does.

CD reads the bundle from the Release, builds, pushes with `cv_score` and
`run_id` as OCI labels, then **pulls the pushed image back and smoke-tests
that** rather than the local build.

## Azure equivalent

**Azure Container Registry** for images and **Azure ML Model Registry** for
bundles — the latter is the closer analogue of the Release channel, since it
versions models with lineage and lets an endpoint reference `model:1` directly.
ACR can also store models as OCI artifacts if you would rather have one system.

## Rejected

- **git-lfs.** Bandwidth is billed, and every clone pays for a file most clones
  do not want.
- **DVC as the model channel.** Its remote here is a local directory by design
  ([0010](0010-dvc-scoped-to-experiments.md)), so CI cannot reach it. Two
  channels for one artifact is how they diverge.
- **Committing a smaller model.** Would mean choosing the artifact to fit the
  tooling, which is backwards.

## Consequences

- Shipping requires cutting a Release. That is a feature: there is no path to
  production that does not produce a version number.
- **GHCR package visibility is independent of repository visibility and defaults
  to private**, so a public repo can still have a package needing
  `docker login ghcr.io`. This surprises people.
- **Revisit when:** bundles get large enough that Release assets (2GB limit)
  stop fitting.
