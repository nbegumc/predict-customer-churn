# Serving image for the churn API.
#
#   docker build --build-arg MODEL_RUN_ID=1024_141318_lgbm -t churn-api:local .
#   docker run -p 8000:8000 churn-api:local
#
# Two stages so the build toolchain never ships: uv, the compilers it may pull,
# and the package cache all stay behind. The runtime carries a virtualenv and
# three source directories.

# ---- build ------------------------------------------------------------------
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# Only the lockfile and manifest, so this layer caches until dependencies change.
COPY pyproject.toml uv.lock ./

# --no-default-groups drops train/notebook/kaggle/mlops/dev; --group serve adds
# fastapi and uvicorn. That is the whole reason for the dependency split in
# pyproject.toml: 29 packages instead of 287, and no sklearn, catboost, mlflow
# or jupyter in a container whose only job is to score rows.
# --frozen fails rather than re-resolving, so the image matches the lockfile
# the tests ran against.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-default-groups --group serve --no-install-project

# ---- runtime ----------------------------------------------------------------
FROM python:3.11-slim-bookworm AS runtime

# LightGBM's manylinux wheel links OpenMP dynamically and python:slim does not
# ship it. Without this the image builds fine and then dies at the first
# `import lightgbm` with "libgomp.so.1: cannot open shared object file" --
# the same dependency CHEATSHEET.md installs as `brew install libomp` on macOS.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Non-root. 10001 is outside the range a host distro assigns, so a bind-mounted
# volume cannot silently land on an existing local account.
RUN useradd --create-home --uid 10001 app
WORKDIR /app

# The venv's shebangs carry absolute paths, so /app must match the build stage.
COPY --from=builder --chown=app:app /app/.venv /app/.venv
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Which bundle this image serves. Baked in rather than mounted: an image should
# never be ambiguous about the model it carries, and 30MB is free. The run id
# also goes into the labels below so `docker inspect` can answer the question.
ARG MODEL_RUN_ID
RUN test -n "$MODEL_RUN_ID" || (echo "build with --build-arg MODEL_RUN_ID=<run_id>" && false)

COPY --chown=app:app src/ ./src/
COPY --chown=app:app models/${MODEL_RUN_ID}/ ./models/${MODEL_RUN_ID}/

ENV MODEL_RUN_ID=${MODEL_RUN_ID} \
    MODEL_STRATEGY=fold_average \
    PORT=8000

USER app
EXPOSE 8000

# urllib rather than curl, which slim does not ship and which is not worth a layer.
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]

LABEL org.opencontainers.image.title="churn-api" \
      org.opencontainers.image.description="Churn scoring service for playground-series-s6e3" \
      org.opencontainers.image.source="https://github.com/nbegumc/predict-customer-churn" \
      ai.model.run_id="${MODEL_RUN_ID}"
