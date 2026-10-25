# Architecture decision records

One file per decision that would otherwise be re-litigated in six months.
Each says what the problem was, what was chosen, **what the Azure equivalent
is**, what was rejected, and what the choice costs.

The format is deliberate. A decision without its alternatives is folklore: you
cannot tell whether it was reasoned or inherited, and you cannot tell when it
has stopped being true. Every ADR here ends with a "revisit when" line for
exactly that reason.

| # | Decision | Azure equivalent |
|---|---|---|
| [0001](0001-mlflow-alongside-csv.md) | MLflow as a second sink, not a replacement | Azure ML (MLflow-native) |
| [0002](0002-fold-average-over-refit.md) | Serve the fold average, not a full refit | Azure ML Model Registry |
| [0003](0003-native-text-and-manifest.md) | LightGBM native text + a JSON manifest, not pickle | MLflow `pyfunc` in the registry |
| [0004](0004-fastapi-one-module.md) | FastAPI in one module; the schema is the contract | Azure ML Managed Online Endpoints |
| [0005](0005-dependency-groups-and-docker.md) | uv dependency groups + a multi-stage image | ACR Tasks |
| [0006](0006-ghcr-and-releases.md) | GHCR for images, GitHub Releases for model bundles | Azure Container Registry |
| [0007](0007-ci-fixture-cd-real.md) | CI builds with a fixture model; CD with the real one | Azure Pipelines |
| [0008](0008-prometheus-in-process.md) | Prometheus client in-process; no Grafana | Azure Monitor / App Insights |
| [0009](0009-evidently-as-batch.md) | Evidently as a batch job, not middleware | Azure ML Model Monitoring |
| [0010](0010-dvc-scoped-to-experiments.md) | DVC for `experiments/` only | Azure Blob + ML Data Assets |
| [0011](0011-local-compose-only.md) | Local compose only; deployment documented | Azure Container Apps |
| [0012](0012-train-serve-parity.md) | Train/serve dtype parity as a discipline | *(none — a practice, not a tool)* |

Numbering is chronological and permanent. A superseded ADR is marked, not
deleted, because the reasoning is the point.
