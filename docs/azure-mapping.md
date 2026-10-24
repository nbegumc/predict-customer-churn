# Open source here, Azure there

Every tool in this project with its Azure counterpart and **what would actually
change in the code**. The short version: most of the migration is environment
variables, because the open-source choices were made with that in mind.

| Concern | Here | Azure | What changes in this repo |
|---|---|---|---|
| Experiment tracking | MLflow | **Azure ML** | `MLFLOW_TRACKING_URI` → `azureml://...`. Nothing else: Azure ML speaks MLflow natively. |
| Model registry | MLflow registry (SQLite) | **Azure ML Model Registry** | Register the bundle as a `pyfunc`; the manifest already holds what a wrapper needs. |
| Model distribution | GitHub Releases | **Azure ML Models** / ACR artifacts | CD's `gh release download` becomes `az ml model download`. |
| Serving | FastAPI + uvicorn | **ML Managed Online Endpoints** or **Container Apps** | Container Apps: nothing — same image. Online Endpoints: a scoring script wrapping `src.predict`, same `init`/`run` shape as the `lifespan` handler. |
| Image registry | GHCR | **Azure Container Registry** | Registry host in `cd.yml`; `azure/login` with OIDC instead of `GITHUB_TOKEN`. |
| Image build | Docker + uv, multi-stage | **ACR Tasks** | Dockerfile unchanged. |
| CI/CD | GitHub Actions | **Azure Pipelines** (or Actions + `az`) | Same jobs, different YAML dialect. |
| Secrets | GitHub Secrets | **Azure Key Vault** | Referenced via managed identity instead of `secrets.*`. |
| Metrics | `prometheus_client` | **Azure Monitor managed Prometheus** | Nothing — it scrapes `/metrics` unchanged. App Insights covers latency but not the prediction histogram. |
| Drift | Evidently, batch | **Azure ML Model Monitoring** | Same shape (reference + collected inference data, on a schedule). Data Collector replaces the JSONL append. |
| Data versioning | DVC, local remote | **Blob remote** / **ML Data Assets** | `dvc remote add -d azure azure://...`. Pointers unchanged. |
| Orchestration | `make` + CI | **Azure ML Pipelines** | Would replace `src.train.run` as the entry point — the biggest change on this list. |
| Compute | a laptop | **Azure ML Compute** | `make data` + `src.train` on a compute target. |

## The three that are not one-liners

**Serving on Managed Online Endpoints** means adopting Azure's scoring-script
contract. Container Apps avoids that entirely and runs this image as-is — the
trade is losing built-in traffic splitting between model versions, which is
exactly what would turn the `fold_average` → `full_refit` switch into a canary
rather than a restart.

**Azure ML Pipelines** would take over orchestration. This repo's invariant is
that every training run goes through `src.train.run` so it lands in
`experiments.csv` with a fold fingerprint; a pipeline step would have to call
the same function rather than reimplement the loop, or that discipline is lost.

**Managed Feature Store** is the structural answer to
[ADR 0012](adr/0012-train-serve-parity.md): one feature definition serving both
training and inference, rather than two paths kept in agreement by a test. It is
worth real money at scale and is over-engineering for 21 features.

## What does not transfer

The practices. The frozen CV scheme, the fold fingerprint, the bitwise parity
test, the habit of writing down why a run was kept or dropped — none of these is
a product on anyone's cloud. They are the part worth carrying between jobs.
