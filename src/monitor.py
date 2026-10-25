"""Data drift between what the model was trained on and what it is being sent.

    uv run python -m src.monitor build-reference      # once, from data/raw/train.csv
    uv run python -m src.monitor report               # reference vs the live log
    uv run python -m src.monitor report --since 7d

A CLI, shaped like `src/log_lb.py`, deliberately **not** part of the request
path. Evidently pulls plotly and nltk; putting it in the serving image would
roughly double it for a question that is not per-request. Drift is a batch
question answered on a batch cadence — which is also how Azure ML Model
Monitoring works, so the shape transfers.

**This dataset comes with its own null hypothesis.** `notebooks/01_eda.ipynb`
ran adversarial validation of train against test and got AUC **0.5117** — the
two frames are the same distribution. So "no drift" here is not a guess at a
threshold, it is a measured property of the data the model was fitted on. A
report that flags columns is reporting something genuinely new.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timedelta

import pandas as pd

from . import config as C
from .data import normalise_columns
from .features import DERIVED
from .utils import use_experiment

REFERENCE = C.PROCESSED / "reference.parquet"
REFERENCE_ROWS = 10_000

# The 19 raw columns a request carries, split the way Evidently needs them.
NUMERIC = ["seniorcitizen", "tenure", "monthlycharges", "totalcharges"]
PREDICTION = "churn_probability"


def _since(spec: str | None) -> datetime | None:
    """Parse `7d`, `12h`, `30m` into a cutoff. None means everything."""
    if not spec:
        return None
    match = re.fullmatch(r"(\d+)([dhm])", spec)
    if not match:
        sys.exit(f"--since takes a value like 7d, 12h or 30m; got {spec!r}")
    n, unit = int(match.group(1)), match.group(2)
    delta = {"d": timedelta(days=n), "h": timedelta(hours=n), "m": timedelta(minutes=n)}[unit]
    return datetime.now().astimezone() - delta


def build_reference(n_rows: int = REFERENCE_ROWS) -> pd.DataFrame:
    """Sample the training frame once and keep it as the drift baseline.

    A sample rather than the whole 77MB table: drift statistics on 10,000 rows
    are precise enough that the extra rows buy nothing, and the file has to be
    small enough to sit beside a model in a container.
    """
    if not C.TRAIN_FILE.exists():
        sys.exit(f"{C.TRAIN_FILE} is absent — run `make data` first.")

    frame = normalise_columns(pd.read_csv(C.TRAIN_FILE))
    frame = frame.drop(columns=[c for c in (C.TARGET, C.ID_COL) if c in frame.columns])
    sample = frame.sample(n=min(n_rows, len(frame)), random_state=C.SEED)

    REFERENCE.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(REFERENCE, index=False)
    print(f"reference: {len(sample):,} rows x {sample.shape[1]} columns -> {REFERENCE}")
    return sample


def load_predictions(since: datetime | None = None) -> pd.DataFrame:
    """Read the JSONL the API appends to, one line per scored row."""
    if not C.PRED_LOG.exists():
        sys.exit(
            f"no prediction log at {C.PRED_LOG}. Score something through the API first:\n"
            "  uv run uvicorn src.api:app  &&  curl -X POST localhost:8000/predict ..."
        )
    rows = [json.loads(line) for line in C.PRED_LOG.read_text().splitlines() if line.strip()]
    frame = pd.DataFrame(rows)
    if since is not None and "ts" in frame.columns:
        frame = frame[pd.to_datetime(frame["ts"], format="ISO8601") >= since]
    if frame.empty:
        sys.exit("no predictions in the selected window.")
    return frame


def report(since: str | None = None, open_browser: bool = False) -> dict:
    """Compare the live request distribution against the reference."""
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset

    if not REFERENCE.exists():
        sys.exit("no reference yet — run `uv run python -m src.monitor build-reference`.")

    reference = pd.read_parquet(REFERENCE)
    current = load_predictions(_since(since))

    # Only columns present in both. The live log also carries the prediction,
    # which the reference cannot have — prediction drift is reported separately
    # by the Prometheus histogram in src/api.py.
    shared = [c for c in reference.columns if c in current.columns and c not in DERIVED]
    definition = DataDefinition(
        numerical_columns=[c for c in NUMERIC if c in shared],
        categorical_columns=[c for c in shared if c not in NUMERIC],
    )

    result = Report([DataDriftPreset()]).run(
        Dataset.from_pandas(current[shared], data_definition=definition),
        Dataset.from_pandas(reference[shared], data_definition=definition),
    )

    # C.REPORTS, read at call time -- not ROOT/"reports", which always meant the
    # real repo and so ignored CHURN_WORK_DIR and every test's tmp redirection.
    C.REPORTS.mkdir(parents=True, exist_ok=True)
    out = C.REPORTS / f"drift_{datetime.now().astimezone():%Y-%m-%d_%H%M}.html"
    result.save_html(str(out))

    summary = {
        "generated": f"{datetime.now().astimezone():%Y-%m-%d %H:%M:%S%z}",
        "reference_rows": len(reference),
        "current_rows": len(current),
        "columns_checked": len(shared),
        "window": since or "all",
        "report": str(out),
    }
    # Metric names are parameterised, e.g. "DriftedColumnsCount(drift_share=0.5)"
    # and "ValueDrift(column=tenure,method=...,threshold=0.1)", so match on the
    # prefix and pull the column and threshold out of the name.
    per_column: dict[str, dict] = {}
    drifted: list[str] = []
    for metric in result.dict()["metrics"]:
        name = metric["metric_name"]
        if name.startswith("DriftedColumnsCount"):
            summary["drifted_columns"] = int(metric["value"]["count"])
            summary["drifted_share"] = round(float(metric["value"]["share"]), 4)
        elif name.startswith("ValueDrift"):
            found = re.search(r"column=([^,)]+),method=([^,)]+),threshold=([0-9.]+)", name)
            if found:
                column, method, threshold = found.group(1), found.group(2), float(found.group(3))
                score = float(metric["value"])
                per_column[column] = {"score": round(score, 5), "method": method}
                # The comparison direction depends on the test, and Evidently
                # picks the test by sample size: under ~1000 rows it uses
                # hypothesis tests ("K-S p_value", "Z-test p_value") where drift
                # means *below* the threshold, above that it uses distances
                # ("Wasserstein", "Jensen-Shannon") where drift means above.
                # Assuming one direction reports every column as drifted when a
                # 200-row sample is compared against itself.
                is_p_value = "p_value" in method
                if (score < threshold) if is_p_value else (score > threshold):
                    drifted.append(column)
    summary["drifted"] = sorted(drifted)

    print(json.dumps(summary, indent=2))
    if drifted:
        print(f"\ndrifted ({len(drifted)} of {summary['columns_checked']}):")
        for column in sorted(drifted):
            detail = per_column[column]
            print(f"  {column:20s} {detail['score']:<12} {detail['method']}")
        print(
            "\nWorth knowing before concluding anything: adversarial validation of train "
            "against test scored AUC 0.5117 (notebooks/01_eda.ipynb), so this data has no "
            "intrinsic shift — a flag here is new information, not the baseline."
        )
    else:
        method = next(iter(per_column.values()), {}).get("method", "n/a")
        print(f"\nno column drift against the training distribution ({method} and friends).")

    _log_to_mlflow(summary, out)
    if open_browser:
        import webbrowser

        webbrowser.open(out.as_uri())
    return summary


def _log_to_mlflow(summary: dict, html) -> None:
    """Best-effort: a drift report is worth keeping next to the runs it judges."""
    try:
        import mlflow
    except ImportError:
        return
    try:
        mlflow.set_tracking_uri(C.MLFLOW_TRACKING_URI)
        use_experiment(mlflow)  # artifacts beside the store, not in the cwd
        with mlflow.start_run(run_name=f"drift_{datetime.now().astimezone():%m%d_%H%M%S}"):
            mlflow.set_tag("kind", "drift-report")
            mlflow.log_metrics({k: v for k, v in summary.items() if isinstance(v, (int, float))})
            mlflow.log_artifact(str(html))
    except Exception as exc:  # monitoring must not break on tracking
        print(f"(MLflow logging skipped: {type(exc).__name__}: {exc})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    ref = sub.add_parser("build-reference", help="sample data/raw/train.csv as the baseline")
    ref.add_argument("--rows", type=int, default=REFERENCE_ROWS)

    rep = sub.add_parser("report", help="compare the live prediction log against the reference")
    rep.add_argument("--since", default=None, help="e.g. 7d, 12h, 30m (default: everything)")
    rep.add_argument("--open", action="store_true", help="open the HTML report when done")

    args = parser.parse_args()
    if args.command == "build-reference":
        build_reference(args.rows)
    else:
        report(args.since, args.open)
