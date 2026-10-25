"""Small helpers you will use in every notebook and script."""

from __future__ import annotations

import csv
import json
import os
import random
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np

from . import config as C

LOG_FILE = C.ROOT / "experiments.csv"
# `folds` is the fold fingerprint from cv.fold_fingerprint — it is what lets
# blend.py refuse to average OOF files produced under different fold schemes.
LOG_FIELDS = [
    "run_id",
    "timestamp",
    "model",
    "cv_score",
    "public_lb",
    "folds",
    "notes",
]


def seed_everything(seed: int = C.SEED) -> None:
    """Seed python, numpy and (if present) torch. Call at the top of everything."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


@contextmanager
def timer(name: str):
    t0 = time.time()
    print(f"[{name}] start")
    yield
    print(f"[{name}] done in {time.time() - t0:.1f}s")


def new_run_id(model: str) -> str:
    """Timestamped id so OOF files, submissions and log rows line up."""
    return f"{datetime.now().astimezone():%m%d_%H%M%S}_{model}"


def log_experiment(
    run_id: str,
    model: str,
    cv_score: float,
    public_lb: float | None = None,
    notes: str = "",
    folds: str | None = None,
) -> None:
    """Append one line per run. Two seconds of typing, saves you on day three."""
    is_new = not LOG_FILE.exists()
    with LOG_FILE.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if is_new:
            w.writeheader()
        w.writerow(
            {
                "run_id": run_id,
                "timestamp": f"{datetime.now().astimezone():%Y-%m-%d %H:%M}",
                "model": model,
                "cv_score": round(float(cv_score), 6),
                "public_lb": public_lb if public_lb is not None else "",
                "folds": folds or "",
                "notes": notes,
            }
        )
    print(f"logged {run_id}: cv={cv_score:.6f}")


class _NoMlflow:
    """Swallows every mlflow call, so train.py needs no conditionals.

    Exists for one documented reason: the README documents
    `from src.train import run` inside a Kaggle kernel, and
    `kaggle/kernel-metadata.json` sets `enable_internet: false`, so mlflow is
    not installed there. Without this, importing mlflow at module scope would
    break the kernel path.
    """

    def __getattr__(self, _name):
        return lambda *_a, **_k: None


class _SafeMlflow:
    """Forwards to mlflow, turning any tracking failure into a printed warning.

    MLflow is the second sink. Left unguarded, a locked SQLite file, a full disk
    or an unreachable server would raise mid-run — after the OOF arrays,
    submission and bundle were saved, but before `log_experiment` wrote the
    experiments.csv row. The secondary record would erase the primary one.
    """

    def __init__(self, module):
        self._mlflow = module

    def __getattr__(self, name):
        attr = getattr(self._mlflow, name)
        if not callable(attr):
            return attr

        def guarded(*args, **kwargs):
            try:
                return attr(*args, **kwargs)
            except Exception as exc:
                print(f"(mlflow.{name} failed; training continues: {type(exc).__name__}: {exc})")
                return None

        return guarded


def artifact_root() -> str | None:
    """Where a new experiment's artifacts go: beside its SQLite database.

    Left to itself, MLflow puts a new experiment's artifacts in ./mlruns
    relative to the *working directory*, not to the tracking store. So a store
    pointed elsewhere still wrote its artifacts into the repo -- every test
    fixture redirects the database to a tmp dir, and over one day of test runs
    that left ~1,100 runs and 226MB in the real mlruns/. Deriving the location
    from the tracking URI, read at call time, means anything that moves the
    database moves the artifacts with it.

    None for a tracking server (or Azure ML): the server decides.
    """
    uri = C.MLFLOW_TRACKING_URI
    if not uri.startswith("sqlite:///"):
        return None
    database = Path(uri.removeprefix("sqlite:///")).resolve()
    return str(database.parent / C.MLFLOW_EXPERIMENT)


def use_experiment(mlflow) -> None:
    """Select the experiment, creating it with artifacts stored beside the DB."""
    if mlflow.get_experiment_by_name(C.MLFLOW_EXPERIMENT) is None:
        mlflow.create_experiment(C.MLFLOW_EXPERIMENT, artifact_location=artifact_root())
    mlflow.set_experiment(C.MLFLOW_EXPERIMENT)


@contextmanager
def mlflow_run(run_id: str, model: str, notes: str = ""):
    """Mirror one training run into MLflow, yielding the module (or a shim).

    This is a *second* sink, not a replacement, and it is built so it cannot
    break the first. `experiments.csv` stays the file `blend.py` reads to check
    fold fingerprints, and `log_experiment` below is untouched — note that it
    writes the CSV header only when the file is absent, so adding a column there
    would append 8-value rows under a 7-column header. The link runs the other
    way: the repo's `run_id` is an MLflow tag, so a run and a CSV row are
    joinable by eye and by query.

    If MLflow cannot start a run, training continues untracked; if any later
    call fails, `_SafeMlflow` reports it and carries on. Only an error in
    training itself propagates, and it marks the MLflow run FAILED on the way.

    Pointing MLFLOW_TRACKING_URI at an Azure ML workspace is the entire
    migration — Azure ML speaks MLflow natively and this code does not change.
    """
    try:
        import mlflow
    except ImportError:
        yield _NoMlflow()
        return

    try:
        mlflow.set_tracking_uri(C.MLFLOW_TRACKING_URI)
        use_experiment(mlflow)
        mlflow.start_run(run_name=run_id)
    except Exception as exc:
        print(f"(MLflow unavailable; training continues untracked: {type(exc).__name__}: {exc})")
        yield _NoMlflow()
        return

    ml = _SafeMlflow(mlflow)
    ml.set_tags(
        {
            "run_id": run_id,
            "model": model,
            "competition": C.COMPETITION,
            "cv_scheme": C.CV_SCHEME,
            "n_folds": C.N_FOLDS,
            "seed": C.SEED,
            "git_sha": git_sha(),
            "notes": notes,
        }
    )
    status = "FAILED"
    try:
        yield ml
        status = "FINISHED"
    finally:
        ml.end_run(status=status)


def git_sha() -> str:
    """Short HEAD, marked dirty when the tree has uncommitted changes.

    A run whose code is unrecorded is a run you cannot reproduce, and `-dirty`
    is the honest answer when the tree did not match any commit.
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            cwd=C.ROOT,
        ).stdout.strip()
        if not sha:
            return "unknown"
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
            cwd=C.ROOT,
        ).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except OSError:  # no git binary, e.g. a Kaggle kernel
        return "unknown"


def save_oof(
    run_id: str, oof: np.ndarray, test_pred: np.ndarray, params: dict | None = None
) -> None:
    """Persist out-of-fold + test predictions. This is what makes blending possible."""
    np.save(C.EXPERIMENTS / f"oof_{run_id}.npy", oof)
    np.save(C.EXPERIMENTS / f"test_{run_id}.npy", test_pred)
    if params is not None:
        (C.EXPERIMENTS / f"params_{run_id}.json").write_text(
            json.dumps(params, indent=2, default=str)
        )


def savefig(fig, name: str, dpi: int = 150) -> None:
    """Save every chart as you make it — visualisation is worth points."""
    path = C.FIGURES / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"saved {path}")


def reduce_memory(df, verbose: bool = True):
    """Downcast numeric dtypes. Handy on wide tabular data in a 16GB kernel."""
    start = df.memory_usage(deep=True).sum() / 1024**2
    for col in df.columns:
        t = df[col].dtype
        if str(t).startswith("int"):
            df[col] = df[col].astype(np.int32) if df[col].abs().max() < 2**31 else df[col]
        elif str(t).startswith("float"):
            df[col] = df[col].astype(np.float32)
    end = df.memory_usage(deep=True).sum() / 1024**2
    if verbose:
        print(f"memory {start:.1f}MB -> {end:.1f}MB")
    return df
