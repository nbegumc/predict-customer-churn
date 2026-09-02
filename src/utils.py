"""Small helpers you will use in every notebook and script."""

from __future__ import annotations

import csv
import json
import os
import random
import time
from contextlib import contextmanager
from datetime import datetime

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
