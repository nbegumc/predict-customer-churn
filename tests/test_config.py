"""Import-time behaviour of `src/`, which the serving image depends on.

Two properties are load-bearing and neither is obvious from reading the code:

1. Importing anything from `src/` must not need `data/raw/` to exist, or the
   API could not start and CI could not run. Failures must happen on *call*.
2. The inference path must not drag in sklearn, catboost or the notebook stack,
   or the serving image goes from ~400MB to ~2.5GB. `src/cv.py` is the module
   that pulls sklearn, which is exactly why inference never imports it.

Both run in a subprocess, because `sys.modules` is process-global and another
test importing `src.cv` first would otherwise make (2) pass or fail by accident.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from src import config as C

SERVING_MODULES = ["src.config", "src.utils", "src.data", "src.features"]

# Anything whose presence in the serving image would be a mistake.
HEAVY = ["sklearn", "catboost", "matplotlib", "seaborn", "shap", "mlflow", "evidently", "torch"]


def run_python(code: str) -> subprocess.CompletedProcess:
    """Run a snippet in a fresh interpreter rooted at the repo."""
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=C.ROOT,
        check=False,
    )


def test_serving_modules_import_without_data():
    """CI has no data/raw at all, so this is a real test there."""
    done = run_python(f"import importlib\nfor m in {SERVING_MODULES!r}: importlib.import_module(m)")
    assert done.returncode == 0, f"import failed:\n{done.stderr}"


def test_serving_imports_stay_light():
    """The one line that keeps the serving image small as the code changes."""
    done = run_python(
        "import importlib, sys\n"
        f"for m in {SERVING_MODULES!r}: importlib.import_module(m)\n"
        f"heavy = [h for h in {HEAVY!r} "
        "if any(k == h or k.startswith(h + '.') for k in sys.modules)]\n"
        "print(','.join(heavy))"
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "", (
        f"the inference path now imports {done.stdout.strip()}; "
        "that lands in the serving image. Import it lazily inside a function instead."
    )


def test_cv_is_the_module_that_pulls_sklearn():
    """The contrapositive, so the reason for the boundary is written down.

    Not a complaint about cv.py — it is correct for cv.py to use sklearn. It is
    a record of why `src/predict.py` must not import it.
    """
    done = run_python(
        "import src.cv, sys\nprint('sklearn' in sys.modules or 'sklearn.metrics' in sys.modules)"
    )
    assert done.stdout.strip() == "True"


def test_missing_data_fails_on_call_not_on_import(monkeypatch, tmp_path):
    """`src.data.load` should raise a plain FileNotFoundError, not something odd."""
    from src import data

    monkeypatch.setattr(C, "TRAIN_FILE", tmp_path / "nope.csv")
    monkeypatch.setattr(C, "TEST_FILE", tmp_path / "also-nope.csv")
    with pytest.raises(FileNotFoundError):
        data.load_raw()


def test_config_paths_are_all_under_the_repo():
    """Nothing should resolve outside the checkout when not on Kaggle."""
    assert not C.ON_KAGGLE, "this test suite assumes it is not running on Kaggle"
    for path in (C.MODEL_DIR, C.EXPERIMENTS, C.SUBMISSIONS, C.FIGURES, C.PRED_LOG):
        assert C.ROOT in path.parents or path.parent == C.ROOT, f"{path} escapes {C.ROOT}"


def test_new_production_paths_exist():
    """config.py creates these at import, so later phases can assume them."""
    assert C.MODEL_DIR.is_dir()
    assert C.PRED_LOG.parent.is_dir()
    assert C.MLFLOW_TRACKING_URI.startswith(("sqlite://", "http://", "https://", "file://"))
    assert C.MLFLOW_EXPERIMENT == C.COMPETITION
