"""Shared fixtures.

Everything here works without `data/raw/`, which is gitignored and therefore
absent in CI. Tests that genuinely need the real 77MB table are marked
`@pytest.mark.realdata` and skip when it is missing.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src import config as C

FIXTURES = Path(__file__).parent / "fixtures"

# The 15 categorical columns and every level each one takes, read off the real
# table. The fixture must cover all of them: a missing level makes
# `pd.Categorical` emit NaN instead, and any test asserting on categorical
# handling would then pass vacuously.
EXPECTED_LEVELS = {
    "gender": {"Male", "Female"},
    "partner": {"Yes", "No"},
    "dependents": {"Yes", "No"},
    "phoneservice": {"Yes", "No"},
    "multiplelines": {"No", "Yes", "No phone service"},
    "internetservice": {"DSL", "Fiber optic", "No"},
    "onlinesecurity": {"No", "Yes", "No internet service"},
    "onlinebackup": {"No", "Yes", "No internet service"},
    "deviceprotection": {"No", "Yes", "No internet service"},
    "techsupport": {"No", "Yes", "No internet service"},
    "streamingtv": {"No", "Yes", "No internet service"},
    "streamingmovies": {"No", "Yes", "No internet service"},
    "contract": {"Month-to-month", "One year", "Two year"},
    "paperlessbilling": {"Yes", "No"},
    "paymentmethod": {
        "Mailed check",
        "Credit card (automatic)",
        "Electronic check",
        "Bank transfer (automatic)",
    },
}


@pytest.fixture(scope="session")
def synthetic_raw() -> pd.DataFrame:
    """200 seeded rows in the raw Kaggle shape, with the original header casing.

    Stands in for `data/raw/train.csv` everywhere the test only needs *a* frame
    of the right shape. Asserts level coverage on the way out, so a regenerated
    fixture that lost a level fails loudly here rather than quietly downstream.
    """
    df = pd.read_csv(FIXTURES / "synthetic_raw.csv")
    lowered = df.rename(columns={c: c.strip().lower().replace(" ", "_") for c in df.columns})
    missing = {
        col: sorted(levels - set(lowered[col].unique()))
        for col, levels in EXPECTED_LEVELS.items()
        if levels - set(lowered[col].unique())
    }
    assert not missing, f"fixture is missing categorical levels: {missing}"
    return df


@pytest.fixture(scope="session")
def real_train() -> pd.DataFrame:
    """The real training frame, or a skip. Only for `realdata`-marked tests."""
    if not C.TRAIN_FILE.exists():
        pytest.skip(f"{C.TRAIN_FILE} is absent (gitignored); run `make data` to fetch it")
    return pd.read_csv(C.TRAIN_FILE)
