"""Load and clean. Keep raw immutable; every transformation lives here."""

from __future__ import annotations

import pandas as pd

from . import config as C
from .utils import reduce_memory


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(C.TRAIN_FILE)
    test = pd.read_csv(C.TEST_FILE)
    return train, test


def load_sample_submission() -> pd.DataFrame:
    return pd.read_csv(C.SAMPLE_SUB_FILE)


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Cleaning that applies identically to train and test.

    Anything fitted on data (imputation values, encoders, scalers) belongs in
    features.py inside the fold loop — otherwise you leak.
    """
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df = df.drop_duplicates()
    return df


def load(clean: bool = True, reduce_mem: bool = True):
    train, test = load_raw()
    if clean:
        train, test = basic_clean(train), basic_clean(test)
    if reduce_mem:
        train, test = reduce_memory(train, False), reduce_memory(test, False)
    print(f"train {train.shape}  test {test.shape}")
    return train, test
