"""Load and clean. Keep raw immutable; every transformation lives here."""

from __future__ import annotations

import pandas as pd

from . import config as C
from .utils import reduce_memory

# The label is text in the raw file; 1 = churned, the class ROC-AUC ranks toward.
TARGET_MAP = {"No": 0, "Yes": 1}


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(C.TRAIN_FILE)
    test = pd.read_csv(C.TEST_FILE)
    return train, test


def load_sample_submission() -> pd.DataFrame:
    return pd.read_csv(C.SAMPLE_SUB_FILE)


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case, strip and underscore the headers, in place.

    Split out of `basic_clean` because it is the *only* part of cleaning a
    serving path also needs — see the warning on `basic_clean` below.
    """
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Cleaning that applies identically to train and test.

    Anything fitted on data (imputation values, encoders, scalers) belongs in
    features.py inside the fold loop — otherwise you leak.

    NOT FOR SERVING. `drop_duplicates` below is correct for a training frame
    and wrong for a request: three identical customers in a batch collapse to
    one row, and the response silently comes back shorter than, and misaligned
    with, the request. `src/predict.py` calls `normalise_columns` instead.
    """
    df = normalise_columns(df.copy())
    df = df.drop_duplicates()
    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Map the "Yes"/"No" target to int8 1/0.

    The raw file stores the label as text. LightGBM will not fit on it and
    `roc_auc_score` will not score it, so this has to happen before anything
    else touches the column. Runs after reduce_memory, which would otherwise
    widen the int8 back to int32.
    """
    if C.TARGET not in df.columns or pd.api.types.is_numeric_dtype(df[C.TARGET]):
        return df
    df = df.copy()
    mapped = df[C.TARGET].map(TARGET_MAP)
    unmapped = df.loc[mapped.isna(), C.TARGET].unique()
    assert len(unmapped) == 0, f"unmapped {C.TARGET} values: {unmapped}"
    df[C.TARGET] = mapped.astype("int8")
    return df


def load(clean: bool = True, reduce_mem: bool = True):
    train, test = load_raw()
    if clean:
        train, test = basic_clean(train), basic_clean(test)
    if reduce_mem:
        train, test = reduce_memory(train, False), reduce_memory(test, False)
    train = encode_target(train)
    print(f"train {train.shape}  test {test.shape}")
    return train, test
