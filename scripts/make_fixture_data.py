"""Expand the committed test fixture into a train/test/sample_submission trio.

    uv run python scripts/make_fixture_data.py /tmp/ci-raw
    CHURN_RAW_DIR=/tmp/ci-raw uv run python -m src.train --model lgbm --rounds 40

CI needs *a* trained bundle to build and smoke-test the serving image, and it
cannot have the real one: the bundle is ~30MB against a 1MB pre-commit limit and
`models/` is gitignored, so no clone ever contains it. The 200-row fixture
trains in under a second and exercises exactly the same code path.

Nothing about this is a stand-in for the real model — the image CI builds is
thrown away. CD builds the shippable image from a bundle attached to a GitHub
Release. See docs/adr/0007-ci-cd.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "synthetic_raw.csv"


def main(out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(FIXTURE)
    df.to_csv(out / "train.csv", index=False)
    # The real test.csv has no target column; mirror that so build_features
    # derives the same feature list it would in a real run.
    df.drop(columns=["Churn"]).to_csv(out / "test.csv", index=False)
    # Header casing matters: train.py writes whichever column is not ID_COL,
    # which is the bug that once produced a clean upload scoring 0.5.
    pd.DataFrame({"id": df["id"], "Churn": 0.0}).to_csv(out / "sample_submission.csv", index=False)

    print(f"wrote {len(df)} rows to {out}/[train|test|sample_submission].csv")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/ci-raw")
