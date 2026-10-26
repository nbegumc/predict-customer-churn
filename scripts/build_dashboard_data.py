"""Aggregate the served model's OOF predictions into the campaign planner's data.

    uv run python -m scripts.build_dashboard_data

Writes four small CSVs to reports/dashboard/, which are uploaded as the data
files of the Churn Campaign Planner dashboard. They hold aggregates only -- no
customer-level rows -- so they are safe to share with the dashboard's readers.

Mirrors notebooks/04_business_case.ipynb: the same inputs, the same two
rankings (by p, and by p * monthly charges), the same cut-offs int(k * N). The
10% rows therefore reproduce the numbers in docs/use-case.md exactly, and the
dashboard's money figures are those of the notebook at its default assumptions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C

SERVED = "1024_141318_lgbm"  # the bundle in models/, and what the API serves
CHAMP = "0904_202729_lgbm"  # the run that scored public LB 0.91393
OUT = C.REPORTS / "dashboard"
STEPS = 1000  # 0.1% of the base per row: fine enough for a slider, small enough to ship


def planner_curve(y: np.ndarray, m: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    """Cumulative totals down the ranked call list, at every 0.1% of the base."""
    n_all = len(y)
    pm = p * m  # expected monthly revenue lost from this customer
    rows = []
    for name, key in [("risk", p), ("value", pm)]:
        order = np.argsort(-key)
        c_y, c_pm, c_p, c_ym = (np.cumsum(a[order]) for a in (y, pm, p, y * m))
        for k in range(1, STEPS + 1):
            n = int(k / STEPS * n_all)
            rows.append(
                {
                    "ranking": name,
                    "depth_pct": round(100 * k / STEPS, 1),
                    "customers": n,
                    "would_leave": int(c_y[n - 1]),
                    # x save rate = expected customers kept
                    "expected_saves_at_100pct": round(c_p[n - 1], 2),
                    # x save rate x months = expected revenue kept
                    "expected_monthly_loss": round(c_pm[n - 1], 2),
                    "leaver_revenue": round(c_ym[n - 1], 2),
                    "last_expected_loss": round(pm[order][n - 1], 4),
                }
            )
    return pd.DataFrame(rows)


def segments(raw: pd.DataFrame, y: np.ndarray, m: np.ndarray) -> pd.DataFrame:
    """Churn rate and leavers' revenue by the groups a retention team recognises."""
    df = raw.assign(
        y=y,
        ym=y * m,
        tenure_band=pd.cut(
            raw.tenure,
            [-1, 12, 24, 48, np.inf],
            labels=["0-12 months", "13-24 months", "25-48 months", "49+ months"],
        ),
    )
    rows = []
    for dim, col in [
        ("Contract", "Contract"),
        ("Tenure", "tenure_band"),
        ("Internet service", "InternetService"),
        ("Payment method", "PaymentMethod"),
    ]:
        for seg, d in df.groupby(col, observed=True):
            rows.append(
                {
                    "dimension": dim,
                    "segment": str(seg),
                    "customers": len(d),
                    "leavers": int(d.y.sum()),
                    "churn_rate": round(d.y.mean(), 4),
                    "leaver_revenue": round(d.ym.sum(), 2),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    raw = pd.read_csv(C.TRAIN_FILE)
    y = (raw.Churn == "Yes").astype("int8").to_numpy()
    m = raw.MonthlyCharges.to_numpy(dtype=float)  # monthly recurring revenue, GBP
    p = np.load(C.EXPERIMENTS / f"oof_{SERVED}.npy")
    assert np.array_equal(p, np.load(C.EXPERIMENTS / f"oof_{CHAMP}.npy")), (
        "served bundle's OOF differs from the champion's -- the dashboard would not describe it"
    )

    totals = pd.DataFrame(
        [
            {
                "customers": len(y),
                "leavers": int(y.sum()),
                "monthly_revenue": round(m.sum(), 2),
                "leaver_revenue": round(m[y == 1].sum(), 2),
                "avg_bill_leavers": round(m[y == 1].mean(), 2),
                "avg_bill_stayers": round(m[y == 0].mean(), 2),
            }
        ]
    )
    calibration = (
        pd.DataFrame({"decile": pd.qcut(p, 10, labels=False) + 1, "p": p, "y": y})
        .groupby("decile")
        .agg(customers=("y", "size"), predicted=("p", "mean"), observed=("y", "mean"))
        .round(4)
        .reset_index()
    )

    OUT.mkdir(parents=True, exist_ok=True)
    planner_curve(y, m, p).to_csv(OUT / "planner_curve.csv", index=False)
    totals.to_csv(OUT / "totals.csv", index=False)
    calibration.to_csv(OUT / "calibration.csv", index=False)
    segments(raw, y, m).to_csv(OUT / "segments.csv", index=False)
    print(f"wrote 4 files to {OUT}")


if __name__ == "__main__":
    main()
