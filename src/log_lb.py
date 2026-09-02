"""Fill in the public LB score for a run, and check CV against LB.

    uv run python -m src.log_lb 0822_143052_lgbm 0.87104
    uv run python -m src.log_lb                     # no args: just the comparison

Kaggle only tells you the LB score after you submit, so `src.train` writes the
row with `public_lb` empty. This fills it in without hand-editing the csv.

The comparison table is the point: if CV and LB move together, trust CV and
stop watching the leaderboard. If they don't, your folds don't mirror the test
split, and that is the most important thing you can find out early.
"""

from __future__ import annotations

import sys

import pandas as pd

from .utils import LOG_FILE


def set_lb(run_id: str, score: float) -> None:
    if not LOG_FILE.exists():
        sys.exit(f"{LOG_FILE.name} does not exist yet — run a training run first.")

    df = pd.read_csv(LOG_FILE)
    match = df["run_id"] == run_id
    if not match.any():
        known = ", ".join(df["run_id"].tail(5)) or "none"
        sys.exit(f"no run '{run_id}' in {LOG_FILE.name}. last few: {known}")

    previous = df.loc[match, "public_lb"].iloc[0]
    df.loc[match, "public_lb"] = score
    df.to_csv(LOG_FILE, index=False)

    was = "" if pd.isna(previous) else f" (was {previous})"
    cv_score = df.loc[match, "cv_score"].iloc[0]
    print(f"{run_id}: cv={cv_score:.6f}  public_lb={score:.6f}{was}")


def compare() -> None:
    if not LOG_FILE.exists():
        sys.exit(f"{LOG_FILE.name} does not exist yet — run a training run first.")

    df = pd.read_csv(LOG_FILE)
    scored = df.dropna(subset=["public_lb"]).copy()
    if scored.empty:
        print("no runs have a public LB score yet.")
        return

    scored["gap"] = (scored["public_lb"] - scored["cv_score"]).round(5)
    print(scored[["run_id", "cv_score", "public_lb", "gap"]].to_string(index=False))

    if len(scored) < 3:
        print(
            f"\n{len(scored)} scored run(s) — need at least 3 before the "
            "correlation means anything."
        )
        return

    r = scored["cv_score"].corr(scored["public_lb"], method="spearman")
    print(f"\nspearman(cv, public_lb) over {len(scored)} runs: {r:.3f}")
    if r >= 0.7:
        print("CV and LB rank runs the same way — select on CV and stop refreshing Kaggle.")
    else:
        print(
            "CV and LB DISAGREE on ranking. Before tuning anything else, work out why: "
            "the usual causes are folds that don't mirror the test split, a leak, or a "
            "public LB too small to mean anything. Write the diagnosis in experiments.md."
        )


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        compare()
    elif len(args) == 2:
        set_lb(args[0], float(args[1]))
        print()
        compare()
    else:
        sys.exit("usage: python -m src.log_lb [<run_id> <public_lb_score>]")
