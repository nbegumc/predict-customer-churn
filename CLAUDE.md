# Working in this repo

Kaggle/datathon competition repo. Human docs: `README.md` (why) and
`CHEATSHEET.md` (commands). Don't duplicate their command lists here.

## Before suggesting anything

Read `experiments.md` first. It records what has already been tried and what
failed — check the **Dead ends** section before proposing an approach, and
don't re-suggest something already listed there.

## Environment

- Run all hooks with `make lint` before saying work is finished.
- uv only. `uv run python -m src.train`, `uv add <pkg>`. Never `pip install`,
  never bare `python` (system python may be old; the project is 3.11).
- `requirements.txt` is generated (`make requirements`), not edited by hand.
- Dependencies live in `pyproject.toml`; `uv.lock` is committed.

## Invariants — do not change without the user explicitly asking

- **`CV_SCHEME` and `N_FOLDS` in `src/config.py`.** Changing them mid-competition
  invalidates every saved OOF file and makes past runs unblendable.
- **`src/cv.py`.** Written once on day one. Same reason.
- **Files in `experiments/`.** Every `oof_*.npy` and `test_*.npy` is blend
  material until the last day. Never clean them up.
- **`data/raw/`.** Immutable and gitignored. Never commit data, never force-add.

## Conventions

- All paths and settings come from `src/config.py`. Nothing hardcodes a path.
- Every training run goes through `src.train.run` so it logs to
  `experiments.csv` and saves OOF/test predictions. No ad-hoc training in a
  notebook cell that doesn't get logged.
- Always pass `--notes` describing what changed vs. the previous run.
- One change per run. Two changes in one run is one measurement, not two.
- Save figures to `reports/figures/` as they're made, not regenerated later.
- Refactor a notebook cell into `src/` when it's been copy-pasted twice, not before.
- Notebooks are stripped of outputs by the `nbstripout` pre-commit hook.
- Formatting and linting are ruff's job, configured in `pyproject.toml`
  (`line-length = 100`). Don't hand-format to a different style and don't add
  `# noqa` to silence a rule — either fix it or, if the rule is genuinely wrong
  for this repo, add it to `ignore` with a comment.
- Pre-commit hooks run on commit. If one rewrites a file, re-stage and retry
  rather than reaching for `--no-verify`; that flag is the user's call, not
  yours.

## After a run

1. Note the OOF score against the previous run's.
2. Public LB scores go in via `uv run python -m src.log_lb <run_id> <score>`,
   never by hand-editing `experiments.csv`.
3. Append the reasoning to `experiments.md` — hypothesis, result, keep/drop.
   The CSV holds numbers; this file holds *why*.
4. If it failed, add a line to **Dead ends** so it isn't tried twice.
