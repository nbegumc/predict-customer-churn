#!/usr/bin/env bash
# Submit a csv:  bash scripts/submit.sh submissions/sub_xxx.csv "lgbm baseline"
set -euo pipefail
FILE=$1
MSG=${2:-"submission"}
COMP=$(uv run python -c "import sys; sys.path.insert(0,'.'); from src.config import COMPETITION; print(COMPETITION)")
uv run kaggle competitions submit -c "$COMP" -f "$FILE" -m "$MSG"
echo "Remember to paste the public LB score into experiments.csv"
