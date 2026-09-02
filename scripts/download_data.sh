#!/usr/bin/env bash
# Download competition data. Needs ~/.kaggle/kaggle.json (chmod 600).
set -euo pipefail

COMP=$(uv run python -c "import sys; sys.path.insert(0,'.'); from src.config import COMPETITION; print(COMPETITION)")

if [ "$COMP" = "CHANGE-ME" ]; then
  echo "Set COMPETITION in src/config.py first." >&2
  exit 1
fi

mkdir -p data/raw
uv run kaggle competitions download -c "$COMP" -p data/raw
unzip -o "data/raw/${COMP}.zip" -d data/raw
rm -f "data/raw/${COMP}.zip"
echo "Data in data/raw:"
ls -lh data/raw
