#!/usr/bin/env bash
# Push src/ up as a Kaggle utility dataset so notebooks can import it.
# First time:  kaggle datasets create -p kaggle/utils
# After that:  this script
set -euo pipefail
rm -rf kaggle/utils/src
cp -r src kaggle/utils/src
uv run kaggle datasets version -p kaggle/utils -m "${1:-sync src/}"
