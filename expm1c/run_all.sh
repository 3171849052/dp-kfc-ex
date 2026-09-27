#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/src:$PWD"
bash expm1c/run_beta.sh
python -B expm1c/analyze_beta.py
bash expm1c/run_lambda.sh
python -B expm1c/analyze.py
