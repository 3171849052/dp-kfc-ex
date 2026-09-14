#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
for p in 0.0 0.25 0.5 0.75 1.0; do
  for seed in 0 1; do
    python exp15/run_exp15.py --p "$p" --seed "$seed"
  done
done
python exp15/run_exp15.py --summarize
