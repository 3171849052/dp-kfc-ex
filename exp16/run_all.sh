#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
python exp16/run_exp16.py --mode hessian --q 0 --seed 42
for q in 0.25 0.5; do
  for mode in hessian fisher nested; do
    python exp16/run_exp16.py --mode "$mode" --q "$q" --seed 42
  done
done
python exp16/run_exp16.py --summarize
