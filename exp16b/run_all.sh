#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
python exp16b/run_exp16b.py --mode hessian --q 0 --seed 42
for q in 0.25 0.5; do
  for mode in hessian fisher nested; do
    python exp16b/run_exp16b.py --mode "$mode" --q "$q" --seed 42
  done
done
python exp16b/run_exp16b.py --summarize
