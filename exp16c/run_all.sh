#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
for q in 0.25 0.5; do
  for mode in fisher nested_ag nested_a; do
    python exp16c/run_exp16c.py --mode "$mode" --q "$q" --seed 42
  done
done
python exp16c/run_exp16c.py --summarize
