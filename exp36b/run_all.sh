#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=0
export PYTHONDONTWRITEBYTECODE=1
mkdir -p exp36b/logs
python -B -c 'from exp36b.oracle_geometry import save_indices; save_indices()'
for method in dp_adamw oracle_dp_kfc_a oracle_dp_kfc; do
    python -B exp36b/run.py --method "$method" > "exp36b/logs/${method}.log" 2>&1
done
python -B exp36b/analyze.py > exp36b/logs/analyze.log 2>&1
