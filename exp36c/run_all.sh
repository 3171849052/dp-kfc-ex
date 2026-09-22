#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=0
export PYTHONDONTWRITEBYTECODE=1
mkdir -p exp36c/logs
for method in oracle_dp_kfc_a_alpha025 oracle_dp_kfc_a_alpha05; do
    python -B exp36c/run.py --method "$method" > "exp36c/logs/${method}.log" 2>&1
done
python -B exp36c/analyze.py > exp36c/logs/analyze.log 2>&1
