#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HOME="$PWD/exp25/cache/huggingface"
export TORCH_HOME="$PWD/exp25/cache/torch"
mode=formal
seeds=(42 7 91 23 58)
extra=()
if [[ "${1:-}" == --smoke ]]; then
  mode=smoke
  seeds=(42)
  extra=(--smoke)
  shift
fi
[[ $# == 0 ]] || { echo 'Usage: run_all.sh [--smoke]' >&2; exit 2; }
mkdir -p "exp25/results/$mode/logs"
for seed in "${seeds[@]}"; do
  for method in dp_sgd_bk_gd dp_kfc_a_public_bk_gd dp_kfc_a_pink_bk_gd dp_kfc_a_oracle_bk_gd dp_kfc_public_bk_gd dp_kfc_pink_bk_gd dp_kfc_oracle_bk_gd; do
    python -B exp25/run_one.py --method "$method" --seed "$seed" "${extra[@]}" 2>&1 | tee "exp25/results/$mode/logs/${method}_${seed}.log"
  done
done
