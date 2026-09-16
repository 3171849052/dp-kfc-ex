#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT="${ROOT_DIR}/exp24/results"
OUTPUT_SET=0
SMOKE=0
SMOKE_BATCHES=2

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output)
      OUTPUT="${ROOT_DIR}/$2"
      OUTPUT_SET=1
      shift 2
      ;;
    --smoke)
      SMOKE=1
      shift
      ;;
    --smoke-batches)
      SMOKE_BATCHES="$2"
      shift 2
      ;;
    *)
      echo "unknown option: $1" >&2
      exit 2
      ;;
  esac
done

mkdir -p "$OUTPUT"
if [[ "$SMOKE" == 1 && "$OUTPUT_SET" == 0 ]]; then
  OUTPUT="${ROOT_DIR}/exp24/results/smoke"
  mkdir -p "$OUTPUT"
fi

export PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="${ROOT_DIR}/exp24/.cache"
export MPLCONFIGDIR="${ROOT_DIR}/exp24/.cache/matplotlib"
export CUDA_CACHE_PATH="${ROOT_DIR}/exp24/.cache/cuda"
export TMPDIR="${ROOT_DIR}/exp24/.cache/tmp"
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$CUDA_CACHE_PATH" "$TMPDIR"

if [[ "$SMOKE" == 1 ]]; then
  seeds=(42)
else
  seeds=(42 7 123 2024 3407)
fi

for seed in "${seeds[@]}"; do
  case "$seed" in
    42|2024) methods=(dp_sgd dp_kfc_a_bk dp_kfc) ;;
    7|3407) methods=(dp_kfc_a_bk dp_kfc dp_sgd) ;;
    123) methods=(dp_kfc dp_sgd dp_kfc_a_bk) ;;
  esac
  for method in "${methods[@]}"; do
    args=(--method "$method" --seed "$seed" --output "$OUTPUT")
    if [[ "$SMOKE" == 1 ]]; then
      args+=(--smoke --smoke-batches "$SMOKE_BATCHES")
    fi
    conda run -n curve python -B "$ROOT_DIR/exp24/run_one.py" "${args[@]}"
  done
done
