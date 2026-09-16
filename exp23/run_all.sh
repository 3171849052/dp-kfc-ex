#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/src:$PWD"
export XDG_CACHE_HOME="$PWD/exp23/.cache"
export MPLCONFIGDIR="$PWD/exp23/.cache/matplotlib"
export CUDA_CACHE_PATH="$PWD/exp23/.cache/cuda"
export TMPDIR="$PWD/exp23/.cache/tmp"
mode="${1:-formal}"
[[ "$mode" == formal || "$mode" == --smoke ]]
seeds=(42 7 123)
args=()
output=exp23/results/formal
if [[ "$mode" == --smoke ]]; then
    seeds=(42)
    args=(--smoke)
    output=exp23/results/smoke
fi
mkdir -p "$output/logs" "$TMPDIR"
for seed in "${seeds[@]}"; do
    for method in a_public_fashion full_public_fashion a_public_cifar10 full_public_cifar10 a_pink full_pink dp_sgd; do
        python -B -m exp23.run_exp23 --method "$method" --seed "$seed" "${args[@]}" 2>&1 | tee "$output/logs/${method}_${seed}.log"
    done
done
python -B -m exp23.analyze "${args[@]}" 2>&1 | tee "$output/logs/analysis.log"
