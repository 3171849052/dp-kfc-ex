#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="$PWD/exp19/.cache"
export MPLCONFIGDIR="$PWD/exp19/.cache/matplotlib"
export CUDA_CACHE_PATH="$PWD/exp19/.cache/cuda"
export TMPDIR="$PWD/exp19/.cache/tmp"
mkdir -p "$TMPDIR"
args=()
seeds=(42 7 123 2024 3407)
if [[ ${1:-} == --smoke ]]; then
    args=(--smoke)
    seeds=(42)
    exec > >(tee exp19/smoke.log) 2>&1
else
    [[ $# == 0 ]]
    exec > >(tee exp19/formal.log) 2>&1
fi
for seed in "${seeds[@]}"; do
    for method in M0_original M1_full_ghost M2_a_ghost_p05 M3_a_ghost_p025; do
        python exp19/run_one.py --method "$method" --seed "$seed" "${args[@]}"
    done
done
python exp19/analyze.py "${args[@]}"
