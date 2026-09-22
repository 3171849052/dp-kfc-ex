#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TMPDIR="$PWD/exp35b/.cache/tmp"
mkdir -p "$TMPDIR" exp35b/logs
run_gpu() {
    local gpu="$1" variant="$2"
    for dataset in mnist vit; do
        CUDA_VISIBLE_DEVICES="$gpu" python -B exp35b/worker.py --dataset "$dataset" --variant "$variant" > "exp35b/logs/${dataset}_dp_kfc_a_${variant}.log" 2>&1
    done
}
run_gpu 1 alpha025 &
pid025=$!
run_gpu 2 alpha075 &
pid075=$!
status=0
wait "$pid025" || status=1
wait "$pid075" || status=1
if (( status != 0 )); then exit "$status"; fi
python -B exp35b/analyze.py > exp35b/logs/analyze.log 2>&1
