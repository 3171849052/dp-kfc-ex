#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
mkdir -p exp36d/logs
worker() {
    export CUDA_VISIBLE_DEVICES="$1"
    shift
    for method in "$@"; do
        python -B exp36d/worker.py --method "$method" >"exp36d/logs/${method}.log" 2>&1 || return "$?"
    done
}
worker 0 dp_adamw dp_kfc_a_p01 &
p0=$!
worker 1 dp_kfc_a_p02 dp_kfc_a_p05 &
p1=$!
worker 2 dp_kfc_a_p03 &
p2=$!
worker 3 dp_kfc_a_p04 &
p3=$!
status=0
for pid in "$p0" "$p1" "$p2" "$p3"; do
    wait "$pid" || status=1
done
if (( status != 0 )); then exit "$status"; fi
python -B exp36d/analyze.py >exp36d/logs/analysis.log 2>&1
