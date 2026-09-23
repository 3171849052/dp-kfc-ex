#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
# Reject the complete launch before opening any per-run log.
for method in dp_adamw trace_a_p01 trace_a_p02 trace_a_p03 trace_a_p04 trace_a_p05 trace_ag_p01 trace_ag_p02 trace_ag_p03 trace_ag_p04 trace_ag_p05; do
    if [[ -e "exp37/results/vit/runs/$method" ]]; then
        echo "Run directory already exists: $method" >&2
        exit 1
    fi
done
mkdir -p exp37/logs
worker() {
    export CUDA_VISIBLE_DEVICES="$1"
    shift
    for method in "$@"; do
        python -B exp37/worker.py --method "$method" >"exp37/logs/${method}.log" 2>&1 || return "$?"
    done
}
worker 1 dp_adamw trace_ag_p01 trace_ag_p04 trace_a_p01 &
p1=$!
worker 2 trace_ag_p02 trace_ag_p05 trace_a_p02 trace_a_p04 &
p2=$!
worker 3 trace_ag_p03 trace_a_p03 trace_a_p05 &
p3=$!
status=0
for pid in "$p1" "$p2" "$p3"; do
    wait "$pid" || status=1
done
if (( status != 0 )); then exit "$status"; fi
python -B exp37/analyze.py >exp37/logs/analysis.log 2>&1
