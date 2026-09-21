#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
mkdir -p exp30/logs exp30/results/runs

# Download/validate once before concurrent readers; all writes stay in Exp30.
python -B - <<'PY'
from exp30.run import load_data
load_data(download=True)
PY

pids=()
for gpu in 0 1 2 3; do
    python -B exp30/worker.py --gpu "$gpu" >"exp30/logs/gpu${gpu}.log" 2>&1 &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
    if wait "$pid"; then
        :
    else
        status=1
    fi
done
if [[ "$status" -ne 0 ]]; then
    echo "Exp30 worker failed; see exp30/logs/gpu*.log" >&2
    exit "$status"
fi
python -B exp30/analyze.py
