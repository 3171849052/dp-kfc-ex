#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
mkdir -p exp34/logs exp34/results/runs

# Download/validate once before concurrent readers; all writes stay in Exp34.
python -B - <<'PY'
from exp34.run import load_data
from exp34 import config as cfg
for _, method, damping in cfg.grid():
    path = cfg.RESULTS / "runs" / cfg.run_name(method, damping)
    if path.exists():
        raise FileExistsError(f"formal run already exists: {path}")
load_data(download=True)
PY

pids=()
for gpu in 0 1 2 3; do
    python -B exp34/worker.py --gpu "$gpu" >"exp34/logs/gpu${gpu}.log" 2>&1 &
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
    echo "Exp34 worker failed; see exp34/logs/gpu*.log" >&2
    exit "$status"
fi
python -B exp34/analyze.py
