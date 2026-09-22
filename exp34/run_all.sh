#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
RESULTS_DIR="exp34/results_clip2"
LOG_DIR="exp34/logs_clip2"
mkdir -p "$LOG_DIR" "$RESULTS_DIR/runs"

# Validate the prepared dataset once before concurrent readers; Exp34 never writes it.
python -B - <<'PY'
from exp34.run import load_data
from exp34 import config as cfg
for _, method, damping in cfg.grid():
    path = cfg.RESULTS / "runs" / cfg.run_name(method, damping)
    if path.exists():
        raise FileExistsError(f"formal run already exists: {path}")
load_data(download=False)
PY

pids=()
for gpu in 0 1; do
    python -B exp34/worker.py --gpu "$gpu" >"$LOG_DIR/gpu${gpu}.log" 2>&1 &
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
    echo "Exp34 worker failed; see $LOG_DIR/gpu*.log" >&2
    exit "$status"
fi
python -B exp34/analyze.py
