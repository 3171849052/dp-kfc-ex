#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
mkdir -p exp32/logs exp32/results/runs

# Download/validate once before concurrent readers; all writes stay in Exp32.
python -B - <<'PY'
from exp32.run import load_data
from exp32 import config as cfg
for _, method, damping in cfg.grid():
    path = cfg.RESULTS / 'runs' / cfg.run_name(method, damping)
    if path.exists():
        raise FileExistsError(f'formal run already exists: {path}')
from exp32.model import initialize
initialize(cfg.SEED, "cpu")
load_data(download=True)
PY

pids=()
for gpu in 0 1 2 3; do
    python -B exp32/worker.py --gpu "$gpu" >"exp32/logs/gpu${gpu}.log" 2>&1 &
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
    echo "Exp32 worker failed; see exp32/logs/gpu*.log" >&2
    exit "$status"
fi
python -B exp32/analyze.py
