#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$PWD/exp32c/.cache/tmp"
export TMP="$TMPDIR" TEMP="$TMPDIR"
mkdir -p "$TMPDIR"
mkdir -p exp32c/logs exp32c/results/runs

# Validate the existing Exp32 dataset before concurrent read-only access.
python -B - <<'PY'
from exp32c.run import load_data
from exp32c import config as cfg
for _, method, damping in cfg.grid():
    path = cfg.RESULTS / 'runs' / cfg.run_name(method, damping)
    if path.exists():
        raise FileExistsError(f'formal run already exists: {path}')
from exp32c.model import initialize
initialize(cfg.SEED, "cpu")
load_data(download=False)
PY

pids=()
for gpu in 0 1 2 3; do
    python -B exp32c/worker.py --gpu "$gpu" >"exp32c/logs/gpu${gpu}.log" 2>&1 &
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
    echo "Exp32c worker failed; see exp32c/logs/gpu*.log" >&2
    exit "$status"
fi
python -B exp32c/analyze.py
