#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$PWD/exp33c/.cache/tmp" TMP="$PWD/exp33c/.cache/tmp" TEMP="$PWD/exp33c/.cache/tmp"
mkdir -p exp33c/logs exp33c/results/runs "$TMPDIR"
python -B - <<'PY'
from exp33c.run import load_data
from exp33c import config as cfg
for method in cfg.RUNS:
    if (cfg.RESULTS / 'runs' / method).exists():
        raise FileExistsError(method)
load_data()
PY
pids=()
for gpu in 0 1 2 3; do
    python -B exp33c/worker.py --gpu "$gpu" >"exp33c/logs/gpu${gpu}.log" 2>&1 &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
    if wait "$pid"; then :; else status=1; fi
done
if [[ "$status" -ne 0 ]]; then
    echo 'Exp33c worker failed; see exp33c/logs/gpu*.log' >&2
    exit "$status"
fi
python -B exp33c/analyze.py
