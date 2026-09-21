#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$PWD/exp33/.cache/tmp" TMP="$PWD/exp33/.cache/tmp" TEMP="$PWD/exp33/.cache/tmp"
mkdir -p exp33/logs exp33/results/runs "$TMPDIR"
python -B - <<'PY'
from exp33.run import load_data
from exp33 import config as cfg
for method in cfg.METHODS:
    if (cfg.RESULTS / 'runs' / method).exists():
        raise FileExistsError(method)
load_data(download=True)
PY
pids=()
for gpu in 0 1 2 3; do
    python -B exp33/worker.py --gpu "$gpu" >"exp33/logs/gpu${gpu}.log" 2>&1 &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
    if wait "$pid"; then :; else status=1; fi
done
if [[ "$status" -ne 0 ]]; then
    echo 'Exp33 worker failed; see exp33/logs/gpu*.log' >&2
    exit "$status"
fi
python -B exp33/analyze.py
