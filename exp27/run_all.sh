#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$PWD/exp27/.mplconfig"
export XDG_CACHE_HOME="$PWD/exp27/.cache"
export TMPDIR="$PWD/exp27/.tmp"
mkdir -p exp27/logs "$TMPDIR"
python -u -B -m exp27.run 2>&1 | tee exp27/logs/run.log
python -u -B -m exp27.analyze 2>&1 | tee exp27/logs/analyze.log
