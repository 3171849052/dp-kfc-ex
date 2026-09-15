#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="$PWD/exp20/.cache"
export MPLCONFIGDIR="$PWD/exp20/.cache/matplotlib"
export CUDA_CACHE_PATH="$PWD/exp20/.cache/cuda"
export TMPDIR="$PWD/exp20/.cache/tmp"
mkdir -p "$TMPDIR"
args=()
if [[ ${1:-} == --smoke ]]; then
    [[ $# == 1 ]]
    args=(--smoke)
    exec > >(tee exp20/smoke.log) 2>&1
else
    [[ $# == 0 ]]
    exec > >(tee exp20/formal.log) 2>&1
fi
for p in 0 .125 .25 .375 .5 .625 .75 1; do
    python exp20/run_one.py --p "$p" --seed 42 "${args[@]}"
done
python exp20/analyze.py "${args[@]}"
