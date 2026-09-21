#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES=1
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$PWD/exp29/.mplconfig"
export XDG_CACHE_HOME="$PWD/exp29/.cache"
export HF_HOME="$PWD/exp29/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
export TORCH_HOME="$PWD/exp29/.cache/torch"
export TRITON_CACHE_DIR="$PWD/exp29/.cache/triton"
export CUDA_CACHE_PATH="$PWD/exp29/.cache/cuda"
export TMPDIR="$PWD/exp29/.tmp"
mkdir -p exp29/logs "$TMPDIR"
python -u -B -m exp29.run 2>&1 | tee exp29/logs/run.log
python -u -B -m exp29.analyze 2>&1 | tee exp29/logs/analyze.log
