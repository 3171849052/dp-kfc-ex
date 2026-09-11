#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_CONFIG="$ROOT/configs/standalone/mnist_dp_equil.yaml"
if [[ $# -eq 0 ]]; then
  CONFIG="$DEFAULT_CONFIG"
elif [[ $# -eq 2 && $1 == --config ]]; then
  CONFIG="$2"
elif [[ $# -eq 1 && $1 != --* ]]; then
  CONFIG="$1"
else
  echo "usage: $0 [--config CONFIG]" >&2
  exit 2
fi
CONFIG="$(realpath "$CONFIG")"
command -v tmux >/dev/null || { echo 'tmux is required' >&2; exit 1; }
cd "$ROOT"
GPU="$(conda run -n curve python scripts/train.py --config "$CONFIG" --print-gpu)"
conda run -n curve python scripts/train.py --config "$CONFIG" --validate-gpu >/dev/null
RUN_DIR="$(conda run -n curve python scripts/train.py --config "$CONFIG" --prepare-run)"
SESSION="$(conda run -n curve python scripts/train.py --tmux-session-name "$RUN_DIR")"
TRAIN_LOG="$RUN_DIR/train.log"
if tmux has-session -t "=$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi
CONDA="$(command -v conda)"
printf -v COMMAND 'cd %q && set -o pipefail && %q run --no-capture-output -n curve python -u scripts/train.py --config %q --run-dir %q 2>&1 | tee -a %q' \
  "$ROOT" "$CONDA" "$CONFIG" "$RUN_DIR" "$TRAIN_LOG"
printf -v SHELL_COMMAND 'bash -c %q' "$COMMAND"
tmux new-session -d -s "$SESSION" "$SHELL_COMMAND"
echo "physical GPU: $GPU"
echo "tmux session: $SESSION"
echo "run directory: $RUN_DIR"
echo "log: $TRAIN_LOG"
echo "attach: tmux attach -t $SESSION"
echo "tail: tail -f $TRAIN_LOG"
echo "kill: tmux kill-session -t $SESSION"
