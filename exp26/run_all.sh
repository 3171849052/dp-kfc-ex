#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
conda run --no-capture-output -n curve python -m exp26.run
conda run --no-capture-output -n curve python -m exp26.analyze
