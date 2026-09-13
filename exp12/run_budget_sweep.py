"""Run the full k=1/3/9, label-seed sweep; all diagnostic CLI options pass through.

Examples:
  python exp12/run_budget_sweep.py
  python exp12/run_budget_sweep.py --checkpoint PATH --warmup 2 --repeats 5
"""
import sys
from run_diagnostics import main

if __name__ == '__main__':
    main(sys.argv[1:])
