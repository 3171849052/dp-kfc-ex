"""Run the full k=1/3/9, label-seed sweep; all diagnostic CLI options pass through.

Examples:
  python exp12/run_budget_sweep.py
  python exp12/run_budget_sweep.py --checkpoint exp12/checkpoints/sgd_seed42_step000500.pt --output exp12/results/sgd_step500
"""
import sys
from run_diagnostics import main

if __name__ == '__main__':
    main(sys.argv[1:])
