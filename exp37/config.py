"""Exp36d protocol with a fixed eleven-run scalar sweep."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
from exp35b.runtime import configure

ROOT = Path(__file__).resolve().parent
cfg = configure(ROOT)
POWERS = (.1, .2, .3, .4, .5)
RUNS = {'dp_adamw': ('baseline', None)}
for family in ('trace_a', 'trace_ag'):
    RUNS.update({f'{family}_p0{i}': (family, p) for i, p in enumerate(POWERS, 1)})
GPU_RUNS = {
    1: ('dp_adamw', 'trace_ag_p01', 'trace_ag_p04', 'trace_a_p01'),
    2: ('trace_ag_p02', 'trace_ag_p05', 'trace_a_p02', 'trace_a_p04'),
    3: ('trace_ag_p03', 'trace_a_p03', 'trace_a_p05'),
}
GPU_BY_RUN = {m: gpu for gpu, methods in GPU_RUNS.items() for m in methods}
cfg.MAX_GRAD_NORM = 2.0
cfg.METHODS = tuple(RUNS)
cfg.EXP22_METHOD = {m: family for m, (family, _) in RUNS.items()}
cfg.grid = lambda dataset: cfg.METHODS
