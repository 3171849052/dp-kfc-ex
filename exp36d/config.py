"""Fixed C=2 raw fractional geometry experiment; all writes stay local."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
from exp35b.runtime import configure

ROOT = Path(__file__).resolve().parent
cfg = configure(ROOT)
POWERS = (.1, .2, .3, .4, .5)
RUNS = {'dp_adamw': ('baseline', None)}
RUNS.update({f'dp_kfc_a_p0{i}': ('a_only', p) for i, p in enumerate(POWERS, 1)})
GPU_RUNS = {
    0: ('dp_adamw', 'dp_kfc_a_p01'),
    1: ('dp_kfc_a_p02', 'dp_kfc_a_p05'),
    2: ('dp_kfc_a_p03',),
    3: ('dp_kfc_a_p04',),
}
GPU_BY_RUN = {method: gpu for gpu, methods in GPU_RUNS.items() for method in methods}
cfg.MAX_GRAD_NORM = 2.0
cfg.METHODS = tuple(RUNS)
cfg.EXP22_METHOD = {m: {'baseline': 'dp_sgd', 'a_only': 'dp_kfc_a_bk'}[family]
                    for m, (family, _) in RUNS.items()}
cfg.grid = lambda dataset: cfg.METHODS
