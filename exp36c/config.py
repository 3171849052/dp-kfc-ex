"""Two fixed runs, with all runtime writes isolated under Exp36c."""
import sys
from pathlib import Path
from exp35b.runtime import configure

ROOT = Path(__file__).resolve().parent
cfg = configure(ROOT)
cfg.GPU = 0
cfg.RESULTS = ROOT / 'results'
ALPHAS = {'oracle_dp_kfc_a_alpha025': .25, 'oracle_dp_kfc_a_alpha05': .5}
cfg.METHODS = tuple(ALPHAS)
cfg.VARIANTS = tuple(ALPHAS)
cfg.EXP22_METHOD = dict.fromkeys(ALPHAS, 'dp_kfc_a_bk')
cfg.grid = lambda dataset: cfg.METHODS
ORACLE_INDICES = ROOT.parent / 'exp36b/results/oracle_indices.csv'
# Reuse Exp36b helpers without executing its output/cache configuration.
sys.modules['exp36b.config'] = sys.modules[__name__]


def privacy(method):
    return dict(privacy_status='oracle_non_private_geometry', nominal_dp_epsilon=3)
