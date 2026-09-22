"""Fixed three-run protocol and experiment-local runtime."""
from pathlib import Path
from exp35b.runtime import configure
ROOT = Path(__file__).resolve().parent
cfg = configure(ROOT)
cfg.GPU = 0
cfg.RESULTS = ROOT / 'results'
cfg.METHODS = ('dp_adamw', 'oracle_dp_kfc_a', 'oracle_dp_kfc')
cfg.EXP22_METHOD = dict(zip(cfg.METHODS, ('dp_sgd', 'dp_kfc_a_bk', 'dp_kfc')))
METHODS = cfg.METHODS


def privacy(method):
    return {'privacy_status': 'standard_dp' if method == 'dp_adamw' else 'oracle_non_private_geometry',
            'nominal_dp_epsilon': 3}
