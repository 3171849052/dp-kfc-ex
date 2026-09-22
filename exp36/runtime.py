from pathlib import Path
from exp35b.runtime import configure
ROOT = Path(__file__).resolve().parent


def setup():
    cfg = configure(ROOT)
    cfg.GPU = 3
    cfg.METHODS = ('dp_adamw',)
    cfg.EXP22_METHOD = {'dp_adamw': 'dp_sgd'}
    return cfg
