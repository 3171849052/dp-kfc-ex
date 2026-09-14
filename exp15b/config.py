"""Fixed exp15 protocol with original K-BFGS damping parameterization."""
from math import sqrt
from pathlib import Path
from exp15.run_exp15 import configuration as exp15_configuration

ROOT = Path(__file__).resolve().parent
LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1, 3e-1)
SEEDS = (42, 7)


def configuration(lambda_damping, seed, smoke=False):
    c = exp15_configuration(.25, seed, smoke)
    c['lambda_damping'] = lambda_damping
    c['lbfgs']['factor_damping'] = sqrt(lambda_damping)
    c['data'].update(root=str(ROOT / 'data'), shuffle=True)
    c['output']['root'] = str(ROOT / 'results')
    return c
