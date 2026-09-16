"""Fixed protocol; no automatic experiment launch."""
from exp23 import HERE, ROOT
METHODS = ('a_public_fashion', 'full_public_fashion', 'a_public_cifar10',
           'full_public_cifar10', 'a_pink', 'full_pink', 'dp_sgd')
SEEDS = (42, 7, 123)
EPOCHS = 5
BATCH_SIZE = 256
CALIBRATION_BATCHES = 10
CALIBRATION_BATCH_SIZE = 256
A_POWER = .4
DAMPING = 1e-3
EPSILON = 1.
DELTA = 1e-5
CLIP = 1.
LEARNING_RATE = .5
ORACLE_BATCHES = 10
ORACLE_SEED = 23000
MNIST_ROOT = ROOT/'exp1/data'
FASHION_ROOT = HERE/'data'
CIFAR_ROOT = ROOT/'exp22/data'
RESEARCH_ONLY = 'research-only: unnoised oracle/clipping/norm diagnostics; not a DP release'


def source(method):
    return 'none' if method == 'dp_sgd' else method.split('_')[-1]
