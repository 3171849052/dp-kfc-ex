"""Exp27: paired MNIST clipping-norm / learning-rate grid."""

import os
from pathlib import Path
import sys

# Keep import bytecode and plotting caches inside this experiment.
sys.dont_write_bytecode = True
os.environ["MPLCONFIGDIR"] = str(Path(__file__).resolve().parent / ".mplconfig")
