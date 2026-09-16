"""Exp23: public proxy mismatch with paired MNIST experiments."""
import os
import sys
import tempfile
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT/'src'))
for key, folder in {'XDG_CACHE_HOME': '.cache', 'MPLCONFIGDIR': '.cache/matplotlib',
                    'CUDA_CACHE_PATH': '.cache/cuda', 'TMPDIR': '.cache/tmp'}.items():
    path = HERE/folder
    path.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(path)
tempfile.tempdir = str(HERE/'.cache/tmp')
