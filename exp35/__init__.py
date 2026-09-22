"""Exp35 isolated output and runtime environment."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.dont_write_bytecode = True

def configure_environment():
    for key, relative in {
        'XDG_CACHE_HOME': '', 'MPLCONFIGDIR': 'matplotlib',
        'HF_HOME': 'huggingface', 'HF_HUB_CACHE': 'huggingface/hub',
        'HUGGINGFACE_HUB_CACHE': 'huggingface/hub', 'HF_XET_CACHE': 'huggingface/xet',
        'TORCH_HOME': 'torch', 'CUDA_CACHE_PATH': 'cuda', 'TRITON_CACHE_DIR': 'triton',
        'TORCHINDUCTOR_CACHE_DIR': 'torchinductor', 'TMPDIR': 'tmp', 'TMP': 'tmp', 'TEMP': 'tmp',
    }.items():
        path = ROOT / '.cache' / relative
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['HF_HUB_OFFLINE'] = '1'

configure_environment()
