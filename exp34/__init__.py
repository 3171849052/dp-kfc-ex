"""Exp34: random-init full-parameter private ViT training."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
for key, relative in {
    "XDG_CACHE_HOME": ".cache",
    "MPLCONFIGDIR": ".cache/matplotlib",
    "HF_HOME": ".cache/huggingface",
    "HF_HUB_CACHE": ".cache/huggingface/hub",
    "HUGGINGFACE_HUB_CACHE": ".cache/huggingface/hub",
    "HF_XET_CACHE": ".cache/huggingface/xet",
    "TORCH_HOME": ".cache/torch",
    "CUDA_CACHE_PATH": ".cache/cuda",
    "TRITON_CACHE_DIR": ".cache/triton",
    "TORCHINDUCTOR_CACHE_DIR": ".cache/torchinductor",
    "TMPDIR": ".cache/tmp",
    "TMP": ".cache/tmp",
    "TEMP": ".cache/tmp",
}.items():
    path = ROOT / relative
    path.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(path)
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

# tempfile may already be imported by the Python launcher.
import tempfile
tempfile.tempdir = str(ROOT / ".cache" / "tmp")
