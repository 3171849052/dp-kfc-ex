"""Isolated runtime boundary for the ExpM1b experiments."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
CACHE_ROOT = ROOT / ".cache"


def configure_environment() -> None:
    """Route every library/runtime cache used by this experiment under ExpM1b."""
    paths = {
        "XDG_CACHE_HOME": CACHE_ROOT,
        "MPLCONFIGDIR": CACHE_ROOT / "matplotlib",
        "HF_HOME": CACHE_ROOT / "huggingface",
        "HF_HUB_CACHE": CACHE_ROOT / "huggingface" / "hub",
        "HUGGINGFACE_HUB_CACHE": CACHE_ROOT / "huggingface" / "hub",
        "HF_XET_CACHE": CACHE_ROOT / "huggingface" / "xet",
        "TORCH_HOME": CACHE_ROOT / "torch",
        "CUDA_CACHE_PATH": CACHE_ROOT / "cuda",
        "TRITON_CACHE_DIR": CACHE_ROOT / "triton",
        "TORCHINDUCTOR_CACHE_DIR": CACHE_ROOT / "torchinductor",
        "NUMBA_CACHE_DIR": CACHE_ROOT / "numba",
        "TMPDIR": CACHE_ROOT / "tmp",
        "TMP": CACHE_ROOT / "tmp",
        "TEMP": CACHE_ROOT / "tmp",
    }
    for key, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    tempfile.tempdir = str(CACHE_ROOT / "tmp")


sys.dont_write_bytecode = True
source_root = str(REPO_ROOT / "src")
if source_root not in sys.path:
    sys.path.insert(0, source_root)
configure_environment()

