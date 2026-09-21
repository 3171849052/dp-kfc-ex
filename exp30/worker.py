"""Sequential fresh subprocesses for one fixed GPU assignment."""
import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from exp30 import config as cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, choices=tuple(cfg.GPU_DAMPINGS), required=True)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    for damping in cfg.GPU_DAMPINGS[args.gpu]:
        for method in cfg.METHODS:
            print(f"GPU {args.gpu}: {method} damping={damping:g}", flush=True)
            subprocess.run([
                sys.executable, "-B", str(cfg.ROOT / "run.py"),
                "--method", method, "--damping", str(damping),
            ], check=True)


if __name__ == "__main__":
    main()
