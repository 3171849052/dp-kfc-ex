"""Four GPU workers; two fresh run processes in order on each GPU."""
import argparse
import os
import subprocess
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from exp33b import config as cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, choices=tuple(cfg.GPU_RUNS), required=True)
    args = parser.parse_args()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu))
    for name in cfg.GPU_RUNS[args.gpu]:
        subprocess.run([sys.executable, '-B', str(cfg.ROOT / 'run.py'), '--method', name],
                       env=env, check=True)


if __name__ == '__main__':
    main()
