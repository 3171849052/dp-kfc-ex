"""Sequential runs on one fixed GPU; subprocess failures are fatal."""
import argparse
import os
import subprocess
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from exp32c import config as cfg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, choices=tuple(cfg.GPU_RUNS), required=True)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    for method, damping in cfg.GPU_RUNS[args.gpu]:
        command = [sys.executable, '-B', str(cfg.ROOT / 'run.py'), '--method', method]
        if damping is not None:
            command += ['--damping', str(damping)]
        print(f'GPU {args.gpu}: {method} damping={damping}', flush=True)
        subprocess.run(command, check=True)

if __name__ == '__main__':
    main()
