"""Sequential fresh subprocesses; paired rotating method order."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from exp21.config import SEEDS, METHODS, METHOD_ORDER


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    for seed in ((42,) if args.smoke else SEEDS):
        for method in (METHODS if args.smoke else METHOD_ORDER[seed]):
            cmd = [sys.executable, '-B', str(ROOT/'exp21/run_one.py'), '--method', method, '--seed', str(seed)]
            if args.smoke:
                cmd.append('--smoke')
            subprocess.run(cmd, cwd=ROOT, check=True, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
    subprocess.run([sys.executable, '-B', str(ROOT/'exp21/analyze.py')] + (['--smoke'] if args.smoke else []), cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
