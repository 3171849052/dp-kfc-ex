"""Sequential fresh subprocesses; paired rotating method order."""
import argparse
import os
from pathlib import Path
import subprocess
import shutil
import sys
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from exp21.config import SEEDS, METHODS, METHOD_ORDER


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--output', type=Path, help='New experiment root; must not already exist')
    args = parser.parse_args()
    output_args = []
    if args.output:
        args.output = args.output.resolve()
        args.output.mkdir(parents=True, exist_ok=False)
        shutil.copy2(ROOT/'exp21/results/correctness.json', args.output/'correctness.json')
        shutil.copytree(ROOT/'exp21/results/integration', args.output/'integration')
        output_args = ['--output', str(args.output)]
    for seed in ((42,) if args.smoke else SEEDS):
        for method in (METHODS if args.smoke else METHOD_ORDER[seed]):
            cmd = [sys.executable, '-B', str(ROOT/'exp21/run_one.py'), '--method', method, '--seed', str(seed)] + output_args
            if args.smoke:
                cmd.append('--smoke')
            subprocess.run(cmd, cwd=ROOT, check=True, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
    subprocess.run([sys.executable, '-B', str(ROOT/'exp21/analyze.py')] + output_args + (['--smoke'] if args.smoke else []), cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
