"""Fresh subprocess per run, all on physical GPU 2."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exp35 import ROOT
from exp35 import config as cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=('mnist', 'vit'))
    parser.add_argument('--method')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.GPU)
    if args.dataset:
        assert args.method in cfg.grid(args.dataset)
        import torch
        torch.set_num_threads(4)
        if args.dataset == 'mnist':
            from exp35.mnist import run
        else:
            from exp35.vit import run
        run(args.method)
    else:
        for dataset in ('mnist', 'vit'):
            for method in cfg.grid(dataset):
                with (ROOT / 'logs' / f'{dataset}_{method}.log').open('w') as log:
                    print(f'GPU 2: {dataset}/{method}', flush=True)
                    subprocess.run([sys.executable, '-B', str(ROOT / 'worker.py'),
                                    '--dataset', dataset, '--method', method],
                                   stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == '__main__':
    main()
