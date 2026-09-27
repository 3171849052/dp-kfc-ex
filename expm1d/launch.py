"""Four static GPU queues; two fresh subprocess runs per queue."""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from expm1d import config as cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, choices=cfg.PHYSICAL_GPUS)
    args = parser.parse_args()
    if args.gpu is not None:
        for run in cfg.GRID:
            if run.gpu == args.gpu:
                with (cfg.LOGS_ROOT / f'{run.name}.log').open('x') as log:
                    subprocess.run([sys.executable, '-B', str(cfg.ROOT / 'worker.py'),
                                    '--source', run.source, '--frequency', run.frequency],
                                   env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu)),
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
        return
    for run in cfg.GRID:
        assert not run.directory.exists(), run.directory
        assert not (cfg.LOGS_ROOT / f'{run.name}.log').exists(), run.name
    # Prepare the shared checkpoint once before starting the four queues.
    from expm1d import data, vit
    data.private_datasets('vit')
    data.public_dataset('vit')
    vit.prepare_checkpoint()
    cfg.LOGS_ROOT.mkdir(parents=True, exist_ok=True)
    (cfg.RESULTS_ROOT / 'runs').mkdir(parents=True, exist_ok=True)
    processes = [subprocess.Popen([sys.executable, '-B', __file__, '--gpu', str(gpu)])
                 for gpu in cfg.PHYSICAL_GPUS]
    codes = [process.wait() for process in processes]
    if any(codes):
        raise SystemExit('A GPU queue failed; analysis is stopped.')


if __name__ == '__main__':
    main()
