"""Four static GPU queues, two sequential runs per queue."""
from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from expm2 import config as cfg


def main():
    if len(sys.argv) == 2:
        gpu = int(sys.argv[1])
        assert gpu in cfg.PHYSICAL_GPUS
        for run in cfg.GRID:
            if run.gpu == gpu:
                with (cfg.LOGS_ROOT / f'{run.name}.log').open('x') as log:
                    subprocess.run([sys.executable, '-B', str(cfg.ROOT / 'worker.py'),
                        '--source', run.source, '--beta', str(run.beta),
                        '--damping', str(run.damping)],
                        env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)),
                        stdout=log, stderr=subprocess.STDOUT, check=True)
        return
    assert len(sys.argv) == 1
    for run in cfg.GRID:
        assert not run.directory.exists(), run.directory
        assert not (cfg.LOGS_ROOT / f'{run.name}.log').exists(), run.name
    from expm2 import data, vit
    data.private_datasets('vit')
    data.public_dataset('vit')
    vit.prepare_checkpoint()
    cfg.LOGS_ROOT.mkdir(parents=True, exist_ok=True)
    (cfg.RESULTS_ROOT / 'runs').mkdir(parents=True, exist_ok=True)
    processes = [subprocess.Popen([sys.executable, '-B', __file__, str(gpu)])
                 for gpu in cfg.PHYSICAL_GPUS]
    codes = [process.wait() for process in processes]
    if any(codes):
        raise SystemExit('A GPU queue failed; analysis stopped.')


if __name__ == '__main__':
    main()
