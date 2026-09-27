"""Static four-GPU phase launcher; each GPU's runs execute sequentially."""
from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from expm1c import config as cfg


def grid(phase):
    if phase == 'beta':
        return cfg.BETA_GRID
    from expm1c.analyze_beta import beta_table, select_betas
    selected = cfg.selected_betas()
    assert selected == select_betas(beta_table())  # All Phase A runs must be complete.
    return cfg.extra_lambda_grid(selected) if phase == 'lambda_extra' else cfg.lambda_grid(selected)


def main():
    phase = sys.argv[1]
    assert phase in ('beta', 'lambda', 'lambda_extra')
    runs = grid(phase)
    if len(sys.argv) == 3:
        gpu = int(sys.argv[2])
        for run in runs:
            if run.gpu != gpu:
                continue
            with (cfg.LOGS_ROOT / phase / f'{run.name}.log').open('x') as log:
                subprocess.run([sys.executable, '-B', str(cfg.ROOT / 'worker.py'),
                    '--source', run.source, '--beta', str(run.beta),
                    '--damping', str(run.damping), '--phase', run.phase],
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)),
                    stdout=log, stderr=subprocess.STDOUT, check=True)
        return
    for run in runs:
        assert not run.directory.exists(), run.directory
        assert not (cfg.LOGS_ROOT / phase / f'{run.name}.log').exists(), run.name
    from expm1c import data, vit
    data.private_datasets('vit')
    data.public_dataset('vit')
    vit.prepare_checkpoint()
    (cfg.LOGS_ROOT / phase).mkdir(parents=True, exist_ok=True)
    (cfg.RESULTS_ROOT / phase).mkdir(parents=True, exist_ok=True)
    processes = [subprocess.Popen([sys.executable, '-B', __file__, phase, str(gpu)])
                 for gpu in cfg.PHYSICAL_GPUS]
    codes = [process.wait() for process in processes]
    if any(codes):
        raise SystemExit('A GPU worker failed; subsequent phases are stopped.')


if __name__ == '__main__':
    main()
