"""Second-resolution run names and durable epoch metrics."""
import csv
from copy import deepcopy
import math
from datetime import datetime, timedelta
from decimal import Decimal
import json
import os
from pathlib import Path
import re
import yaml

METRICS_FIELDS = tuple('epoch global_step train_loss test_loss test_accuracy epsilon_spent noise_multiplier sample_rate clip_fraction mean_clip_factor precond_build_seconds epoch_train_seconds epoch_total_seconds peak_allocated_mb_epoch preconditioner_storage_mb gain_p10 gain_median gain_p90 gain_max'.split())


def format_number(value):
    number = Decimal(str(value))
    if number == number.to_integral_value():
        return str(number.quantize(Decimal('1')))
    if abs(number) < Decimal('0.001'):
        return re.sub(r'e([+-])0+(\d+)$', r'e\1\2', format(number.normalize(), 'E').lower()).replace('e+', 'e')
    return format(number.normalize(), 'f')


def format_run_name(c, timestamp=None):
    t, p, s = c['training'], c['privacy'], c['synthetic']
    tokens = [(timestamp or datetime.now()).strftime('%m%d-%H%M%S'),
              c['model']['name'], c['data']['dataset'], c['algorithm']]
    values = [('s', c['seed']), ('ep', t['epochs']), ('bs', c['data']['batch_size']),
              ('lr', t['learning_rate']), ('mom', t['momentum']), ('eps', p['epsilon']),
              ('d', p['delta']), ('C', p['max_grad_norm'])]
    if c['algorithm'] != 'dp_sgd':
        values += [('M', s['samples']), ('U', s['refresh_every_epochs'])]
    if c['algorithm'] == 'dp_kfc':
        values += [('damp', c['kfac']['damping'])]
    if c['algorithm'] == 'dp_equil':
        values += [('K', c['equil']['probes']), ('tau', c['equil']['tau'])]
    tokens += [key + format_number(value) for key, value in values]
    if c['algorithm'] == 'dp_equil':
        tokens += ['cap' + format_number(c['equil']['scale_min']) + '-' + format_number(c['equil']['scale_max'])]
    return '_'.join(tokens)


def format_tmux_session_name(directory):
    return re.sub(r'[^A-Za-z0-9_-]', '_', 'dp_kfac_' + Path(directory).name)


def write_yaml(path, value):
    Path(path).write_text(yaml.safe_dump(value, sort_keys=False), encoding='utf-8')


def write_summary(directory, value):
    path = Path(directory) / 'summary.tmp'
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    path.replace(Path(directory) / 'summary.json')


class MetricsCSVWriter:
    def __init__(self, path):
        self.path = Path(path)
        if not self.path.exists():
            with self.path.open('w', newline='') as stream:
                csv.DictWriter(stream, fieldnames=METRICS_FIELDS).writeheader()
                stream.flush()
                os.fsync(stream.fileno())

    def append(self, row):
        if set(row) != set(METRICS_FIELDS):
            raise ValueError('metrics must contain exactly the epoch schema')
        with self.path.open('a', newline='') as stream:
            csv.DictWriter(stream, fieldnames=METRICS_FIELDS).writerow(row)
            stream.flush()
            os.fsync(stream.fileno())


def prepare_run(c, config_path, now=None):
    root = Path(c['output']['root']).resolve()
    root.mkdir(parents=True, exist_ok=True)
    stamp = now or datetime.now()
    while True:
        directory = root / format_run_name(c, stamp)
        try:
            directory.mkdir()
        except FileExistsError:
            stamp += timedelta(seconds=1)
        else:
            break
    (directory / 'config.yaml').write_bytes(Path(config_path).read_bytes())
    resolved = deepcopy(c)
    steps = math.ceil(60000 / c['data']['batch_size'])
    resolved.update(train_size=60000, test_size=10000, steps_per_epoch=steps,
        total_steps=steps * c['training']['epochs'], sample_rate=c['data']['batch_size']/60000,
        noise_multiplier=None, actual_device=None, physical_gpu_index=None, gpu_name=None,
        run_directory=str(directory), rng_seeds=rng_seeds(c))
    write_yaml(directory / 'resolved_config.yaml', resolved)
    MetricsCSVWriter(directory / 'metrics.csv')
    (directory / 'train.log').touch()
    return directory


def rng_seeds(c):
    return {'model_and_dp_noise': c['seed'], 'train_loader': c['seed'],
            'test_loader': c['seed'],
            'synthetic_epoch_seeds': {epoch: c['seed'] + 10000 + epoch
                for epoch in range(1, c['training']['epochs'] + 1)
                if c['algorithm'] != 'dp_sgd' and (epoch-1) % c['synthetic']['refresh_every_epochs'] == 0},
            'probe_epoch_seeds': {epoch: c['seed'] + 30000 + epoch
                for epoch in range(1, c['training']['epochs'] + 1)
                if c['algorithm'] == 'dp_equil' and (epoch-1) % c['synthetic']['refresh_every_epochs'] == 0}}
