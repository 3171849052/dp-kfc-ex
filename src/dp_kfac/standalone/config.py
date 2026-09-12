"""Strict standalone configuration; paths are relative to the working directory."""
from copy import deepcopy
from pathlib import Path
import math
import yaml

DEFAULTS = {
    'seed': 42,
    'model': {'name': 'simple_cnn'},
    'data': {'dataset': 'mnist', 'root': 'exp1/data', 'batch_size': 256,
             'eval_batch_size': 256, 'num_workers': 0},
    'training': {'epochs': 5, 'optimizer': 'sgd', 'learning_rate': 0.1,
                 'momentum': 0.9, 'weight_decay': 0.0,
                 'betas': [0.9, 0.999], 'eps': 1e-8},
    'privacy': {'epsilon': 1.0, 'delta': 1e-5, 'max_grad_norm': 1.0, 'accountant': 'rdp'},
    'synthetic': {'samples': 2560, 'batch_size': 256, 'distribution': 'pink_noise',
                  'refresh_every_epochs': 1},
    'equil': {'probes': 8, 'tau': 0.01},
    'kfac': {'damping': 1e-3},
    'runtime': {'device': 'cuda', 'gpu': 0, 'threads': 4, 'deterministic': True},
    'output': {'root': 'outputs'},
}


def load_config(path):
    raw = yaml.safe_load(Path(path).read_text())
    c = deepcopy(DEFAULTS)
    if not isinstance(raw, dict):
        raise ValueError('config must be a mapping')
    if raw.get('algorithm') not in ('dp_sgd', 'dp_kfc', 'dp_equil'):
        raise ValueError('algorithm must be dp_sgd, dp_kfc or dp_equil')
    for key, value in raw.items():
        if key == 'algorithm':
            c[key] = value
        elif key not in c:
            raise ValueError(f'unknown config field: {key}')
        elif isinstance(c[key], dict):
            if not isinstance(value, dict) or value.keys() - c[key].keys():
                raise ValueError(f'invalid fields in {key}')
            c[key].update(value)
        else:
            c[key] = value
    for section, name, expected in [('model', 'name', 'simple_cnn'),
            ('data', 'dataset', 'mnist'),
            ('privacy', 'accountant', 'rdp'), ('synthetic', 'distribution', 'pink_noise')]:
        if c[section][name] != expected:
            raise ValueError(f'{section}.{name} must be {expected}')
    if c['training']['optimizer'] not in ('sgd', 'adamw'):
        raise ValueError('training.optimizer must be sgd or adamw')
    betas = c['training']['betas']
    if (not isinstance(betas, list) or len(betas) != 2
            or any(type(v) not in (int, float) or not math.isfinite(v)
                   or not 0 <= v < 1 for v in betas)):
        raise ValueError('training.betas must contain two finite numbers in [0, 1)')
    for section, fields in {'data': ['batch_size', 'eval_batch_size'],
            'training': ['epochs'], 'synthetic': ['samples', 'batch_size', 'refresh_every_epochs'],
            'equil': ['probes'], 'runtime': ['threads']}.items():
        for name in fields:
            if type(c[section][name]) is not int or c[section][name] <= 0:
                raise ValueError(f'{section}.{name} must be a positive integer')
    for section, fields in {'training': ['learning_rate', 'eps'], 'privacy': ['epsilon', 'delta', 'max_grad_norm'],
            'equil': ['tau'], 'kfac': ['damping']}.items():
        for name in fields:
            v = c[section][name]
            if type(v) not in (float, int) or not math.isfinite(v) or v <= 0:
                raise ValueError(f'{section}.{name} must be finite and positive')
    for section, name in [('runtime', 'gpu'), ('data', 'num_workers')]:
        if type(c[section][name]) is not int or c[section][name] < 0:
            raise ValueError(f'{section}.{name} must be a nonnegative integer')
    if type(c['seed']) is not int or c['seed'] < 0:
        raise ValueError('seed must be a nonnegative integer')
    if c['runtime']['device'] not in ('cpu', 'cuda'):
        raise ValueError('runtime.device must be cpu or cuda')
    if type(c['runtime']['deterministic']) is not bool:
        raise ValueError('runtime.deterministic must be boolean')
    for name in ('momentum', 'weight_decay'):
        v = c['training'][name]
        if type(v) not in (float, int) or not math.isfinite(v) or v < 0:
            raise ValueError(f'training.{name} must be finite and nonnegative')
    if c['privacy']['delta'] >= 1:
        raise ValueError('delta must be less than 1')
    if c['synthetic']['samples'] % c['synthetic']['batch_size']:
        raise ValueError('synthetic.samples must be divisible by synthetic.batch_size')
    return c
