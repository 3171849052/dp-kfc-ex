"""Local dependency injection; reference modules/files are never modified."""
import json
from types import FunctionType
import pandas as pd
import torch
from opacus.accountants.utils import get_noise_multiplier
from exp35 import ROOT
from exp35 import config as cfg


def bind(function, **dependencies):
    namespace = dict(function.__globals__)
    namespace.update(dependencies)
    return FunctionType(function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__)


def prepare(dataset, method):
    directory = ROOT / 'results' / dataset / 'runs' / method
    directory.mkdir(parents=True, exist_ok=False)
    mnist = dataset == 'mnist'
    n, epsilon = (60000, 2) if mnist else (50000, 3)
    configuration = {}
    configuration.update(dataset=dataset, method=method, epochs=5, seed=42,
        model='SimpleCNN' if mnist else cfg.MODEL_NAME, pretrained=not mnist,
        full_model_fine_tuning=True, optimizer='Adam' if mnist else 'AdamW',
        learning_rate=.002 if mnist else 1e-4, weight_decay=0 if mnist else .01,
        betas=[.9, .999], optimizer_eps=1e-8, epsilon=epsilon, delta=1e-5,
        max_grad_norm=1, logical_batch_size=256, physical_batch_size=256 if mnist else 128,
        damping=.001, a_power=.4, a_scale_matching=False, source='pink', engine='bk',
        geometry_rebuild_every_epochs=1, synthetic_batches=1 if mnist else 10,
        synthetic_batch_size=256, dataset_root=str(cfg.DATA_ROOT), download=False,
        data_transform={'resize': None if mnist else [224,224],
                        'interpolation': None if mnist else 'bicubic',
                        'mean': [.1307] if mnist else [.5]*3,
                        'std': [.3081] if mnist else [.5]*3, 'augmentation': None},
        accounting='RDP fixed shuffled logical batch, drop_last=True',
        rng={'initialization':42, 'private_shuffle':42, 'synthetic_x':'42+10000+epoch',
             'synthetic_y':'42+20000+epoch (unused for A-only)', 'dp_noise':20042 if mnist else 40042},
        total_steps=5*(n//256), noise_multiplier=get_noise_multiplier(
            target_epsilon=epsilon, target_delta=1e-5, sample_rate=256/n,
            steps=5*(n//256), accountant='rdp'))
    path = directory / 'config.json'
    path.write_text(json.dumps(configuration, indent=2, default=str)+'\n')
    return directory, configuration


class MetricsSink:
    """Reference pandas output boundary: enrich rows and use one fixed filename."""
    def __init__(self, directory, dataset, method, extra):
        self.directory, self.dataset, self.method, self.extra = directory, dataset, method, extra

    def DataFrame(self, rows):
        row = rows[-1]
        row.update(self.extra())
        row['method'] = self.method
        if self.dataset == 'mnist':
            row['test_accuracy'] = row['accuracy']
            row['epsilon'] = row['epsilon_spent']
            for q in ('p50', 'p90', 'p99', 'max'):
                row['transformed_norm_'+q] = row['norm_'+q]
        row['best_accuracy'] = max(r['test_accuracy'] for r in rows)
        for key in ('logical_steps', 'optimizer_steps', 'noise_events'):
            row['epoch_'+key] = row[key]
            row[key] *= row['epoch']
        if self.dataset == 'mnist':
            row['accountant_steps'] *= row['epoch']
        row['noise_steps'] = row['noise_events']
        assert row['logical_steps'] == row['optimizer_steps'] == row['noise_steps'] == row['accountant_steps']
        assert row['parameters_finite'] and row['parameters_updated']
        frame = pd.DataFrame(rows)
        directory = self.directory
        class Output:
            def to_csv(self, path, index=False):
                frame.to_csv(directory / 'metrics.csv', index=index)
        return Output()
