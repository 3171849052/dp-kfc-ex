"""Run one method using the unchanged Exp30 loop and Exp35 metrics adapter."""
import argparse
import json
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exp36b.config import cfg, privacy
from exp35.vit import reference, load_data, prepare_checkpoint
from exp35.adapters import bind, prepare, MetricsSink
from exp36b.oracle_geometry import build, calibration
import torch


def run(method):
    prepare_checkpoint()
    prepare_local = bind(prepare, ROOT=cfg.ROOT, cfg=cfg)
    # prepare's dataset nesting is redirected to our requested flat run layout.
    directory, configuration = prepare_local('vit', method)
    target = cfg.RESULTS / 'runs' / method
    target.parent.mkdir(parents=True, exist_ok=True)
    directory.rename(target)
    directory = target
    configuration.update(privacy(method), source='none' if method == 'dp_adamw' else 'private_cifar10',
                         calibration_samples=2560, calibration_batches=10,
                         calibration_batch_size=256, calibration_indices='exp36b/results/oracle_indices.csv',
                         calibration_true_labels=method == 'oracle_dp_kfc',
                         a_scale=1, full_a_power=.5, full_g_power=.5,
                         gpu=0)
    configuration['rng'] = {'initialization':42, 'private_shuffle':42,
                            'calibration_permutation':42, 'dp_noise':40042}
    config_path = directory / 'config.json'
    config_path.write_text(json.dumps(configuration, indent=2)+'\n')
    train, test = load_data(download=False)
    def private_data(download=False):
        assert download is False
        return train, test
    def private_stream(seed, epoch, device):
        return calibration(train, device)
    # Prevent the legacy loop's final config serialization from replacing the
    # richer pre-training configuration, including the oracle privacy boundary.
    class ConfigJSON:
        @staticmethod
        def dumps(value, **kwargs):
            return json.dumps(configuration, **kwargs)
    loop = bind(reference.run, cfg=cfg, load_data=private_data,
                synthetic_stream=private_stream, build_from_batches=build,
                pd=MetricsSink(directory, 'vit', method, lambda: privacy(method)), json=ConfigJSON)
    loop(method, cfg.DAMPING)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=cfg.METHODS, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    run(args.method)
