"""Run the unchanged Exp35/Exp30 ViT loop with bounded Oracle A geometry."""
import argparse
import json
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exp36c.config import cfg, ALPHAS, ORACLE_INDICES, privacy
from exp35.vit import reference, load_data, prepare_checkpoint
from exp35.adapters import bind, prepare, MetricsSink
from exp36c.geometry import builder, calibration
import torch


def run(method):
    prepare_checkpoint()
    directory, configuration = prepare('vit', method)
    target = cfg.RESULTS / 'runs' / method
    target.parent.mkdir(parents=True, exist_ok=True)
    directory.rename(target)
    directory = target
    configuration.update(privacy(method), source='private_cifar10', alpha=ALPHAS[method],
        operator='(1-alpha) I + alpha (A+0.001 I)^(-0.4) / spectral_norm',
        gain_range=[1-ALPHAS[method], 1], a_scale=1,
        calibration_samples=2560, calibration_batches=10, calibration_batch_size=256,
        calibration_indices=str(ORACLE_INDICES.relative_to(cfg.ROOT.parent)),
        calibration_true_labels=False, gpu=0,
        rng={'initialization':42, 'private_shuffle':42, 'calibration_permutation':42, 'dp_noise':40042})
    (directory / 'config.json').write_text(json.dumps(configuration, indent=2)+'\n')
    train, test = load_data(download=False)

    def private_data(download=False):
        assert download is False
        return train, test

    def private_stream(seed, epoch, device):
        return calibration(train, device)

    class ConfigJSON:
        @staticmethod
        def dumps(value, **kwargs):
            return json.dumps(configuration, **kwargs)

    loop = bind(reference.run, cfg=cfg, load_data=private_data,
        synthetic_stream=private_stream, build_from_batches=builder(method),
        pd=MetricsSink(directory, 'vit', method, lambda: privacy(method)), json=ConfigJSON)
    loop(method, cfg.DAMPING)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=cfg.METHODS, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    run(args.method)
