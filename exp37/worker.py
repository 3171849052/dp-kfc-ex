"""One fresh run, with the unchanged Exp30 loop and Exp35 metrics boundary."""
import argparse
import json
import os
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exp37.config import ROOT, cfg, RUNS, GPU_BY_RUN
from exp35 import configure_environment
from exp35.adapters import bind, prepare, MetricsSink
from exp37.geometry import build_from_batches


class FamilyMetricsSink(MetricsSink):
    def DataFrame(self, rows):
        row = rows[-1]
        for family in ('attention_qkv', 'attention_out', 'patch_head'):
            row['family_norm_' + family] = row['group_norm_' + family]
        for family in ('mlp_fc1', 'mlp_fc2'):
            row['family_norm_' + family] = sum(
                value ** 2 for key, value in row.items()
                if key.startswith('layer_norm_') and key.endswith('_' + family)) ** .5
        return super().DataFrame(rows)


def setup(method):
    family, power = RUNS[method]
    cfg.GPU = GPU_BY_RUN[method]
    cfg.A_POWER = power
    cache_root = ROOT / 'workers' / f'gpu{cfg.GPU}'
    bind(configure_environment, ROOT=cache_root)()
    from exp35 import vit
    bind(configure_environment, ROOT=cache_root)()
    vit.model_cfg.ROOT = cache_root
    vit.prepare_checkpoint = bind(vit.prepare_checkpoint, ROOT=cache_root)
    return vit


def run(method):
    vit = setup(method)
    directory, configuration = prepare('vit', method)
    vit.prepare_checkpoint()
    family, power = RUNS[method]
    configuration.update(max_grad_norm=cfg.MAX_GRAD_NORM, p=power, a_power=power,
        g_power=power if family == "trace_ag" else None, family=family,
        scalar_rms_normalization=family != "baseline",
        physical_gpu=cfg.GPU, spectral_normalization=False,
        alpha_mixing=False, gain_cap=None, a_scale_matching=False)
    configuration['rng']['dp_noise'] = 40042
    configuration['rng']['synthetic_y'] = '42+20000+epoch' if family == 'trace_ag' else 'unused; no labels generated'
    path = directory / 'config.json'
    path.write_text(json.dumps(configuration, indent=2)+'\n')
    loop = bind(vit.reference.run, cfg=cfg, load_data=vit.load_data,
                build_from_batches=build_from_batches,
                pd=FamilyMetricsSink(directory, 'vit', method,
                    lambda: dict(p=power, family=family, max_grad_norm=2)))
    loop(method, cfg.DAMPING)
    path.write_text(json.dumps(configuration, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=tuple(RUNS), required=True)
    args = parser.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == str(GPU_BY_RUN[args.method])
    import torch
    torch.set_num_threads(4)
    run(args.method)


if __name__ == '__main__':
    main()
