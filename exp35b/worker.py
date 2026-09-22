"""One fresh run; shell owns the fixed physical GPU assignment."""
import argparse
import os
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exp35b.runtime import configure
from exp35b.geometry import ALPHAS, GPU_BY_VARIANT, install


def setup(variant):
    cfg = configure(Path(__file__).resolve().parent)
    cfg.GPU = GPU_BY_VARIANT[variant]
    cfg.VARIANTS = tuple(ALPHAS)
    cfg.METHODS = tuple('dp_kfc_a_'+v for v in ALPHAS)
    cfg.EXP22_METHOD = {m: 'dp_kfc_a_bk' for m in cfg.METHODS}
    cfg.grid = lambda dataset: cfg.METHODS
    install()
    from exp35 import adapters
    prepare = adapters.prepare

    def local_prepare(dataset, method):
        import json
        directory, configuration = prepare(dataset, method)
        configuration.update(alpha=ALPHAS[variant], physical_gpu=cfg.GPU)
        (directory/'config.json').write_text(json.dumps(configuration, indent=2)+'\n')
        return directory, configuration

    adapters.prepare = local_prepare
    # The two GPUs may initialize ViT concurrently. Give each worker its own
    # pretrained/cache destination, while keeping run outputs at cfg.ROOT.
    from exp35 import configure_environment
    cfg.CACHE_ROOT = cfg.ROOT / 'workers' / variant
    adapters.bind(configure_environment, ROOT=cfg.CACHE_ROOT)()
    from exp35 import vit
    # Exp22.model assigns HF_HOME/TORCH_HOME at import; restore the local paths.
    adapters.bind(configure_environment, ROOT=cfg.CACHE_ROOT)()
    vit.model_cfg.ROOT = cfg.CACHE_ROOT
    vit.prepare_checkpoint = adapters.bind(vit.prepare_checkpoint, ROOT=cfg.CACHE_ROOT)
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=('mnist', 'vit'), required=True)
    parser.add_argument('--variant', choices=tuple(ALPHAS), required=True)
    args = parser.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == str(GPU_BY_VARIANT[args.variant])
    setup(args.variant)
    import torch
    torch.set_num_threads(4)
    from exp35 import mnist, vit
    {'mnist': mnist.run, 'vit': vit.run}[args.dataset]('dp_kfc_a_'+args.variant)


if __name__ == '__main__':
    main()
