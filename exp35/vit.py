"""Execute Exp30's unchanged training loop with Exp35 dependencies.

A process-local package alias avoids executing exp30/__init__.py, whose cache
setup writes outside Exp35. The actual exp30/run.py is loaded without copying it.
Each worker runs in its own interpreter; no existing source is modified.
"""
import importlib.util
import json
import sys
import shutil
import torch
from types import ModuleType
from exp35 import ROOT, REPO
from exp35 import config as cfg
from exp22 import config as model_cfg
model_cfg.ROOT = ROOT
from exp22.model import initialize
from exp22 import geometry as reference_geometry
from exp35.geometry import SpectralAOperator, diagnostics
from exp35.adapters import bind, prepare, MetricsSink
from torchvision import datasets

package = ModuleType('exp30')
package.__path__ = [str(REPO / 'exp30')]
package.config = cfg
sys.modules['exp30'] = package
sys.modules['exp30.config'] = cfg
spec = importlib.util.spec_from_file_location('exp35._vit_reference', REPO / 'exp30' / 'run.py')
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


def load_data(download=False):
    assert download is False
    transform = reference.data_transform()
    return (datasets.CIFAR10(cfg.DATA_ROOT, train=True, download=False, transform=transform),
            datasets.CIFAR10(cfg.DATA_ROOT, train=False, download=False, transform=transform))


def prepare_checkpoint():
    # Reuse the already downloaded Exp30 checkpoint, keeping all cache writes local.
    # Dereference source links: copytree cannot recreate existing destination
    # symlinks even with dirs_exist_ok=True on the second ViT run.
    checkpoint = 'models--timm--' + cfg.MODEL_NAME
    shutil.copytree(REPO / 'exp30' / '.cache/huggingface/hub' / checkpoint,
                    ROOT / '.cache/huggingface/hub' / checkpoint,
                    symlinks=False, dirs_exist_ok=True)


def run(method):
    prepare_checkpoint()
    directory, configuration = prepare('vit', method)
    variant = method.removeprefix('dp_kfc_a_')
    builder = bind(reference_geometry.build_a_operator.__wrapped__,
                   AOnlyOperator=lambda factors, power, damping: SpectralAOperator(factors, variant))

    @torch.no_grad()
    def build(model, exp22_method, probes, seed, epoch, damping, power):
        if method == 'dp_adamw':
            operator, stats = reference_geometry.build_from_batches(model, 'dp_sgd', probes)
            stats.update(diagnostics({}))
            return operator, stats
        return builder(model, probes, power, damping)

    loop = bind(reference.run, cfg=cfg, load_data=load_data, build_from_batches=build,
                pd=MetricsSink(directory, 'vit', method, lambda: {}))
    loop(method, cfg.DAMPING)
    # Exp30 writes its historical config at completion; retain the richer
    # Exp35 configuration that was already saved before initialization/training.
    (directory / 'config.json').write_text(json.dumps(configuration, indent=2, default=str)+'\n')
