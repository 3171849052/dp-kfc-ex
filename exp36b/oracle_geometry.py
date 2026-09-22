"""Exp22 builders with fixed private images and true calibration labels."""
import csv
from functools import lru_cache
import torch
from torch.utils.data import DataLoader, Subset
from exp36b.config import cfg
from exp22 import geometry
from exp35.adapters import bind
from exp35.geometry import SpectralAOperator


@lru_cache(maxsize=1)
def indices():
    return torch.randperm(cfg.TRAIN_SAMPLES, generator=torch.Generator().manual_seed(42))[:2560].tolist()


def save_indices():
    cfg.RESULTS.mkdir(parents=True, exist_ok=True)
    with (cfg.RESULTS / 'oracle_indices.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['index'])
        writer.writerows((i,) for i in indices())


def calibration(dataset, device):
    with (cfg.RESULTS / 'oracle_indices.csv').open() as stream:
        fixed_indices = [int(row['index']) for row in csv.DictReader(stream)]
    loader = DataLoader(Subset(dataset, fixed_indices), batch_size=256, shuffle=False,
                        num_workers=0, generator=torch.Generator().manual_seed(42))
    for images, labels in loader:
        yield images.to(device), labels.to(device)


def build(model, method, batches, seed=42, epoch=1, damping=.001, power=.4):
    if method == 'dp_sgd':
        return geometry.build_from_batches(model, method, ())
    if method == 'dp_kfc_a_bk':
        builder = bind(geometry.build_a_operator.__wrapped__,
                       AOnlyOperator=lambda factors, power, damping: SpectralAOperator(factors, 'raw'))
        with torch.no_grad():
            return builder(model, (x for x, y in batches), power, damping)
    # The reference builder requests labels immediately after consuming each
    # logical image batch. Supply that batch's real labels at that exact boundary.
    labels = None
    def images():
        nonlocal labels
        for x, labels in batches:
            yield x
    def true_labels(seed, epoch, device, batch_size, num_classes, generator):
        assert labels.shape == (batch_size,) and labels.device == device
        return labels
    builder = bind(geometry.build_full_operator, synthetic_labels=true_labels)
    return builder(model, images(), seed, epoch, damping)
