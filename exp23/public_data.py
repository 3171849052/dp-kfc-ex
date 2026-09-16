"""Paper Fig. 2 / Appendix G sources; input adaptation does not change domains."""
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from exp20.methods import synthetic_cache
from exp23 import config as cfg


def transform(cifar=False):
    # Same adapter as scripts/paper/ablation_fim_spectrum.py.
    prefix = [transforms.Resize((28, 28)), transforms.Grayscale(1)] if cifar else []
    return transforms.Compose(prefix + [transforms.ToTensor(), transforms.Normalize((.1307,), (.3081,))])


def public_dataset(source):
    return {'fashion': lambda: datasets.FashionMNIST(cfg.FASHION_ROOT, train=True, download=False, transform=transform()),
            'cifar10': lambda: datasets.CIFAR10(cfg.CIFAR_ROOT, train=True, download=False, transform=transform(True))}[source]()


def private_data():
    return tuple(datasets.MNIST(cfg.MNIST_ROOT, train=train, download=False, transform=transform()) for train in (True, False))


def fixed_batches(data, seed, device, batches):
    indices = torch.randperm(len(data), generator=torch.Generator().manual_seed(seed))[:batches*256]
    assert len(indices) == batches*256
    loader = DataLoader(Subset(data, indices.tolist()), batch_size=256,
                        generator=torch.Generator().manual_seed(seed))
    return [(x.to(device), y.to(device)) for x, y in loader]


def calibration(source, seed, epoch, device):
    if source == 'pink':
        xs = synthetic_cache(seed, epoch, device, cfg.CALIBRATION_BATCHES, cfg.CALIBRATION_BATCH_SIZE)
        generator = torch.Generator(device=device).manual_seed(seed+20000+epoch)
        return [(x, torch.randint(10, (len(x),), device=device, generator=generator)) for x in xs]
    return fixed_batches(public_dataset(source), seed+10000+epoch, device, cfg.CALIBRATION_BATCHES)
