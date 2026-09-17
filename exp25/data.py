from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from dp_kfac.data import IMAGENET_MEAN, IMAGENET_STD
from .config import ROOT, BATCH_SIZE

OFFSETS = {'private': 100000, 'dp_noise': 200000, 'pink': 300000,
           'pink_labels': 400000, 'oracle': 500000, 'public': 600000}

def rng(seed, stream, epoch=0, device='cpu'):
    return torch.Generator(device=device).manual_seed(seed + OFFSETS[stream] + epoch * 1000000)


def load_data(root=None):
    transform = transforms.Compose([transforms.Resize((240, 240)), transforms.ToTensor(),
                                     transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    # The original entry point keeps its data location; standalone supplies its YAML root.
    existing = Path(root) if root is not None else ROOT.parent / 'data'
    root_override = root is not None
    def dataset(cls, train):
        root = existing if root_override or (existing / cls.base_folder).exists() else ROOT / 'data'
        return cls(str(root), train=train, download=root_override or root != existing, transform=transform)
    return dataset(datasets.CIFAR100, True), dataset(datasets.CIFAR100, False), dataset(datasets.CIFAR10, True)


def private_loader(dataset, seed, epoch, smoke=False, batch_size=BATCH_SIZE):
    indices = torch.randperm(len(dataset), generator=rng(seed, 'private', epoch))
    if smoke:
        indices = indices[:batch_size]
    return DataLoader(Subset(dataset, indices.tolist()), batch_size=batch_size,
                      drop_last=True, num_workers=0)


def auxiliary(source, private, public, seed, epoch, device, batch_size=BATCH_SIZE):
    if source == 'pink':
        # Same frequency law and input-space standardization as the original experiment.
        white = torch.randn(batch_size, 3, 240, 240, dtype=torch.cfloat, device=device,
                            generator=rng(seed, 'pink', epoch, device))
        f = torch.fft.fftfreq(240, device=device)
        fx, fy = torch.meshgrid(f, f, indexing='ij')
        radius = (fx.square() + fy.square()).sqrt()
        radius[0, 0] = 1
        scale = radius.reciprocal()
        scale[0, 0] = 0
        x = torch.fft.ifft2(white * scale).real
        x = x / (x.flatten(1).std(1)[:, None, None, None] + 1e-8) * .5
        y = torch.randint(100, (batch_size,), generator=rng(seed, 'pink_labels', epoch, device), device=device)
        return x, y
    dataset = public if source == 'public' else private
    indices = torch.randperm(len(dataset), generator=rng(seed, source, epoch))[:batch_size]
    x, y = next(iter(DataLoader(Subset(dataset, indices.tolist()), batch_size=batch_size)))
    return x.to(device), y.to(device)
