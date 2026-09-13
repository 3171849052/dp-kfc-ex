from pathlib import Path
import torch
from torchvision import datasets, transforms
from .curvature import estimate


def private_cache(samples, batch_size, device, seed):
    dataset = datasets.MNIST(str(Path(__file__).parents[1]/'exp9/data'), train=True,
                            download=False, transform=transforms.Compose([
                                transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))[:samples]
    x = torch.stack([dataset[int(i)][0] for i in indices]).to(device)
    return list(x.split(batch_size))


def build(model, cache):
    return estimate(model, cache, 'KFLR')
