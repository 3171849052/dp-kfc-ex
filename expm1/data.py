"""Read-only datasets and provenance-tagged geometry batches for ExpM1."""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from expm1 import config as cfg
from expm1.geometry import FactorBatch


# Geometry owns this type.  Keeping the alias makes the provenance boundary
# explicit without creating a second, incompatible batch representation.
CalibrationBatch = FactorBatch


def mnist_transform() -> transforms.Compose:
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )


def vit_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(
                (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


def mnist_datasets() -> tuple[Dataset, Dataset]:
    transform = mnist_transform()
    train = datasets.MNIST(cfg.DATA_ROOT, train=True, download=False, transform=transform)
    test = datasets.MNIST(cfg.DATA_ROOT, train=False, download=False, transform=transform)
    assert (len(train), len(test)) == (cfg.MNIST.train_samples, cfg.MNIST.test_samples)
    return train, test


def cifar10_datasets() -> tuple[Dataset, Dataset]:
    transform = vit_transform()
    train = datasets.CIFAR10(cfg.DATA_ROOT, train=True, download=False, transform=transform)
    test = datasets.CIFAR10(cfg.DATA_ROOT, train=False, download=False, transform=transform)
    assert (len(train), len(test)) == (cfg.VIT.train_samples, cfg.VIT.test_samples)
    return train, test


def fashion_mnist_public_dataset() -> Dataset:
    dataset = datasets.FashionMNIST(
        cfg.DATA_ROOT, train=True, download=False, transform=mnist_transform()
    )
    assert len(dataset) == 60_000 and len(dataset.classes) == cfg.NUM_CLASSES
    return dataset


def require_stl10() -> Path:
    location = cfg.DATA_ROOT / "stl10_binary"
    if not location.is_dir():
        raise FileNotFoundError(
            f"required matched-public STL10 is absent at {location}; "
            "the fixed protocol forbids downloading or substituting a dataset"
        )
    return location


def stl10_public_dataset() -> Dataset:
    require_stl10()
    dataset = datasets.STL10(
        cfg.DATA_ROOT, split="train", download=False, transform=vit_transform()
    )
    assert len(dataset) == 5_000 and len(dataset.classes) == cfg.NUM_CLASSES
    return dataset


def private_datasets(task: str) -> tuple[Dataset, Dataset]:
    assert task in cfg.TASKS
    return mnist_datasets() if task == "mnist" else cifar10_datasets()


def public_dataset(task: str) -> Dataset:
    assert task in cfg.TASKS
    return (
        fashion_mnist_public_dataset()
        if task == "mnist"
        else stl10_public_dataset()
    )


def private_loader(dataset: Dataset, task: str, seed: int) -> DataLoader:
    protocol = cfg.task_config(task)
    assert seed in cfg.SEEDS
    return DataLoader(
        dataset,
        batch_size=protocol.logical_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )


def test_loader(dataset: Dataset, task: str) -> DataLoader:
    protocol = cfg.task_config(task)
    return DataLoader(
        dataset,
        batch_size=protocol.physical_batch_size,
        shuffle=False,
        num_workers=0,
        generator=torch.Generator().manual_seed(0),
    )


def _fixed_dataset_batches(
    dataset: Dataset,
    *,
    selection_seed: int,
    provenance: str,
    device: torch.device | str,
    batches: int,
    batch_size: int,
) -> Iterator[FactorBatch]:
    count = batches * batch_size
    assert len(dataset) >= count
    indices = torch.randperm(
        len(dataset), generator=torch.Generator().manual_seed(selection_seed)
    )[:count]
    subset = Subset(dataset, indices.tolist())
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(selection_seed),
    )
    for x, y in loader:
        yield FactorBatch(x.to(device), y.to(device), provenance)


def public_calibration(
    task: str,
    seed: int,
    epoch: int,
    device: torch.device | str,
) -> Iterator[FactorBatch]:
    assert seed in cfg.SEEDS and 1 <= epoch <= cfg.EPOCHS
    selection_seed = seed + cfg.PUBLIC_SEED_OFFSET + epoch
    return _fixed_dataset_batches(
        public_dataset(task),
        selection_seed=selection_seed,
        provenance="public",
        device=device,
        batches=cfg.AUXILIARY_BATCHES,
        batch_size=cfg.AUXILIARY_BATCH_SIZE,
    )


def _pink_input_shape(task: str) -> tuple[int, int, int]:
    assert task in cfg.TASKS
    return (1, 28, 28) if task == "mnist" else (3, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)


def pink_calibration(
    task: str,
    seed: int,
    epoch: int,
    device: torch.device | str,
) -> Iterator[FactorBatch]:
    """Yield a deterministic epoch-local pink stream without changing caller RNGs."""
    from dp_kfac.optimizer import generate_pink_noise

    assert seed in cfg.SEEDS and 1 <= epoch <= cfg.EPOCHS
    device = torch.device(device)
    cuda_devices = []
    if device.type == "cuda":
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=cuda_devices):
        image_seed = seed + cfg.PINK_SEED_OFFSET + epoch
        torch.random.default_generator.manual_seed(image_seed)
        if cuda_devices:
            torch.cuda.default_generators[cuda_devices[0]].manual_seed(image_seed)
        label_generator = torch.Generator(device=device).manual_seed(
            seed + cfg.PINK_LABEL_SEED_OFFSET + epoch
        )
        for _ in range(cfg.AUXILIARY_BATCHES):
            x = generate_pink_noise(
                cfg.AUXILIARY_BATCH_SIZE, _pink_input_shape(task), device
            )
            y = torch.randint(
                cfg.NUM_CLASSES,
                (cfg.AUXILIARY_BATCH_SIZE,),
                device=device,
                generator=label_generator,
            )
            yield FactorBatch(x, y, "pink")


def oracle_calibration(
    task: str,
    device: torch.device | str,
) -> Iterator[FactorBatch]:
    """Return the one fixed private oracle sample, never a current training batch."""
    train, _ = private_datasets(task)
    return _fixed_dataset_batches(
        train,
        selection_seed=cfg.ORACLE_SEED,
        provenance="oracle",
        device=device,
        batches=cfg.ORACLE_BATCHES,
        batch_size=cfg.ORACLE_BATCH_SIZE,
    )


def geometry_batches(batches: Iterator[FactorBatch]) -> Iterator[FactorBatch]:
    """Enforce that training geometry cannot consume private/oracle diagnostics."""
    for batch in batches:
        assert isinstance(batch, FactorBatch)
        assert batch.provenance in {"public", "pink"}
        yield batch


def oracle_batches(batches: Iterator[FactorBatch]) -> Iterator[FactorBatch]:
    for batch in batches:
        assert isinstance(batch, FactorBatch)
        assert batch.provenance == "oracle"
        yield batch


# Short aliases used by worker/check code.
load_private = private_datasets
public_batches = public_calibration
pink_batches = pink_calibration
fixed_oracle_batches = oracle_calibration
