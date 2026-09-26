"""Fixed ExpM1b protocol and complete formal experiment grid."""
from __future__ import annotations

from dataclasses import dataclass

from expm1b import CACHE_ROOT, REPO_ROOT, ROOT


DATA_ROOT = REPO_ROOT / "data"
RESULTS_ROOT = ROOT / "results"
LOGS_ROOT = ROOT / "logs"
TMP_ROOT = CACHE_ROOT / "tmp"

TASKS = ("mnist", "vit")
METHODS = ("dp_kfm", "dp_kfm_a")
SOURCES = ("pink", "public")
BETAS = (0.25, 0.5)
SEEDS = (42,)
PHYSICAL_GPUS = (0, 1, 2, 3)

EPOCHS = 5
DAMPING = 1e-3
MAX_GRAD_NORM = 1.0
DELTA = 1e-5
AUXILIARY_BATCHES = 10
AUXILIARY_BATCH_SIZE = 256
ORACLE_BATCHES = 10
ORACLE_BATCH_SIZE = 256
ORACLE_SEED = 23_000
PUBLIC_SEED_OFFSET = 10_000
PINK_SEED_OFFSET = 10_000
PINK_LABEL_SEED_OFFSET = 20_000
PUBLIC_LABEL_SEED_OFFSET = 30_000
SAMPLING_SEED_OFFSET = 50_000
NOISE_SEED_OFFSET = 40_000
RESEARCH_ONLY = "research-only: unnoised private diagnostics; not a DP release"

MODEL_NAME = "vit_tiny_patch16_224.augreg_in21k_ft_in1k"
IMAGE_SIZE = 224
NUM_CLASSES = 10
EMBED_DIM = 192


@dataclass(frozen=True)
class TaskConfig:
    name: str
    private_dataset: str
    public_dataset: str
    model: str
    train_samples: int
    test_samples: int
    epochs: int
    logical_batch_size: int
    physical_batch_size: int
    epsilon: float
    delta: float
    max_grad_norm: float
    damping: float
    optimizer: str
    learning_rate: float
    momentum: float
    weight_decay: float
    betas: tuple[float, float] | None
    optimizer_eps: float | None

    @property
    def accumulation_steps(self) -> int:
        return (self.logical_batch_size + self.physical_batch_size - 1) // self.physical_batch_size

    @property
    def logical_batch(self) -> int:
        return self.logical_batch_size

    @property
    def physical_batch(self) -> int:
        return self.physical_batch_size

    @property
    def steps_per_epoch(self) -> int:
        return self.train_samples // self.logical_batch_size

    @property
    def accountant_steps(self) -> int:
        return self.epochs * self.steps_per_epoch

    @property
    def sample_rate(self) -> float:
        return self.logical_batch_size / self.train_samples


MNIST = TaskConfig(
    name="mnist",
    private_dataset="MNIST",
    public_dataset="FashionMNIST",
    model="dp_kfac.models.SimpleCNN",
    train_samples=60_000,
    test_samples=10_000,
    epochs=EPOCHS,
    logical_batch_size=256,
    physical_batch_size=256,
    epsilon=1.0,
    delta=DELTA,
    max_grad_norm=MAX_GRAD_NORM,
    damping=DAMPING,
    optimizer="sgd",
    learning_rate=0.5,
    momentum=0.0,
    weight_decay=0.0,
    betas=None,
    optimizer_eps=None,
)

VIT = TaskConfig(
    name="vit",
    private_dataset="CIFAR10",
    public_dataset="CIFAR100",
    model=MODEL_NAME,
    train_samples=50_000,
    test_samples=10_000,
    epochs=EPOCHS,
    logical_batch_size=256,
    physical_batch_size=128,
    epsilon=3.0,
    delta=DELTA,
    max_grad_norm=MAX_GRAD_NORM,
    damping=DAMPING,
    optimizer="adamw",
    learning_rate=1e-4,
    momentum=0.0,
    weight_decay=0.01,
    betas=(0.9, 0.999),
    optimizer_eps=1e-8,
)

TASK_CONFIGS = {"mnist": MNIST, "vit": VIT}


@dataclass(frozen=True, order=True)
class RunSpec:
    task: str
    method: str
    source: str
    beta: float | None
    seed: int

    def __post_init__(self) -> None:
        assert self.task in TASKS
        assert self.method in METHODS
        assert self.seed in SEEDS
        assert self.source in SOURCES and self.beta in BETAS

    @property
    def run_name(self) -> str:
        fields = [self.task, self.method, self.source]
        if self.beta is not None:
            fields.append(f"beta{self.beta:g}")
        fields.append(f"seed{self.seed}")
        return "_".join(fields)

    @property
    def name(self) -> str:
        return self.run_name

    @property
    def gpu(self) -> int:
        return GPU_BY_RUN[self.run_name]

# Each GPU runs ViT then MNIST for pink, then public.
GPU_RUNS = {
    gpu: tuple(RunSpec(task, method, source, beta, 42)
               for source in SOURCES for task in ("vit", "mnist"))
    for gpu, (method, beta) in enumerate(
        (("dp_kfm", .25), ("dp_kfm", .5),
         ("dp_kfm_a", .25), ("dp_kfm_a", .5)))
}
GPU_BY_RUN = {run.name: gpu for gpu, runs in GPU_RUNS.items() for run in runs}
FORMAL_GRID = tuple(run for runs in GPU_RUNS.values() for run in runs)


def formal_grid(task: str | None = None) -> tuple[RunSpec, ...]:
    assert task is None or task in TASKS
    return tuple(run for run in FORMAL_GRID if task is None or run.task == task)


formal_runs = formal_grid


def task_config(task: str) -> TaskConfig:
    assert task in TASK_CONFIGS
    return TASK_CONFIGS[task]
