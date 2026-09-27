"""Fixed ViT protocol and inverse geometry grid."""
from __future__ import annotations

from dataclasses import dataclass

from expm2 import CACHE_ROOT, REPO_ROOT, ROOT


DATA_ROOT = REPO_ROOT / "data"
RESULTS_ROOT = ROOT / "results"
LOGS_ROOT = ROOT / "logs"
TMP_ROOT = CACHE_ROOT / "tmp"

TASKS = ("vit",)
METHODS = ("inverse_dp_kfm_a",)
SOURCES = ("public", "pink")
BETAS = (0.1, 0.2, 0.3, 0.4)
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

TASK_CONFIGS = {"vit": VIT}


@dataclass(frozen=True)
class RunSpec:
    source: str
    beta: float
    damping: float = DAMPING
    seed: int = 42
    task: str = "vit"
    method: str = "inverse_dp_kfm_a"
    geometry_power: str = "inverse"

    def __post_init__(self):
        assert self.task in TASKS and self.method in METHODS
        assert self.source in SOURCES and self.seed in SEEDS
        assert self.beta in BETAS and self.beta > 0
        assert self.damping == DAMPING
        assert self.geometry_power == "inverse"

    @property
    def run_name(self):
        return f"vit_inverse_dp_kfm_a_{self.source}_beta{self.beta:g}_lambda{self.damping:g}_seed{self.seed}"

    @property
    def name(self):
        return self.run_name

    @property
    def gpu(self):
        return BETAS.index(self.beta)

    @property
    def directory(self):
        return RESULTS_ROOT / "runs" / self.name


GRID = tuple(RunSpec(source, beta) for beta in BETAS for source in SOURCES)
BETA_GRID = GRID


def task_config(task):
    assert task == "vit"
    return VIT

