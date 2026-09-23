"""Fixed ExpM1 protocol and complete formal experiment grid."""
from __future__ import annotations

from dataclasses import dataclass

from expm1 import CACHE_ROOT, REPO_ROOT, ROOT


DATA_ROOT = REPO_ROOT / "data"
RESULTS_ROOT = ROOT / "results"
LOGS_ROOT = ROOT / "logs"
TMP_ROOT = CACHE_ROOT / "tmp"

TASKS = ("mnist", "vit")
METHODS = ("dp_sgd", "dp_kfc", "dp_kfm", "dp_kfm_a")
SOURCES = ("pink", "public")
BETAS = (0.25, 0.5, 0.75, 1.0)
SEEDS = (42, 7, 123)
PHYSICAL_GPUS = (1, 2, 3)

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
        assert self.logical_batch_size % self.physical_batch_size == 0
        return self.logical_batch_size // self.physical_batch_size

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
    public_dataset="STL10",
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
        if self.method == "dp_sgd":
            assert self.source == "none" and self.beta is None
        elif self.method == "dp_kfc":
            assert self.source in SOURCES and self.beta is None
        else:
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

def formal_grid(task: str | None = None) -> tuple[RunSpec, ...]:
    selected_tasks = TASKS if task is None else (task,)
    assert all(name in TASKS for name in selected_tasks)
    runs: list[RunSpec] = []
    for task_name in selected_tasks:
        conditions = [("dp_sgd", "none", None)]
        conditions += [("dp_kfc", source, None) for source in SOURCES]
        conditions += [
            (method, source, beta)
            for method in ("dp_kfm", "dp_kfm_a")
            for source in SOURCES
            for beta in BETAS
        ]
        runs.extend(
            RunSpec(task_name, method, source, beta, seed)
            for method, source, beta in conditions
            for seed in SEEDS
        )
    return tuple(runs)


formal_runs = formal_grid
FORMAL_GRID = formal_grid()
assert len(formal_grid("mnist")) == 57
assert len(formal_grid("vit")) == 57
assert len(FORMAL_GRID) == 114
assert MNIST.accumulation_steps == 1 and VIT.accumulation_steps == 2

# Round-robin within the task-major grid gives every physical GPU exactly
# 19 MNIST and 19 ViT runs.  The mapping is static and serialized by run_all.sh.
GPU_BY_RUN = {
    spec.run_name: PHYSICAL_GPUS[index % len(PHYSICAL_GPUS)]
    for index, spec in enumerate(FORMAL_GRID)
}
GPU_RUNS = {
    gpu: tuple(spec for spec in FORMAL_GRID if GPU_BY_RUN[spec.run_name] == gpu)
    for gpu in PHYSICAL_GPUS
}
assert {gpu: len(runs) for gpu, runs in GPU_RUNS.items()} == {1: 38, 2: 38, 3: 38}
assert all(sum(run.task == task for run in runs) == 19 for runs in GPU_RUNS.values() for task in TASKS)


def task_config(task: str) -> TaskConfig:
    assert task in TASK_CONFIGS
    return TASK_CONFIGS[task]
