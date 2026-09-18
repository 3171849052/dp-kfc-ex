"""Full-parameter DistilBERT SST-2 prompt MLM: Explicit versus tiled BK.

The three conditions are DP-Adam, synthetic Full KFC, and synthetic A-only
KFC. Each epoch builds geometry from 256 independent random-token prompts;
no public or private examples are used for calibration. Embeddings, LayerNorm,
and the tied vocabulary projection use identity geometry. Other Linear layers
use unnormalized, bias-augmented geometry, as in exp_cnn_mnist_a.py.

All 67,349 training examples are visited per epoch (including the short final
logical batch). RDP uses the reference experiment's shuffled-batch sample-rate
convention, conservatively charging every step at 1024 / 67349; this is not a
Poisson-sampling privacy proof. Validation is used only for final evaluation.
BK keeps raw factors, computes tiled norms including the embedding/head cross
term, and preconditions only the clipped aggregate. Explicit materializes all
per-example gradients. Both keep every parameter requires_grad=True.
Main accuracy runs use BK physical batch 128. Controlled implementation profiling
compares BK and Explicit only at physical batch 16; practical runtime at batch 128
is end-to-end efficiency, not an algorithm speedup against Explicit at batch 16.
Cache counters measure retained tensor storage, workspace counters track local
tensors (excluding backend workspaces), and CUDA peaks cover the whole allocator.
"""

import argparse
import math
from contextlib import contextmanager
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from opacus import GradSampleModule
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier
from transformers import AutoTokenizer, DistilBertForMaskedLM

from exp21.handlers import Record, nbytes
from exp22.methods import Clipper, _retained_bytes
from dp_kfac.trainer import set_seed

EPSILONS = [0.5, 3.0, 8.0]
PROFILE_EPSILON = 3.0
SEEDS = [42, 7, 91, 23, 58]
EPOCHS = 3
LR = 5e-5
MAX_GRAD_NORM = 1.0
DELTA = 1e-5
LOGICAL_BATCH_SIZE = 1024
BK_PHYSICAL_BATCH_SIZE = 128
EXPLICIT_PHYSICAL_BATCH_SIZE = 16
PROFILE_PHYSICAL_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16
GEOMETRY_BATCH_SIZE = 256
GEOMETRY_PHYSICAL_BATCH_SIZE = 16
DAMPING = 1e-3
A_POWER = 0.4
GHOST_TILE = 32
MAX_LENGTH = 128
MODEL_NAME = "distilbert-base-uncased"
TRAIN_SIZE = 67349
CONDITIONS = [("base", "none"), ("full", "synthetic"), ("a_only", "synthetic")]
TIMINGS = (
    "differentiation_seconds", "factor_precondition_seconds",
    "sample_gradient_precondition_seconds", "norm_clip_seconds",
    "aggregate_seconds", "aggregate_precondition_seconds", "noise_optimizer_seconds",
)
COUNTERS = (
    "sample_gradient_matrices_preconditioned", "sample_gradient_elements_preconditioned",
    "factor_elements_preconditioned_for_norm", "aggregate_matrices_preconditioned",
    "aggregate_elements_preconditioned",
)
MEMORY = ("grad_sample_peak_bytes", "bk_cache_peak_bytes", "bk_temporary_peak_bytes")


def timestamp(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


@contextmanager
def timed(stats, key, device, profile):
    if not profile:
        yield
        return
    started = timestamp(device)
    try:
        yield
    finally:
        stats[key] += timestamp(device) - started


def reset_peak(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def peaks(device, prefix, profile):
    return {f"{prefix}_peak_{kind}_bytes": (
        getattr(torch.cuda, f"max_memory_{kind}")(device) if device.type == "cuda" else 0
    ) if profile else float("nan") for kind in ("allocated", "reserved")}


def new_stats(profile):
    return {key: 0 if profile else float("nan") for key in (*TIMINGS, *COUNTERS, *MEMORY)}


def peak(stats, key, value, profile):
    if profile:
        stats[key] = max(stats[key], value)


def prompt_parts(tokenizer):
    suffix = tokenizer.encode("It was " + tokenizer.mask_token + ".", add_special_tokens=False)
    verbalizer = [tokenizer.encode(word, add_special_tokens=False) for word in ("terrible", "great")]
    if (any(len(ids) != 1 or ids[0] == tokenizer.unk_token_id for ids in verbalizer)
            or verbalizer[0] == verbalizer[1] or suffix.count(tokenizer.mask_token_id) != 1):
        raise ValueError("The prompt requires one mask and single-token verbalizers")
    return suffix, [ids[0] for ids in verbalizer]


def pack_prompts(sentences, tokenizer):
    """Truncate only the sentence, retaining the complete suffix and its mask."""
    suffix, _ = prompt_parts(tokenizer)
    room = MAX_LENGTH - len(suffix) - 2
    encoded = tokenizer(sentences, add_special_tokens=False, truncation=True, max_length=room)["input_ids"]
    rows = []
    for sentence in encoded:
        ids = [tokenizer.cls_token_id, *sentence, *suffix, tokenizer.sep_token_id]
        mask_position = 1 + len(sentence) + suffix.index(tokenizer.mask_token_id)
        attention = [1] * len(ids) + [0] * (MAX_LENGTH - len(ids))
        ids += [tokenizer.pad_token_id] * (MAX_LENGTH - len(ids))
        # Packed tensor gives Opacus and the BK engine a conventional batch axis.
        rows.append([ids, attention, [mask_position] * MAX_LENGTH])
    return torch.tensor(rows, dtype=torch.long)


def load_data(tokenizer):
    from datasets import load_dataset
    dataset = load_dataset("nyu-mll/glue", "sst2")
    if len(dataset["train"]) != TRAIN_SIZE:
        raise ValueError(f"Expected complete SST-2 train ({TRAIN_SIZE}), got {len(dataset['train'])}")
    result = []
    for split in ("train", "validation"):
        data = dataset[split]
        tensors = [pack_prompts(data[start:start + 2048]["sentence"], tokenizer)
                   for start in range(0, len(data), 2048)]
        result.append(TensorDataset(torch.cat(tensors), torch.tensor(data["label"], dtype=torch.long)))
    return tuple(result)


class PromptMLM(nn.Module):
    def __init__(self, mlm, verbalizer):
        super().__init__()
        self.mlm = mlm
        self.register_buffer("verbalizer", torch.tensor(verbalizer, dtype=torch.long), persistent=False)
        for parameter in self.parameters():
            parameter.requires_grad_(True)
        # HF broadcasts position embeddings from [1,T,D]. Expand their INPUT
        # before both Opacus and BK hooks so gradients retain the sample axis.
        self._batch_size = 0
        self.mlm.distilbert.embeddings.position_embeddings.register_forward_pre_hook(self._positions)

    def _positions(self, module, inputs):
        ids = inputs[0]
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        return (ids.expand(self._batch_size, -1),)

    def forward(self, packed):
        self._batch_size = len(packed)
        hidden = self.mlm.distilbert(input_ids=packed[:, 0], attention_mask=packed[:, 1]).last_hidden_state
        # The token-wise MLM head is evaluated only at the prompt mask. This is
        # exactly the same objective without allocating [batch,128,vocabulary].
        hidden = hidden[torch.arange(len(packed), device=packed.device), packed[:, 2, 0]]
        hidden = self.mlm.vocab_transform(hidden)
        hidden = self.mlm.activation(hidden)
        hidden = self.mlm.vocab_layer_norm(hidden)
        return self.mlm.vocab_projector(hidden).index_select(-1, self.verbalizer)


def make_model(tokenizer, device):
    _, verbalizer = prompt_parts(tokenizer)
    mlm = DistilBertForMaskedLM.from_pretrained(MODEL_NAME, attn_implementation="eager")
    return PromptMLM(mlm, verbalizer).to(device)


def geometry_layers(model):
    """Shared vocabulary weights have one identity map across both owners."""
    owners = {}
    for module in model.modules():
        for parameter in module.parameters(recurse=False):
            owners[id(parameter)] = owners.get(id(parameter), 0) + 1
    return {name: module for name, module in model.named_modules()
            if isinstance(module, nn.Linear)
            and all(owners[id(p)] == 1 for p in module.parameters(recurse=False))}


def synthetic_prompts(tokenizer, seed, epoch):
    """Variable-length uniform non-special payloads with independent random labels."""
    rng = torch.Generator().manual_seed(seed + 10000 + epoch)
    suffix, _ = prompt_parts(tokenizer)
    allowed = torch.tensor(sorted(set(range(len(tokenizer))) - set(tokenizer.all_special_ids)))
    room = MAX_LENGTH - len(suffix) - 2
    lengths = torch.randint(1, room + 1, (GEOMETRY_BATCH_SIZE,), generator=rng)
    rows = []
    for length in lengths.tolist():
        payload = allowed[torch.randint(len(allowed), (length,), generator=rng)].tolist()
        ids = [tokenizer.cls_token_id, *payload, *suffix, tokenizer.sep_token_id]
        attention = [1] * len(ids) + [0] * (MAX_LENGTH - len(ids))
        ids += [tokenizer.pad_token_id] * (MAX_LENGTH - len(ids))
        mask_position = 1 + length + suffix.index(tokenizer.mask_token_id)
        rows.append([ids, attention, [mask_position] * MAX_LENGTH])
    return torch.tensor(rows, dtype=torch.long), torch.randint(2, (GEOMETRY_BATCH_SIZE,), generator=rng)


def matrix_power(covariance, power):
    regularized = covariance + (DAMPING + 1e-5) * torch.eye(
        len(covariance), device=covariance.device, dtype=covariance.dtype)
    values, vectors = torch.linalg.eigh(regularized)
    return (vectors * values.clamp_min(1e-6).pow(-power)) @ vectors.T


def build_geometry(model, layers, geometry, tokenizer, device, seed, epoch):
    if geometry == "base":
        return {}, {}
    x, y = synthetic_prompts(tokenizer, seed, epoch)
    sums_a, sums_g, counts = {}, {}, {}
    handles = []

    def capture(name):
        def forward(module, inputs, output):
            with torch.no_grad():
                a = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
                a = torch.cat((a, torch.ones_like(a[:, :1])), -1)
                covariance = a.T @ a
                if name not in sums_a:
                    sums_a[name] = covariance
                    counts[name] = len(a)
                else:
                    sums_a[name].add_(covariance)
                    counts[name] += len(a)
            if geometry == "full":
                def backward(grad):
                    b = grad.detach().reshape(-1, grad.shape[-1])
                    covariance = b.T @ b
                    if name in sums_g:
                        sums_g[name].add_(covariance)
                    else:
                        sums_g[name] = covariance
                output.register_hook(backward)
        return forward

    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    try:
        handles = [module.register_forward_hook(capture(name)) for name, module in layers.items()]
        # Calibration dropout cannot perturb private dropout/shuffling/noise RNG.
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed + 10000 + epoch)
            for start in range(0, len(x), GEOMETRY_PHYSICAL_BATCH_SIZE):
                part = x[start:start + GEOMETRY_PHYSICAL_BATCH_SIZE].to(device)
                if geometry == "full":
                    loss = F.cross_entropy(model(part), y[start:start + len(part)].to(device), reduction="sum")
                    loss.backward()
                    model.zero_grad(set_to_none=True)
                else:
                    with torch.no_grad():
                        model(part)
    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad(set_to_none=True)
    ua, ug = {}, {}
    for name in layers:
        ua[name] = matrix_power(sums_a.pop(name) / counts[name], .5 if geometry == "full" else A_POWER)
        if geometry == "full":
            ug[name] = matrix_power(sums_g.pop(name) / counts[name], .5)
    return ua, ug


def bk_differentiate(model, x, y, stats, device, profile):
    activations, backprops, anchors, handles = {}, {}, [], []
    layers = {name: module for name, module in model.named_modules()
              if isinstance(module, (nn.Linear, nn.Embedding, nn.LayerNorm))}
    parameters = {id(p) for p in model.parameters()}
    assert all(p.requires_grad and p.grad is None for p in model.parameters())
    assert parameters == {id(p) for module in layers.values() for p in module.parameters(recurse=False)}

    def capture(name):
        def forward(module, inputs, output):
            if name in activations:
                raise RuntimeError(f"Unexpected reused module: {name}")
            activations[name] = inputs[0].detach()
            output.register_hook(lambda grad: backprops.__setitem__(name, grad.detach()))
            if isinstance(module, nn.Embedding):
                anchors.append(output)
        return forward

    try:
        handles = [module.register_forward_hook(capture(name)) for name, module in layers.items()]
        losses = F.cross_entropy(model(x), y, reduction="none")
        # Differentiate to the two embedding outputs, without computing parameter
        # .grad tensors or changing requires_grad. All intermediate hooks fire.
        torch.autograd.grad(losses.sum(), anchors)
    finally:
        for handle in handles:
            handle.remove()
    assert set(activations) == set(backprops) == set(layers)
    peak(stats, "bk_cache_peak_bytes", _retained_bytes(list(activations.values()) + list(backprops.values())), profile)
    records = []
    for name, module in layers.items():
        records.append(Record(name, module, activations.pop(name), backprops.pop(name), None, parameters))
        if profile:
            tensors = list(activations.values()) + list(backprops.values())
            tensors += [t for r in records for t in (r.x, r.z, r.b) if t is not None]
            peak(stats, "bk_cache_peak_bytes", _retained_bytes(tensors), profile)
    return losses.detach().sum(), records


@torch.no_grad()
def bk_clip(model, records, ua, ug, stats, device, profile):
    sq = records[0].b.new_zeros(len(records[0].b))
    embeddings = [r for r in records if r.kind == "embedding"]
    head = next(r for r in records if r.module is model.mlm.vocab_projector)
    word = next(r for r in embeddings if r.params["weight"] is head.params["weight"])
    for record in records:
        with timed(stats, "factor_precondition_seconds", device, profile and record.name in ua):
            z, b = record.z, record.b
            transformed_bytes = 0
            if record.name in ua:
                z = z @ ua[record.name]
                transformed_bytes += nbytes(z)
                if profile:
                    stats["factor_elements_preconditioned_for_norm"] += z.numel()
                if record.name in ug:
                    b = b @ ug[record.name].T
                    transformed_bytes += nbytes(b)
                    if profile:
                        stats["factor_elements_preconditioned_for_norm"] += b.numel()
        with timed(stats, "norm_clip_seconds", device, profile):
            if record.kind == "linear":
                part, workspace = Clipper._linear_norm_squared(z, b, tile=GHOST_TILE)
                sq.add_(part)
                peak(stats, "bk_temporary_peak_bytes", transformed_bytes + workspace + nbytes(part) + nbytes(sq), profile)
            elif record.kind == "embedding":
                samples, ids, values = record.embedding_rows()
                sq.index_add_(0, samples, values.square().sum(-1))
                peak(stats, "bk_temporary_peak_bytes", 2 * nbytes(values) + nbytes(samples) + nbytes(ids), profile)
                if record is word:
                    # Shared parameter: ||embedding + head||² includes 2<e,h>.
                    # Only the sparse token rows are needed, never [B,V,D].
                    head_z = head.z[..., :-1]
                    for start in range(0, len(ids), GHOST_TILE):
                        ss, vv = samples[start:start + GHOST_TILE], ids[start:start + GHOST_TILE]
                        rows = (head.b[ss, :, vv].unsqueeze(-1) * head_z[ss]).sum(1)
                        sq.index_add_(0, ss, 2 * (values[start:start + GHOST_TILE] * rows).sum(-1))
                        peak(stats, "bk_temporary_peak_bytes", 2 * nbytes(values) + 3 * nbytes(rows), profile)
                del samples, ids, values
            else:
                gradients = record.fast()
                sq.add_(sum(g.flatten(1).square().sum(1) for g in gradients.values()))
                peak(stats, "bk_temporary_peak_bytes", 2 * sum(nbytes(g) for g in gradients.values()), profile)
                del gradients
        del z, b
    with timed(stats, "norm_clip_seconds", device, profile):
        norms = sq.clamp_min(0).sqrt()
        factors = (MAX_GRAD_NORM / (norms + 1e-6)).clamp(max=1)
    for record in records:
        with timed(stats, "aggregate_seconds", device, profile):
            if record.kind == "linear":
                weighted = record.b * factors[:, None, None]
                h = weighted.reshape(-1, weighted.shape[-1]).T @ record.z.reshape(-1, record.z.shape[-1])
                peak(stats, "bk_temporary_peak_bytes", nbytes(weighted) + nbytes(h), profile)
                del weighted
            else:
                gradients = record.aggregate(factors)
                peak(stats, "bk_temporary_peak_bytes", sum(nbytes(g) for g in gradients.values()), profile)
        if record.kind == "linear":
            if record.name in ua:
                with timed(stats, "aggregate_precondition_seconds", device, profile):
                    if record.name in ug:
                        h = ug[record.name] @ h
                    h = h @ ua[record.name]
                    if profile:
                        stats["aggregate_matrices_preconditioned"] += 1
                        stats["aggregate_elements_preconditioned"] += h.numel()
                    peak(stats, "bk_temporary_peak_bytes", 2 * nbytes(h), profile)
            gradients = record.split(h)
        with timed(stats, "aggregate_seconds", device, profile):
            for parameter, grad in gradients.items():
                if parameter.grad is None:
                    parameter.grad = grad
                else:
                    parameter.grad.add_(grad)
        del gradients
    records.clear()
    assert all(p.grad is not None and p.requires_grad for p in model.parameters())
    return norms, factors


@torch.no_grad()
def explicit_clip(model, layers, ua, ug, stats, device, batch_size, profile):
    def sample_bytes():
        return _retained_bytes([p.grad_sample for p in model.parameters()])
    if profile:
        peak(stats, "grad_sample_peak_bytes", sample_bytes(), profile)
    with timed(stats, "sample_gradient_precondition_seconds", device, profile and bool(ua)):
        for name, operator in ua.items():
            module = layers[name]
            g = torch.cat((module.weight.grad_sample, module.bias.grad_sample.unsqueeze(-1)), -1)
            if name in ug:
                g = ug[name] @ g
            g = g @ operator
            module.weight.grad_sample, module.bias.grad_sample = g[..., :-1], g[..., -1]
            if profile:
                stats["sample_gradient_matrices_preconditioned"] += batch_size
                stats["sample_gradient_elements_preconditioned"] += g.numel()
                peak(stats, "grad_sample_peak_bytes", sample_bytes(), profile)
    with timed(stats, "norm_clip_seconds", device, profile):
        norms = sum(torch.linalg.vector_norm(p.grad_sample, dim=tuple(range(1, p.grad_sample.ndim))).square()
                    for p in model.parameters()).clamp_min(0).sqrt()
        factors = (MAX_GRAD_NORM / (norms + 1e-6)).clamp(max=1)
    with timed(stats, "aggregate_seconds", device, profile):
        for parameter in model.parameters():
            parameter.grad = torch.einsum("b,b...->...", factors, parameter.grad_sample)
            parameter.grad_sample = None
    return norms, factors


@torch.no_grad()
def evaluate(model, validation, device):
    model.eval()
    correct = total = 0
    loss = 0.0
    for x, y in DataLoader(validation, batch_size=EVAL_BATCH_SIZE):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        correct += (logits.argmax(-1) == y).sum().item()
        loss += F.cross_entropy(logits, y, reduction="sum").item()
        total += len(y)
    return correct / total, loss / total


def run_one(train, validation, tokenizer, geometry, source, engine, epsilon, seed, epochs, device, output_dir, physical_batch_size, profile, profile_mode, lr=LR):
    set_seed(seed)
    raw = make_model(tokenizer, device)
    layers = geometry_layers(raw)
    model = GradSampleModule(raw, loss_reduction="sum") if engine == "explicit" else raw
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(train, batch_size=LOGICAL_BATCH_SIZE, shuffle=True, drop_last=False,
                        generator=torch.Generator().manual_seed(seed))
    sample_rate = min(LOGICAL_BATCH_SIZE / len(train), 1.0)
    sigma = get_noise_multiplier(target_epsilon=epsilon, target_delta=DELTA, sample_rate=sample_rate,
                                 steps=epochs * len(loader), accountant="rdp")
    accountant = RDPAccountant()
    noise_rng = torch.Generator(device=device).manual_seed(seed + 20000)
    rows = []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ua, ug = {}, {}
    try:
        for epoch in range(1, epochs + 1):
            model.train()
            model.zero_grad(set_to_none=True)
            ua.clear()
            ug.clear()
            stats = new_stats(profile)
            if profile:
                reset_peak(device)
            started = timestamp(device) if profile else float("nan")
            if engine == "explicit":
                model.disable_hooks()
            try:
                ua, ug = build_geometry(raw, layers, geometry, tokenizer, device, seed, epoch)
            finally:
                if engine == "explicit":
                    model.enable_hooks()
            stats["geometry_build_seconds"] = timestamp(device) - started if profile else float("nan")
            stats.update(peaks(device, "geometry", profile))
            if profile:
                reset_peak(device)
            started = timestamp(device) if profile else float("nan")
            norms_list, factors_list = [], []
            total_loss = 0.0
            examples = steps = physical_steps = optimizer_steps = noise_events = accountant_steps = 0
            for x, y in loader:
                # One persistent aggregate across physical batches; no intermediate
                # Gaussian noise, optimizer update, or accountant event.
                aggregate = {p: torch.zeros_like(p) for p in model.parameters()}
                for start in range(0, len(x), physical_batch_size):
                    part_x = x[start:start + physical_batch_size].to(device)
                    part_y = y[start:start + physical_batch_size].to(device)
                    model.zero_grad(set_to_none=True)
                    with timed(stats, "differentiation_seconds", device, profile):
                        if engine == "explicit":
                            loss = F.cross_entropy(model(part_x), part_y, reduction="none").sum()
                            loss.backward()
                            loss = loss.detach()
                        else:
                            loss, records = bk_differentiate(raw, part_x, part_y, stats, device, profile)
                    if engine == "explicit":
                        norms, factors = explicit_clip(model, layers, ua, ug, stats, device, len(part_x), profile)
                    else:
                        norms, factors = bk_clip(raw, records, ua, ug, stats, device, profile)
                    with timed(stats, "aggregate_seconds", device, profile), torch.no_grad():
                        for parameter, grad in aggregate.items():
                            grad.add_(parameter.grad)
                    model.zero_grad(set_to_none=True)
                    total_loss += loss.item()
                    examples += len(part_x)
                    physical_steps += 1
                    if profile:
                        norms_list.append(norms.cpu())
                        factors_list.append(factors.cpu())
                with timed(stats, "noise_optimizer_seconds", device, profile), torch.no_grad():
                    for parameter, grad in aggregate.items():
                        noise = torch.randn(parameter.shape, device=device, dtype=parameter.dtype, generator=noise_rng)
                        parameter.grad = grad.add_(noise, alpha=sigma * MAX_GRAD_NORM).div_(len(x))
                    noise_events += 1
                    optimizer.step()
                    optimizer_steps += 1
                    accountant.step(noise_multiplier=sigma, sample_rate=sample_rate)
                    accountant_steps += 1
                model.zero_grad(set_to_none=True)
                del aggregate, grad, noise
                steps += 1
            assert examples == len(train)
            assert steps == optimizer_steps == noise_events == accountant_steps == len(loader)
            stats["private_train_seconds"] = timestamp(device) - started if profile else float("nan")
            stats.update(peaks(device, "private", profile))
            stats["algorithm_seconds"] = stats["geometry_build_seconds"] + stats["private_train_seconds"]
            accuracy = test_loss = float("nan")
            started = timestamp(device) if profile else float("nan")
            if epoch == epochs:
                accuracy, test_loss = evaluate(raw, validation, device)
            stats["evaluation_seconds"] = timestamp(device) - started if profile else float("nan")
            if profile:
                norms, factors = torch.cat(norms_list), torch.cat(factors_list)
            row = dict(method={"base": "DP-Adam", "full": "DP-KFC", "a_only": "DP-KFC-A"}[geometry],
                       geometry=geometry, source=source, engine=engine, profiled=profile, profile_mode=profile_mode,
                       epsilon_target=epsilon, epsilon_spent=accountant.get_epsilon(DELTA), delta=DELTA,
                       noise_multiplier=sigma, sample_rate=sample_rate, seed=seed, epoch=epoch,
                       lr=lr, device=str(device),
                       train_size=len(train), train_examples=examples, validation_size=len(validation),
                       logical_batch_size=LOGICAL_BATCH_SIZE, physical_batch_size=physical_batch_size,
                       geometry_batch_size=GEOMETRY_BATCH_SIZE if ua else 0,
                       geometry_physical_batch_size=GEOMETRY_PHYSICAL_BATCH_SIZE,
                       preconditioned_layers=len(ua), trainable_parameters=sum(p.numel() for p in raw.parameters()),
                       train_loss=total_loss / examples, test_loss=test_loss, accuracy=accuracy, **stats,
                       clip_fraction=(factors < 1).float().mean().item() if profile else float("nan"),
                       mean_clip_factor=factors.mean().item() if profile else float("nan"),
                       norm_p50=norms.quantile(.5).item() if profile else float("nan"),
                       norm_p90=norms.quantile(.9).item() if profile else float("nan"),
                       norm_p99=norms.quantile(.99).item() if profile else float("nan"),
                       norm_max=norms.max().item() if profile else float("nan"),
                       logical_steps=steps, physical_steps=physical_steps, optimizer_steps=optimizer_steps,
                       noise_events=noise_events, accountant_steps=accountant_steps)
            rows.append(row)
            pd.DataFrame(rows).to_csv(output_dir / f"{geometry}_{source}_{engine}_{profile_mode}_eps{epsilon:g}_seed{seed}.csv", index=False)
            print(f"{geometry}/{source}/{engine}/{profile_mode} physical={physical_batch_size} eps={epsilon:g} seed={seed} epoch={epoch} "
                  f"examples={examples} accuracy={accuracy:.4f} algorithm={stats['algorithm_seconds']:.2f}s", flush=True)
    finally:
        if engine == "explicit":
            model.remove_hooks()
    summary = rows[-1].copy()
    for key in ("private_train_seconds", "algorithm_seconds", "geometry_build_seconds", "evaluation_seconds",
                *TIMINGS, *COUNTERS, "train_examples", "logical_steps", "physical_steps", "optimizer_steps", "noise_events", "accountant_steps"):
        summary[key] = sum(row[key] for row in rows)
    for key in (*MEMORY, "private_peak_allocated_bytes", "private_peak_reserved_bytes",
                "geometry_peak_allocated_bytes", "geometry_peak_reserved_bytes"):
        summary[key] = max(row[key] for row in rows)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true", help="All 3 geometries with BK practical and both controlled engines, seed=42, epsilon=3, one FULL epoch")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epsilon", type=float)
    parser.add_argument("--engine", choices=("auto", "explicit", "bk", "all"), default="auto")
    parser.add_argument("--lr", type=float, default=LR, help="Adam learning rate (default: %(default)s)")
    parser.add_argument("--gpu", type=int, help="CUDA device index among visible GPUs (e.g. 1)")
    parser.add_argument("--output_dir", type=Path, default=Path("results/distilbert_sst2_lr"))
    args = parser.parse_args()
    if args.epsilon is not None and args.epsilon <= 0:
        parser.error("--epsilon must be positive")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("--lr must be finite and positive")
    if args.gpu is not None:
        if args.gpu < 0 or not torch.cuda.is_available() or args.gpu >= torch.cuda.device_count():
            parser.error("--gpu must name an available CUDA device among visible GPUs")
        torch.cuda.set_device(args.gpu)
    seeds = [42] if args.fast else [args.seed] if args.seed is not None else SEEDS
    epsilons = [PROFILE_EPSILON] if args.fast else [args.epsilon] if args.epsilon is not None else EPSILONS
    epochs = 1 if args.fast else EPOCHS
    device = torch.device(f"cuda:{args.gpu}" if args.gpu is not None else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; lr={args.lr:g}; output_dir={args.output_dir}; seeds={seeds}; epsilons={epsilons}; epochs={epochs}; engine={'all' if args.fast else args.engine}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train, validation = load_data(tokenizer)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for epsilon in epsilons:
        profile = epsilon == PROFILE_EPSILON
        if args.fast or args.engine == "all" or (args.engine == "auto" and profile):
            runs = [("bk", BK_PHYSICAL_BATCH_SIZE, profile, "practical"),
                    ("bk", PROFILE_PHYSICAL_BATCH_SIZE, True, "controlled"),
                    ("explicit", PROFILE_PHYSICAL_BATCH_SIZE, True, "controlled")]
        elif args.engine == "explicit":
            runs = [("explicit", EXPLICIT_PHYSICAL_BATCH_SIZE, profile, "practical")]
        else:
            runs = [("bk", BK_PHYSICAL_BATCH_SIZE, profile, "practical")]
        for seed in seeds:
            for geometry, source in CONDITIONS:
                for engine, physical_batch_size, profiled, profile_mode in runs:
                    summaries.append(run_one(train, validation, tokenizer, geometry, source, engine,
                                             epsilon, seed, epochs, device, args.output_dir,
                                             physical_batch_size, profiled, profile_mode, lr=args.lr))
                    pd.DataFrame(summaries).to_csv(args.output_dir / "summary.csv", index=False)


if __name__ == "__main__":
    main()
