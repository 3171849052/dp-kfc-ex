"""Exp6 shuffled-minibatch training, synthetic KFC and full-Fisher Equil."""
from copy import deepcopy
from pathlib import Path
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from opacus import GradSampleModule
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier
from dp_kfac.models import SimpleCNN
from dp_kfac.optimizer import generate_pink_noise
from dp_kfac.recorder import KFACRecorder
from dp_kfac.covariance import compute_covariances, compute_inverse_sqrt
from dp_kfac.precondition import precondition_per_sample_gradients
from dp_kfac.privacy import (clip_and_noise_gradients,
    _compute_per_sample_norms_squared, _compute_clip_factors)
from .run_logging import MetricsCSVWriter, write_yaml, write_summary, rng_seeds


def timestamp(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    return time.perf_counter()


def load_data(c):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((.1307,), (.3081,))])
    return tuple(datasets.MNIST(c['data']['root'], train=train, download=True,
                               transform=transform) for train in (True, False))


def pink_batches(c, device):
    s = c['synthetic']
    for _ in range(s['samples'] // s['batch_size']):
        yield (generate_pink_noise(s['batch_size'], (1, 28, 28), device),
               torch.randint(0, 10, (s['batch_size'],), device=device))


def build_kfc(model, c, device):
    recorder = KFACRecorder(model)
    recorder.enable()
    for index, (x, y) in enumerate(pink_batches(c, device)):
        model.zero_grad(set_to_none=True)
        F.cross_entropy(model(x), y, reduction='sum').backward()
        cov = compute_covariances(model, recorder.activations, recorder.backprops, eps=0.0)
        if index == 0:
            total = cov
        else:
            for name in total.A:
                total.A[name].add_(cov.A[name])
                total.G[name].add_(cov.G[name])
        recorder.clear()
    recorder.remove()
    model.zero_grad(set_to_none=True)
    for name in total.A:
        total.A[name].div_(c['synthetic']['samples'] // c['synthetic']['batch_size'])
        total.G[name].div_(c['synthetic']['samples'] // c['synthetic']['batch_size'])
    return compute_inverse_sqrt(total, damping=c['kfac']['damping'])


def build_equil(model, c, device, epoch):
    params = [p for p in model.parameters() if p.requires_grad]
    k = c['equil']['probes']
    with torch.random.fork_rng(devices=[device.index] if device.type == 'cuda' else []):
        torch.manual_seed(c['seed'] + 30000 + epoch)
        probes = {p: torch.randint(0, 2, (p.numel(), k), device=device)
                  .to(p.dtype).mul_(2).sub_(1) for p in params}
    products = {p: torch.zeros_like(probes[p]) for p in params}
    for x, y in pink_batches(c, device):
        model.zero_grad(set_to_none=True)
        F.cross_entropy(model(x), y, reduction='sum').backward()
        with torch.no_grad():
            projection = torch.zeros(len(x), k, device=device)
            for p in params:
                projection.add_(p.grad_sample.flatten(1) @ probes[p])
            for p in params:
                products[p].add_(p.grad_sample.flatten(1).T @ projection)
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        e = {p: (v / c['synthetic']['samples']).square().mean(1).sqrt().view_as(p)
             for p, v in products.items()}
        values = torch.cat([v.flatten() for v in e.values()])
        gamma = c['equil']['tau'] * values.median()
        scales = {p: (v + gamma).rsqrt() for p, v in e.items()}
        normalizer = (sum(v.log().sum() for v in scales.values()) / values.numel()).exp()
        return {p: (v / normalizer).clamp(c['equil']['scale_min'], c['equil']['scale_max'])
                for p, v in scales.items()}


def build_preconditioner(model, c, device, epoch):
    with torch.random.fork_rng(devices=[device.index] if device.type == 'cuda' else []):
        torch.manual_seed(c['seed'] + 10000 + epoch)
        if c['algorithm'] == 'dp_kfc':
            return build_kfc(model, c, device)
        return build_equil(model, c, device, epoch)


@torch.no_grad()
def preconditioner_metrics(model, algorithm, state):
    if algorithm == 'dp_sgd':
        return 0., dict.fromkeys(('gain_p10', 'gain_median', 'gain_p90', 'gain_max'), 1.)
    if algorithm == 'dp_kfc':
        a, g = state
        tensors = [*a.values(), *g.values()]
        gains = []
        for name, module in model._module.named_modules():
            if name in a:
                gains.append((g[name].norm(dim=0)[:, None] * a[name].norm(dim=0)[None, :]).flatten())
    else:
        tensors = list(state.values())
        gains = [v.flatten() for v in tensors]
    values = torch.cat(gains)
    quantiles = values.quantile(values.new_tensor([.1, .5, .9, 1.])).tolist()
    return (sum(t.numel() * t.element_size() for t in tensors) / 2**20,
            dict(zip(('gain_p10', 'gain_median', 'gain_p90', 'gain_max'), quantiles)))


def private_step(model, optimizer, accountant, x, y, c, state, sigma, sample_rate):
    model.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(x), y, reduction='sum')
    loss.backward()
    if c['algorithm'] == 'dp_kfc':
        precondition_per_sample_gradients(model, *state)
    elif c['algorithm'] == 'dp_equil':
        with torch.no_grad():
            for p, scale in state.items():
                p.grad_sample.mul_(scale)
    params = [p for p in model.parameters() if p.requires_grad]
    norms = _compute_per_sample_norms_squared(params, len(x), x.device)
    bound = c['privacy']['max_grad_norm']
    factors = _compute_clip_factors(norms, bound)
    clipped, factor_sum = (norms.sqrt() > bound).sum().item(), factors.sum().item()
    clip_and_noise_gradients(model, sigma, bound, len(x))
    optimizer.step()
    accountant.step(noise_multiplier=sigma, sample_rate=sample_rate)
    return loss.item(), clipped, factor_sum


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss, correct = 0., 0
    for x, y in loader:
        y = y.to(device)
        output = model(x.to(device))
        loss += F.cross_entropy(output, y, reduction='sum').item()
        correct += (output.argmax(1) == y).sum().item()
    return loss / len(loader.dataset), correct / len(loader.dataset)


def train(c, run_dir):
    directory = Path(run_dir).resolve()
    # Prepared runs are single-use; never append another training to an old run.
    from .config import load_config
    if load_config(directory / 'config.yaml') != c:
        raise ValueError('config differs from the prepared run')
    if len((directory / 'metrics.csv').read_text().splitlines()) != 1 or (directory / 'summary.json').exists():
        raise ValueError('run directory has already been used')
    device = torch.device('cuda:0' if c['runtime']['device'] == 'cuda' else 'cpu')
    torch.set_num_threads(c['runtime']['threads'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = c['runtime']['deterministic']
    torch.use_deterministic_algorithms(c['runtime']['deterministic'])
    torch.manual_seed(c['seed'])
    model = GradSampleModule(SimpleCNN().to(device), loss_reduction='sum')
    t, d, p = c['training'], c['data'], c['privacy']
    optimizer = torch.optim.SGD(model.parameters(), lr=t['learning_rate'],
                                momentum=t['momentum'], weight_decay=t['weight_decay'])
    train_data, test_data = load_data(c)
    if d['batch_size'] > len(train_data):
        raise ValueError('batch_size exceeds training dataset size')
    loader = DataLoader(train_data, batch_size=d['batch_size'], shuffle=True,
        num_workers=d['num_workers'], generator=torch.Generator().manual_seed(c['seed']))
    test_loader = DataLoader(test_data, batch_size=d['eval_batch_size'],
        num_workers=d['num_workers'], generator=torch.Generator().manual_seed(c['seed']))
    sample_rate = d['batch_size'] / len(train_data)
    sigma = get_noise_multiplier(target_epsilon=p['epsilon'], target_delta=p['delta'],
        sample_rate=sample_rate, steps=t['epochs'] * len(loader), accountant='rdp')
    resolved = deepcopy(c)
    resolved.update(train_size=len(train_data), test_size=len(test_data), steps_per_epoch=len(loader),
        total_steps=t['epochs'] * len(loader), sample_rate=sample_rate, noise_multiplier=sigma,
        actual_device=str(device), physical_gpu_index=c['runtime']['gpu'] if device.type == 'cuda' else None,
        gpu_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        run_directory=str(directory), rng_seeds=rng_seeds(c))
    write_yaml(directory / 'resolved_config.yaml', resolved)
    writer = MetricsCSVWriter(directory / 'metrics.csv')
    accountant = RDPAccountant()
    state, global_step, best_accuracy = None, 0, 0.
    run_peak, max_storage, epoch_times = None, 0., []
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    training_start = timestamp(device)
    for epoch in range(1, t['epochs'] + 1):
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        model.train()
        epoch_start = timestamp(device)
        build_seconds = 0.
        if c['algorithm'] != 'dp_sgd' and (epoch - 1) % c['synthetic']['refresh_every_epochs'] == 0:
            state = build_preconditioner(model, c, device, epoch)
            build_seconds = timestamp(device) - epoch_start
        storage, gains = preconditioner_metrics(model, c['algorithm'], state)
        max_storage = max(max_storage, storage)
        train_start = timestamp(device)
        loss_sum, clipped, factor_sum, count = 0., 0., 0., 0
        for x, y in loader:
            loss, clip, factor = private_step(model, optimizer, accountant,
                x.to(device), y.to(device), c, state, sigma, sample_rate)
            loss_sum += loss
            clipped += clip
            factor_sum += factor
            count += len(x)
            global_step += 1
        train_seconds = timestamp(device) - train_start
        test_loss, accuracy = evaluate(model, test_loader, device)
        evaluation_end = timestamp(device)
        epoch_seconds = evaluation_end - epoch_start
        peak = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else None
        if peak is not None:
            run_peak = max(run_peak or 0., peak)
        epoch_times.append(epoch_seconds)
        best_accuracy = max(best_accuracy, accuracy)
        epsilon = accountant.get_epsilon(delta=p['delta'])
        writer.append(dict(epoch=epoch, global_step=global_step, train_loss=loss_sum/count,
            test_loss=test_loss, test_accuracy=accuracy, epsilon_spent=epsilon,
            noise_multiplier=sigma, sample_rate=sample_rate, clip_fraction=clipped/count,
            mean_clip_factor=factor_sum/count, precond_build_seconds=build_seconds,
            epoch_train_seconds=train_seconds, epoch_total_seconds=epoch_seconds,
            peak_allocated_mb_epoch=peak, preconditioner_storage_mb=storage, **gains))
        print(f"{c['algorithm']} epoch={epoch} accuracy={accuracy:.4f} epsilon={epsilon:.4f}", flush=True)
    summary = dict(status='completed', algorithm=c['algorithm'], seed=c['seed'],
        completed_epochs=t['epochs'], global_step=global_step, final_test_loss=test_loss,
        final_test_accuracy=accuracy, best_test_accuracy=best_accuracy, final_epsilon=epsilon,
        noise_multiplier=sigma, training_total_seconds=evaluation_end-training_start,
        mean_epoch_total_seconds=sum(epoch_times)/len(epoch_times), max_peak_allocated_mb=run_peak,
        max_preconditioner_storage_mb=max_storage)
    write_summary(directory, summary)
    return summary
