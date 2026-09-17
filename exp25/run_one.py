"""Run one fixed-protocol condition; --smoke uses one real private batch."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import argparse
import hashlib
import json
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from opacus.accountants import RDPAccountant
from exp25 import config as c
from exp25.model import make_model
from exp25.data import load_data, private_loader, auxiliary, rng
from exp25.geometry import build
from exp25.methods import aggregate, update


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


@torch.no_grad()
def evaluate(model, loader, device):
    loss, correct, count = 0., 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss += F.cross_entropy(logits, y, reduction='sum').item()
        correct += (logits.argmax(1) == y).sum().item()
        count += len(y)
    return loss / count, correct / count


def run(method, seed, smoke=False, device=None):
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    kind, source = c.condition(method)
    model = make_model(seed, device)
    init_hash = hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes()
                                     for p in model.classifier.parameters())).hexdigest()
    train, test, public = load_data()
    assert len(train) == c.TRAIN_SAMPLES
    if smoke:
        test = Subset(test, range(c.BATCH_SIZE))
    test_loader = DataLoader(test, batch_size=c.BATCH_SIZE, num_workers=0)
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=c.LR)
    accountant = RDPAccountant()
    sigma = c.noise_multiplier()
    noise_rng = rng(seed, 'dp_noise', device=device)
    folder = c.ROOT / 'results' / ('smoke' if smoke else 'formal')
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'{method}_{seed}.jsonl'
    best = 0.
    with path.open('w') as output:
        for epoch in range(1, (1 if smoke else c.EPOCHS) + 1):
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            sync(device)
            start = time.perf_counter()
            if source == 'none':
                operator, diagnostics = build(model, None, None, kind)
            else:
                aux_x, aux_y = auxiliary(source, train, public, seed, epoch, device)
                operator, diagnostics = build(model, aux_x, aux_y, kind)
                del aux_x, aux_y
            sync(device)
            geometry_time = time.perf_counter() - start
            loader = private_loader(train, seed, epoch, smoke)
            order_hash = hashlib.sha256(bytes(str(loader.dataset.indices), 'ascii')).hexdigest()
            norms_all, factors_all, train_loss = [], [], 0.
            sync(device)
            start = time.perf_counter()
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                losses, summed, norms, factors, count = aggregate(model, x, y, operator)
                update(model, optimizer, summed, sigma, len(y), noise_rng, accountant)
                train_loss += losses.sum().item()
                norms_all.append(norms.cpu())
                factors_all.append(factors.cpu())
            sync(device)
            private_time = time.perf_counter() - start
            test_loss, accuracy = evaluate(model, test_loader, device)
            best = max(best, accuracy)
            norms, factors = torch.cat(norms_all), torch.cat(factors_all)
            steps = sum(entry[2] for entry in accountant.history)
            assert steps == epoch * (1 if smoke else c.STEPS_PER_EPOCH)
            row = dict(method=method, source=source, geometry=kind, seed=seed, epoch=epoch,
                smoke=smoke, oracle_non_deployable_upper_bound=source == 'oracle',
                privacy_scope='training_only_excludes_oracle_geometry_and_private_diagnostics',
                sampling='fixed_size_shuffle_drop_last; nominal_Opacus_RDP',
                classifier_initialization_sha256=init_hash, private_order_sha256=order_hash,
                train_loss=train_loss/len(norms), test_loss=test_loss, test_accuracy=accuracy,
                best_accuracy=best, noise_multiplier=sigma,
                epsilon_spent=accountant.get_epsilon(c.DELTA), accountant_steps=steps,
                target_epsilon=c.EPSILON, delta=c.DELTA, sample_rate=c.SAMPLE_RATE,
                scheduled_private_steps=c.TOTAL_STEPS, logical_batch_size=c.BATCH_SIZE,
                physical_batch_size=c.BATCH_SIZE, clip_norm=c.CLIP, optimizer='Adam', lr=c.LR,
                clip_fraction=(factors < 1).float().mean().item(), mean_clip_factor=factors.mean().item(),
                transformed_norm_p50=norms.quantile(.5).item(), transformed_norm_p90=norms.quantile(.9).item(),
                transformed_norm_p99=norms.quantile(.99).item(), transformed_norm_max=norms.max().item(),
                geometry_diagnostics=diagnostics, geometry_build_time=geometry_time,
                private_training_time=private_time,
                cuda_peak_memory=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0,
                trainable_parameter_count=sum(p.numel() for p in model.parameters() if p.requires_grad),
                first_pass_parameter_grad_count=count, pretrained=True, input_size=240)
            output.write(json.dumps(row, allow_nan=False) + '\n')
            output.flush()
            print(f'{method} seed={seed} epoch={epoch} accuracy={accuracy:.6f} steps={steps}', flush=True)
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=c.CONDITIONS, required=True)
    parser.add_argument('--seed', type=int, choices=c.SEEDS, default=42)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--device', default=None)
    args = parser.parse_args()
    run(args.method, args.seed, args.smoke, args.device)
