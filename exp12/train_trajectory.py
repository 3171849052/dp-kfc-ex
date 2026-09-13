"""Ordinary non-private SGD, used only to produce frozen model states."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
import argparse
import json
import math
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from dp_kfac.models import SimpleCNN
from exp12.runtime import runtime

SAVE_STEPS = (0, 50, 100, 250, 500, 1000, 2000)


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda')
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--max-steps', type=int)
    p.add_argument('--diagnostic-samples', type=int, default=2560)
    p.add_argument('--test-samples', type=int, default=10000)
    p.add_argument('--output', type=Path)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args(argv)
    if args.smoke:
        args.max_steps, args.diagnostic_samples, args.test_samples = 2, 32, 64
    if args.output is None:
        args.output = ROOT/'exp12/checkpoints'/('smoke' if args.smoke else '')
    assert args.output.resolve().is_relative_to(ROOT/'exp12')
    return args


@torch.no_grad()
def state_metrics(model, diagnostic_loader, test_loader, device):
    model.eval()
    loss = entropy = confidence = 0.
    count = correct = test_count = 0
    for x, y in diagnostic_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        logp = logits.double().log_softmax(-1)
        p = logp.exp()
        loss += F.cross_entropy(logits, y, reduction='sum').item()
        entropy += (-(p*logp).sum()).item()
        confidence += p.max(-1).values.sum().item()
        count += len(x)
    for x, y in test_loader:
        prediction = model(x.to(device)).argmax(-1)
        correct += (prediction == y.to(device)).sum().item()
        test_count += len(x)
    entropy /= count
    return dict(train_loss=loss/count, test_accuracy=correct/test_count,
                mean_max_probability=confidence/count,
                mean_prediction_entropy=entropy,
                normalized_entropy=entropy/math.log(10),
                mean_kl_to_uniform=math.log(10)-entropy)


def train(args):
    with runtime(args.device):
        return _train(args)


def _train(args):
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    model = SimpleCNN().to(args.device)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((.1307,), (.3081,))])
    train_data = datasets.MNIST(ROOT/'exp9/data', train=True, download=False, transform=transform)
    test_data = datasets.MNIST(ROOT/'exp9/data', train=False, download=False, transform=transform)
    indices = torch.randperm(len(train_data), generator=torch.Generator().manual_seed(args.seed+2))[:args.diagnostic_samples].tolist()
    diagnostic_loader = DataLoader(Subset(train_data, indices), batch_size=256)
    test_loader = DataLoader(Subset(test_data, range(args.test_samples)), batch_size=256)
    train_loader = DataLoader(train_data, batch_size=256, shuffle=True,
                              generator=torch.Generator().manual_seed(args.seed))
    optimizer = torch.optim.SGD(model.parameters(), lr=.5, momentum=0, weight_decay=0)
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = dict(vars(args), model='SimpleCNN', dataset='MNIST', normalization=[.1307, .3081],
                    optimizer='SGD', lr=.5, momentum=0, weight_decay=0, batch_size=256,
                    shuffle=True, diagnostic_split='train', diagnostic_indices=indices,
                    cudnn_benchmark=False, cudnn_deterministic=True,
                    matmul_allow_tf32=False, cudnn_allow_tf32=False,
                    train_loss_definition='Mean cross entropy on the fixed diagnostic train subset',
                    epoch_definition='Completed steps divided by batches per epoch', save_steps=SAVE_STEPS)
    (args.output/'metadata.json').write_text(json.dumps(metadata, default=str, indent=2))
    rows = []
    def save(step, final=False):
        suffix = 'final' if final else f'step{step:06d}'
        path = args.output/f'sgd_seed{args.seed}_{suffix}.pt'
        torch.save(model.state_dict(), path)
        if final and any(row['step'] == step for row in rows):
            return
        row = dict(checkpoint_path=str(path.resolve().relative_to(ROOT)), seed=args.seed,
                   step=step, epoch=step/len(train_loader),
                   **state_metrics(model, diagnostic_loader, test_loader, args.device))
        rows.append(row)
        pd.DataFrame(rows).to_csv(args.output/'trajectory.csv', index=False)
        print(f'{path.name}: step={step}, accuracy={row["test_accuracy"]:.4f}, '
              f'entropy/log(10)={row["normalized_entropy"]:.4f}', flush=True)
    step = 0
    save(0)
    limit = args.epochs*len(train_loader)
    if args.max_steps is not None:
        limit = min(limit, args.max_steps)
    for _ in range(args.epochs):
        for x, y in train_loader:
            if step >= limit:
                break
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x.to(args.device)), y.to(args.device))
            loss.backward()
            optimizer.step()
            step += 1
            if step in SAVE_STEPS:
                save(step)
        if step >= limit:
            break
    save(step, final=True)
    return pd.DataFrame(rows)


if __name__ == '__main__':
    train(parse(sys.argv[1:]))
