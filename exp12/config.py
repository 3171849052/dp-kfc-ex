import argparse
from pathlib import Path
import torch


def parse():
    p = argparse.ArgumentParser(description='Frozen SimpleCNN curvature diagnostics')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--synthetic-batches', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--private-samples', type=int, default=2560)
    p.add_argument('--damping', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--label-seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output', type=Path, default=Path(__file__).parent/'results')
    a = p.parse_args()
    if a.smoke:
        a.synthetic_batches, a.batch_size, a.private_samples = 1, 4, 4
        a.label_seeds = [0, 1]
    return a
