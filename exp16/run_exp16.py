#!/usr/bin/env python
"""Run one exponent/seed, or the single tiny verification suite."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import csv
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
sys.dont_write_bytecode = True
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from opacus import GradSampleModule
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier
import yaml
from dp_kfac.models import SimpleCNN
from dp_kfac.standalone.trainer import load_data, build_optimizer, evaluate
from dp_kfac.privacy import clip_and_noise_gradients, _compute_per_sample_norms_squared
from exp16.preconditioner import Preconditioner


def configuration(q, seed=42, smoke=False, mode="nested"):
    extra = yaml.safe_load((ROOT / 'exp16/configs/mnist.yaml').read_text())
    c = yaml.safe_load((ROOT / extra['base']).read_text())
    for section in ('data', 'synthetic'):
        c[section].update(extra[section])
    c['accounting_convention'] = 'fixed-batch shuffled; drop_last=True; RDP q=batch_size/train_size, steps=epochs*floor(train_size/batch_size)'
    c.update(lbfgs=extra['lbfgs'], q=q, mode=mode, seed=seed, smoke=smoke)
    c['data']['root'] = str(ROOT / 'exp16/data')
    c['algorithm'] = 'synthetic_two_stage'
    c['output']['root'] = str(ROOT / 'exp16/results')
    if smoke:
        c['data']['batch_size'] = 4
        c['data']['eval_batch_size'] = 16
        c['synthetic'].update(samples=24, batch_size=8)
        c['training']['epochs'] = 2
    return c


def private_step(model, optimizer, x, y, c, state, sigma):
    model.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(x), y, reduction='sum')
    loss.backward()
    state.apply(model, c['q'])
    params = list(model.parameters())
    norms = _compute_per_sample_norms_squared(params, len(x), x.device).sqrt()
    q = norms.quantile(norms.new_tensor([.5, .9, .99])).tolist()
    before = [parameter.detach().clone() for parameter in params]
    clip_and_noise_gradients(model, sigma, c['privacy']['max_grad_norm'], len(x))
    optimizer.step()
    update = sum((parameter.detach() - old).square().sum()
                 for parameter, old in zip(params, before)).sqrt().item()
    return dict(**state.norm_metrics, train_loss=loss.item()/len(x), preclip_norm_mean=norms.mean().item(),
                preclip_norm_p50=q[0], preclip_norm_p90=q[1], preclip_norm_p99=q[2],
                clip_fraction=(norms > c['privacy']['max_grad_norm']).float().mean().item(),
                update_norm=update)


def run(q, seed=42, smoke=False, mode="nested"):
    c = configuration(q, seed, smoke, mode)
    directory = ROOT / 'exp16/results' / ('smoke' if smoke else 'formal') / f'{mode}_q{q:g}_seed{seed}'
    directory.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda:0')
    torch.set_num_threads(c['runtime']['threads'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    model = GradSampleModule(SimpleCNN().to(device), loss_reduction='sum')
    optimizer = build_optimizer(model, c['training'])
    train_data, test_data = load_data(c)
    if smoke:
        train_data, test_data = Subset(train_data, range(8)), Subset(test_data, range(32))
    d, privacy = c['data'], c['privacy']
    loader = DataLoader(train_data, batch_size=d['batch_size'], shuffle=True, drop_last=d['drop_last'],
                        num_workers=d['num_workers'], generator=torch.Generator().manual_seed(seed))
    test_loader = DataLoader(test_data, batch_size=d['eval_batch_size'],
                             generator=torch.Generator().manual_seed(seed))
    sample_rate = d['batch_size'] / len(train_data)
    steps = c['training']['epochs'] * len(loader)
    sigma = get_noise_multiplier(target_epsilon=privacy['epsilon'], target_delta=privacy['delta'],
                                 sample_rate=sample_rate, steps=steps, accountant=privacy['accountant'])
    c.update(noise_multiplier=sigma, sample_rate=sample_rate, total_steps=steps,
             train_size=len(train_data), test_size=len(test_data), actual_device=str(device))
    (directory / 'config.json').write_text(json.dumps(c, indent=2) + '\n')
    state = Preconditioner(c, device)
    accountant = RDPAccountant()
    global_step = 0
    with (directory / 'metrics.csv').open('w') as metrics, (directory / 'training.csv').open('w') as training, (directory / 'train.log').open('w') as log:
        step_writer = epoch_writer = None
        for epoch in range(1, c['training']['epochs'] + 1):
            model.train()
            if (epoch - 1) % c['synthetic']['refresh_every_epochs'] == 0:
                state.refresh(model, epoch)
            loss_sum, count = 0., 0
            for x, y in loader:
                row = private_step(model, optimizer, x.to(device), y.to(device), c, state, sigma)
                accountant.step(noise_multiplier=sigma, sample_rate=sample_rate)
                global_step += 1
                loss_sum += row['train_loss'] * len(x)
                count += len(x)
                row.update(step=global_step, epoch=epoch, **state.diagnostics())
                if step_writer is None:
                    step_writer = csv.DictWriter(metrics, fieldnames=list(row))
                    step_writer.writeheader()
                step_writer.writerow(row)
                metrics.flush()
            test_loss, accuracy = evaluate(model, test_loader, device)
            epsilon = accountant.get_epsilon(delta=privacy['delta'])
            row = dict(epoch=epoch, step=global_step, train_loss=loss_sum/count,
                       test_loss=test_loss, test_accuracy=accuracy, epsilon_spent=epsilon)
            if epoch_writer is None:
                epoch_writer = csv.DictWriter(training, fieldnames=list(row))
                epoch_writer.writeheader()
            epoch_writer.writerow(row)
            training.flush()
            message = f'mode={mode} q={q:g} seed={seed} epoch={epoch} accuracy={accuracy:.4f} epsilon={epsilon:.6f}'
            print(message, flush=True)
            log.write(message + '\n')
            log.flush()
    summary = dict(q=q, mode=mode, seed=seed, smoke=smoke, test_accuracy=accuracy,
                   epsilon_spent=epsilon, steps=global_step, diagnostics=state.diagnostics(),
                   accounting_convention=c['accounting_convention'],
                   drop_last=d['drop_last'], sample_rate=sample_rate, noise_multiplier=sigma,
                   total_steps=steps, synthetic_batches=c['synthetic']['samples']//c['synthetic']['batch_size'])
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


CASES = [('hessian', 0.)] + [(mode, q) for q in (.25, .5)
                                      for mode in ('hessian', 'fisher', 'nested')]


def analyze(smoke=False):
    directory = ROOT / 'exp16/results' / ('smoke' if smoke else 'formal')
    records = [json.loads((directory / f'{mode}_q{q:g}_seed42/summary.json').read_text())
               for mode, q in CASES]
    keys = ('mode', 'q', 'seed', 'test_accuracy', 'epsilon_spent', 'steps')
    with (directory / 'summary.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows({key: record[key] for key in keys} for record in records)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--q', type=float, choices=[0., .25, .5])
    parser.add_argument('--mode', choices=['hessian', 'fisher', 'nested'], default='nested')
    parser.add_argument('--seed', type=int, choices=[42], default=42)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--summarize', action='store_true')
    args = parser.parse_args()
    if args.summarize:
        analyze(args.smoke)
    elif args.smoke and args.q is None:
        for mode, q in CASES:
            run(q, args.seed, True, mode)
        analyze(True)
    else:
        if args.q is None:
            parser.error('--q is required')
        run(args.q, args.seed, args.smoke, args.mode)
