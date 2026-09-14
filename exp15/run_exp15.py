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
from exp15.preconditioner import SyntheticKLBFGS


def configuration(p, seed, smoke=False):
    extra = yaml.safe_load((ROOT / 'exp15/configs/mnist.yaml').read_text())
    c = yaml.safe_load((ROOT / extra['base']).read_text())
    for section in ('data', 'synthetic'):
        c[section].update(extra[section])
    c['accounting_convention'] = 'fixed-batch shuffled; drop_last=True; RDP q=batch_size/train_size, steps=epochs*floor(train_size/batch_size)'
    c.update(lbfgs=extra['lbfgs'], p=p, seed=seed, smoke=smoke)
    c['data']['root'] = str(ROOT / c['data']['root'])
    c['algorithm'] = 'synthetic_klbfgs_h_power'
    c['output']['root'] = str(ROOT / 'exp15/results')
    if smoke:
        c['data']['batch_size'] = 4
        c['data']['eval_batch_size'] = 16
        c['synthetic'].update(samples=24, batch_size=8)
        c['training']['epochs'] = 2
    return c


@torch.no_grad()
def check_factors(state):
    errors = []
    for factors in state.factors.values():
        for factor in factors:
            v = torch.arange(1, factor.pairs['s'].shape[1] + 1,
                             device=state.device, dtype=torch.float64)
            torch.testing.assert_close(factor.power(v, 0), v, rtol=0, atol=0)
            expected = factor.hv(v)
            actual = factor.power(v, 1)
            torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-8)
            errors.append(float((actual - expected).norm() / expected.norm()))
            for p in (.25, .5, .75):
                assert torch.isfinite(factor.power(v, p)).all()
    return max(errors)


def private_step(model, optimizer, x, y, c, state, sigma):
    model.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(x), y, reduction='sum')
    loss.backward()
    state.apply(model, c['p'])
    params = list(model.parameters())
    norms = _compute_per_sample_norms_squared(params, len(x), x.device).sqrt()
    q = norms.quantile(norms.new_tensor([.5, .9, .99])).tolist()
    before = [parameter.detach().clone() for parameter in params]
    clip_and_noise_gradients(model, sigma, c['privacy']['max_grad_norm'], len(x))
    optimizer.step()
    update = sum((parameter.detach() - old).square().sum()
                 for parameter, old in zip(params, before)).sqrt().item()
    return dict(train_loss=loss.item()/len(x), preclip_norm_mean=norms.mean().item(),
                preclip_norm_p50=q[0], preclip_norm_p90=q[1], preclip_norm_p99=q[2],
                clip_fraction=(norms > c['privacy']['max_grad_norm']).float().mean().item(),
                update_norm=update)


def run(p, seed, smoke=False):
    c = configuration(p, seed, smoke)
    directory = ROOT / 'exp15/results' / ('smoke' if smoke else 'formal') / f'p{p:g}_seed{seed}'
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
    if smoke:
        from exp15.verification import CheckedSyntheticKLBFGS
        state = CheckedSyntheticKLBFGS(c, device)
    else:
        state = SyntheticKLBFGS(c, device)
    accountant = RDPAccountant()
    global_step = 0
    with (directory / 'metrics.csv').open('w') as metrics, (directory / 'training.csv').open('w') as training, (directory / 'train.log').open('w') as log:
        step_writer = epoch_writer = None
        for epoch in range(1, c['training']['epochs'] + 1):
            model.train()
            if (epoch - 1) % c['synthetic']['refresh_every_epochs'] == 0:
                state.refresh(model, epoch)
                if smoke:
                    error = check_factors(state)
                    min_pairs = min(len(f.pairs['s']) for fs in state.factors.values() for f in fs)
                    log.write(f'epoch={epoch} p0_identity=passed p1_relative_error={error:.3e} fractional_finite=passed probe_reset=passed synthetic_batches={len(state.before_pair_counts)} min_memory_pairs={min_pairs}\n')
            loss_sum, count = 0., 0
            for x, y in loader:
                row = private_step(model, optimizer, x.to(device), y.to(device), c, state, sigma)
                accountant.step(noise_multiplier=sigma, sample_rate=sample_rate)
                global_step += 1
                loss_sum += row['train_loss'] * len(x)
                count += len(x)
                diag = list(state.diagnostics().values())
                row.update(step=global_step, epoch=epoch,
                           **{key: sum(v[key] for v in diag) for key in diag[0]})
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
            message = f'p={p:g} seed={seed} epoch={epoch} accuracy={accuracy:.4f} epsilon={epsilon:.6f}'
            print(message, flush=True)
            log.write(message + '\n')
            log.flush()
    summary = dict(p=p, seed=seed, smoke=smoke, test_accuracy=accuracy,
                   epsilon_spent=epsilon, steps=global_step, diagnostics=state.diagnostics(),
                   accounting_convention=c['accounting_convention'],
                   drop_last=d['drop_last'], sample_rate=sample_rate, noise_multiplier=sigma,
                   total_steps=steps, synthetic_batches=c['synthetic']['samples']//c['synthetic']['batch_size'])
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def analyze(smoke=False):
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory = ROOT / 'exp15/results' / ('smoke' if smoke else 'formal')
    paths = sorted(directory.glob('p*_seed*/summary.json'))
    frame = pd.DataFrame([json.loads(path.read_text()) for path in paths])
    if not smoke:
        # Old protocol results must not be pooled with corrected runs.
        assert {'total_steps', 'synthetic_batches', 'drop_last'} <= set(frame.columns), 'Old protocol results: rerun all formal p/seed combinations before summarizing'
        assert (frame.total_steps == 1170).all() and (frame.synthetic_batches == 10).all()
        assert not frame.smoke.any() and frame.drop_last.all()
    table = frame.groupby('p').test_accuracy.agg(['count', 'mean', 'std']).reset_index()
    table.to_csv(directory / 'summary.csv', index=False)
    fig, ax = plt.subplots()
    ax.errorbar(table.p, table['mean'], yerr=table['std'].fillna(0), marker='o', capsize=4)
    ax.set(xlabel='Inverse-Hessian exponent p', ylabel='Test accuracy',
           title='Tiny smoke (not a performance result)' if smoke else 'MNIST + CNN, epsilon=1 (mean ± sample std)')
    fig.savefig(directory / 'test_accuracy_vs_h_power.png', bbox_inches='tight')
    plt.close(fig)
    fig, ax = plt.subplots()
    for path in paths:
        metrics = pd.read_csv(path.parent / 'metrics.csv')
        ax.plot(metrics.step, metrics.clip_fraction, label=path.parent.name, alpha=.75)
    ax.set(xlabel='Logical step', ylabel='Clipping fraction', ylim=(0, 1.05))
    ax.legend(fontsize=7)
    fig.savefig(directory / 'clip_fraction_vs_step.png', bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--p', type=float, choices=[0., .25, .5, .75, 1.])
    parser.add_argument('--seed', type=int, choices=[0, 1], default=0)
    parser.add_argument('--smoke', action='store_true', help='one tiny suite covering p=0,0.5,1')
    parser.add_argument('--summarize', action='store_true')
    args = parser.parse_args()
    if args.summarize:
        analyze(args.smoke)
    elif args.smoke:
        for p in (0., .5, 1.):
            run(p, 0, True)
        analyze(True)
    else:
        if args.p is None:
            parser.error('--p is required for a formal single run')
        run(args.p, args.seed)
