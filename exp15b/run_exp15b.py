#!/usr/bin/env python
"""Run the damping sweep, or two tiny smoke runs."""
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
from exp15b.config import configuration, ROOT as EXP_ROOT, LAMBDAS, SEEDS
from dp_kfac.models import SimpleCNN
from dp_kfac.standalone.trainer import load_data, build_optimizer, evaluate
from dp_kfac.privacy import clip_and_noise_gradients, _compute_per_sample_norms_squared
from exp15.preconditioner import SyntheticKLBFGS


from exp15.run_exp15 import private_step, check_factors


def factor_diagnostics(state):
    result = state.diagnostics()
    for name, values in result.items():
        attempts = values['accepted'] + values['rejected']
        eligible = attempts if name.endswith('/g') else 0
        values['damping_eligible_pairs'] = eligible
        values['powell_rate'] = values['powell_damping'] / eligible if eligible else 0.
        values['modified_rate'] = values['modified_damping'] / eligible if eligible else 0.
        values['damping_trigger_rate'] = (values['powell_rate'] + values['modified_rate']) / 2
    return result


def damping_rates(diagnostics):
    eligible = sum(v['damping_eligible_pairs'] for v in diagnostics.values())
    powell = sum(v['powell_damping'] for v in diagnostics.values()) / eligible
    modified = sum(v['modified_damping'] for v in diagnostics.values()) / eligible
    return dict(powell_rate=powell, modified_rate=modified,
                damping_trigger_rate=(powell + modified) / 2)


def run(lambda_damping, seed, smoke=False):
    c = configuration(lambda_damping, seed, smoke)
    directory = EXP_ROOT / 'results' / ('smoke' if smoke else 'formal') / f'lambda{lambda_damping:g}_seed{seed}'
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
    loader = DataLoader(train_data, batch_size=d['batch_size'], shuffle=d['shuffle'], drop_last=d['drop_last'],
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
            epoch_metrics = []
            for x, y in loader:
                row = private_step(model, optimizer, x.to(device), y.to(device), c, state, sigma)
                epoch_metrics.append(row.copy())
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
                       test_loss=test_loss, test_accuracy=accuracy, epsilon_spent=epsilon,
                       **{key: sum(r[key] for r in epoch_metrics)/len(epoch_metrics)
                          for key in ('clip_fraction', 'preclip_norm_mean', 'preclip_norm_p90', 'preclip_norm_p99')})
            diagnostics = factor_diagnostics(state)
            (directory / f'factors_epoch{epoch}.json').write_text(json.dumps(diagnostics, indent=2) + '\n')
            row.update(damping_rates(diagnostics))
            if epoch_writer is None:
                epoch_writer = csv.DictWriter(training, fieldnames=list(row))
                epoch_writer.writeheader()
            epoch_writer.writerow(row)
            training.flush()
            message = f'lambda={lambda_damping:g} seed={seed} epoch={epoch} accuracy={accuracy:.4f} epsilon={epsilon:.6f}'
            print(message, flush=True)
            log.write(message + '\n')
            log.flush()
    summary = dict(lambda_damping=lambda_damping, factor_damping=c['lbfgs']['factor_damping'], p=c['p'],
                   test_loss=test_loss,
                   **{key: row[key] for key in ('clip_fraction', 'preclip_norm_mean', 'preclip_norm_p90', 'preclip_norm_p99', 'powell_rate', 'modified_rate', 'damping_trigger_rate')}, seed=seed, smoke=smoke, test_accuracy=accuracy,
                   epsilon_spent=epsilon, steps=global_step, diagnostics=diagnostics,
                   accounting_convention=c['accounting_convention'],
                   drop_last=d['drop_last'], sample_rate=sample_rate, noise_multiplier=sigma,
                   total_steps=steps, synthetic_batches=c['synthetic']['samples']//c['synthetic']['batch_size'])
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    for damping in (LAMBDAS[0], LAMBDAS[-1]) if args.smoke else LAMBDAS:
        for seed in (SEEDS[0],) if args.smoke else SEEDS:
            run(damping, seed, args.smoke)
    from exp15b.analyze import analyze
    analyze(args.smoke)
