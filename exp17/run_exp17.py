#!/usr/bin/env python
"""Run identity DP-SGD with same-probe Fisher / K-LBFGS diagnostics."""
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
from exp17.curvature import Diagnostic, TinyCNN, save_checkpoint


def configuration(smoke=False):
    c = yaml.safe_load((ROOT / 'exp17/configs/mnist.yaml').read_text())
    c['data']['root'] = str(ROOT / 'exp17/data')
    if smoke:
        c['data'].update(batch_size=4, eval_batch_size=16)
        c['synthetic'].update(samples=24, batch_size=8)
        c['training']['epochs'] = 2
    return c


def private_step(model, optimizer, x, y, c, sigma):
    model.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(x), y, reduction='sum')
    loss.backward()
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


def run(smoke=False):
    seed = 42
    c = configuration(smoke)
    directory = ROOT / 'exp17/results' / ('smoke' if smoke else 'formal')
    directory.mkdir(parents=True, exist_ok=False)
    device = torch.device('cuda:0')
    torch.set_num_threads(c['runtime']['threads'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    factory = TinyCNN if smoke else SimpleCNN
    model = GradSampleModule(factory().to(device), loss_reduction='sum')
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
    state = Diagnostic(c, device, factory)
    accountant = RDPAccountant()
    global_step = 0
    with (directory / 'metrics.csv').open('w') as metrics, (directory / 'training.csv').open('w') as training, (directory / 'train.log').open('w') as log:
        step_writer = epoch_writer = None
        for epoch in range(1, c['training']['epochs'] + 1):
            model.train()
            state.refresh(model, epoch)
            save_checkpoint(state, directory, epoch, global_step)
            print(f'diagnostic epoch={epoch} private_step={global_step}', flush=True)
            loss_sum, count = 0., 0
            for x, y in loader:
                row = private_step(model, optimizer, x.to(device), y.to(device), c, sigma)
                accountant.step(noise_multiplier=sigma, sample_rate=sample_rate)
                global_step += 1
                loss_sum += row['train_loss'] * len(x)
                count += len(x)
                row.update(step=global_step, epoch=epoch)
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
            message = f'identity DP-SGD epoch={epoch} steps={global_step} epsilon={epsilon:.6f}'
            print(message, flush=True)
            log.write(message + '\n')
            log.flush()
    summary = dict(seed=seed, smoke=smoke, test_accuracy=accuracy,
                   epsilon_spent=epsilon, steps=global_step, diagnostics=state.diagnostics(),
                   accounting_convention='fixed batch shuffled; drop_last; RDP batch/train_size',
                   drop_last=d['drop_last'], sample_rate=sample_rate, noise_multiplier=sigma,
                   total_steps=steps, synthetic_batches=c['synthetic']['samples']//c['synthetic']['batch_size'])
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    run(parser.parse_args().smoke)
