"""Standalone dispatch must preserve the tested Exp25 protocol and numerics."""
import copy
import csv
from pathlib import Path
import pytest
import torch
import yaml
from opacus.accountants import RDPAccountant
from dp_kfac.standalone.config import BK_GD_ALGORITHMS, load_config
from dp_kfac.standalone.exp25_adapter import protocol
from dp_kfac.standalone.run_logging import prepare_run, format_run_name
from dp_kfac.standalone.trainer import train
from exp25 import config as original
from exp25.data import rng
from exp25.geometry import build
from exp25.methods import aggregate, update
from exp25.test_exp25 import toy, reference

ROOT = Path(__file__).resolve().parents[1]


def config_path(method):
    return ROOT / 'configs/standalone' / f'crossvit_cifar100_{method}.yaml'


@pytest.mark.parametrize('method,kind', list(zip(BK_GD_ALGORITHMS, ('identity', 'a_only', 'full'))))
def test_protocol_and_math(method, kind):
    c = load_config(config_path(method))
    p = protocol(c)
    assert c['seed'] == 42
    assert p.condition(method) == (kind, 'none' if kind == 'identity' else 'pink')
    for key in ('EPOCHS', 'BATCH_SIZE', 'TRAIN_SAMPLES', 'STEPS_PER_EPOCH', 'TOTAL_STEPS',
                'SAMPLE_RATE', 'EPSILON', 'DELTA', 'CLIP', 'LR', 'DAMPING', 'A_POWER'):
        assert getattr(p, key) == getattr(original, key)
    assert p.noise_multiplier() == original.noise_multiplier()
    raw = yaml.safe_load(config_path(method).read_text())
    assert not {'sample_rate', 'steps_per_epoch', 'total_steps', 'noise_multiplier'} & raw.keys()
    name = format_run_name(c)
    for token in ('crossvit_tiny_240', 'cifar100', method, 's42', 'ep5', 'bs256',
                  'lr0.001', 'eps8', 'd1e-5', 'C1'):
        assert token in name
    if kind != 'identity':
        assert 'damp0.001' in name
    if kind == 'a_only':
        assert 'p0.4' in name
    model, x, y = toy()
    ref = copy.deepcopy(model)
    old_op, _ = build(ref, x, y, kind)
    op, _ = build(model, x, y, kind, a_power=p.A_POWER, damping=p.DAMPING)
    old = aggregate(ref, x, y, old_op)
    new = aggregate(model, x, y, op, clip=p.CLIP)
    norms, factors, summed = reference(ref, x, y, old_op)
    for actual, expected in zip(new[1:4], (summed, norms, factors)):
        torch.testing.assert_close(actual, expected)
    for actual, expected in zip(new[:4], old[:4]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    acc, refacc = RDPAccountant(), RDPAccountant()
    update(model, torch.optim.Adam(model.parameters(), lr=p.LR), new[1], .7,
           len(y), rng(42, 'dp_noise'), acc, clip=p.CLIP, sample_rate=p.SAMPLE_RATE)
    update(ref, torch.optim.Adam(ref.parameters(), lr=original.LR), old[1], .7,
           len(y), rng(42, 'dp_noise'), refacc)
    for actual, expected in zip(model.parameters(), ref.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert acc.history == refacc.history


@pytest.mark.parametrize('method', BK_GD_ALGORITHMS)
def test_dispatch_and_outputs(method, monkeypatch):
    import dp_kfac.standalone.exp25_adapter as adapter
    c = load_config(config_path(method))
    directory = prepare_run(c, config_path(method))
    assert directory.parent == ROOT / 'outputs'
    seen = []

    def fake_run(algorithm, seed, smoke, device, *, protocol, on_resolved, on_epoch, threads, data_root):
        seen.append((algorithm, seed, smoke))
        on_resolved(.7, torch.device('cpu'), 50000, 256)
        on_epoch(dict(epoch=1, accountant_steps=1, train_loss=1., test_loss=1.,
            test_accuracy=.1, best_accuracy=.1, epsilon_spent=1., noise_multiplier=.7,
            sample_rate=protocol.SAMPLE_RATE, clip_fraction=.5, mean_clip_factor=.5,
            geometry_build_time=.1, private_training_time=.1, epoch_total_seconds=.3,
            cuda_peak_memory=0))

    monkeypatch.setattr(adapter, 'run', fake_run)
    result = train(c, directory, smoke=True)
    assert seen == [(method, 42, True)]
    assert result['status'] == 'completed'
    resolved = yaml.safe_load((directory / 'resolved_config.yaml').read_text())
    assert resolved['total_steps'] == 975 and resolved['steps_per_epoch'] == 195
    assert resolved['noise_multiplier'] == .7
    assert len(list(csv.DictReader((directory / 'metrics.csv').open()))) == 1
    assert all((directory / f).exists() for f in ('config.yaml', 'train.log', 'summary.json'))
    with pytest.raises(ValueError, match='already been used'):
        train(c, directory)
    # Remove only the test's own temporary prepared run.
    for path in directory.iterdir():
        path.unlink()
    directory.rmdir()


@pytest.mark.parametrize('section,key,value', [
    ('output', 'root', 'exp25/results'), ('synthetic', 'samples', 512),
    ('model', 'pretrained', False), ('training', 'optimizer', 'sgd'),
    ('data', 'normalization', 'cifar100'), ('synthetic', 'refresh_every_epochs', 2)])
def test_invalid_protocol(tmp_path, section, key, value):
    raw = yaml.safe_load(config_path(BK_GD_ALGORITHMS[0]).read_text())
    raw[section][key] = value
    path = tmp_path / 'invalid.yaml'
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError):
        load_config(path)
