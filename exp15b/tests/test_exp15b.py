import math
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
sys.dont_write_bytecode = True
import pytest
import torch
from exp15b.config import configuration, LAMBDAS, SEEDS
from exp15b import run_exp15b as runner
from exp15.preconditioner import SyntheticKLBFGS, InverseLBFGS
from exp15.run_exp15 import private_step


@pytest.mark.parametrize('damping', LAMBDAS)
def test_protocol_and_mapping(damping):
    configuration(damping, 42, True)
    for seed in SEEDS:
        c = configuration(damping, seed)
        assert c['p'] == .25
        assert c['lbfgs'] == dict(memory=100, initial_scale=1.,
            factor_damping=math.sqrt(damping), activation_decay=.9,
            pair_decay=.9, synthetic_lookahead_lr=.5)
        assert c['training'] == dict(epochs=5, optimizer='sgd',
            learning_rate=.5, momentum=0., weight_decay=0.)
        assert c['privacy'] == dict(epsilon=1., delta=1e-5, max_grad_norm=1., accountant='rdp')
        assert c['data']['batch_size'] == c['data']['eval_batch_size'] == 256
        assert c['data']['shuffle'] and c['data']['drop_last']
        assert c['training']['epochs'] * (60000 // c['data']['batch_size']) == 1170
        assert c['synthetic']['samples'] // c['synthetic']['batch_size'] == 10
        assert c['synthetic']['refresh_every_epochs'] == 1
        factor = InverseLBFGS(torch.device('cpu'), c['lbfgs'])
        assert factor.params['Kron_BFGS_H_epsilon'] == math.sqrt(damping)
    assert SEEDS == (42, 7)
    assert len(LAMBDAS) * len(SEEDS) == 10
    assert runner.SyntheticKLBFGS is SyntheticKLBFGS
    assert runner.private_step is private_step


def test_two_lambda_smoke(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'EXP_ROOT', tmp_path)
    for damping in (LAMBDAS[0], LAMBDAS[-1]):
        result = runner.run(damping, 42, True)
        assert result['total_steps'] == result['steps'] == 4
        assert result['factor_damping'] == math.sqrt(damping)
        directory = tmp_path / 'results/smoke' / f'lambda{damping:g}_seed42'
        for filename in ('config.json', 'summary.json', 'metrics.csv', 'training.csv',
                         'train.log', 'factors_epoch1.json', 'factors_epoch2.json'):
            assert (directory / filename).stat().st_size > 0
    from exp15b import analyze
    monkeypatch.setattr(analyze, 'ROOT', tmp_path)
    analyze.analyze(True)
    assert len(list((tmp_path / 'results/smoke').glob('*.png'))) == 4
    assert (tmp_path / 'results/smoke/lambda_summary.csv').exists()

from opacus import GradSampleModule
from dp_kfac.models import SimpleCNN
from dp_kfac.standalone.trainer import build_optimizer
from dp_kfac.privacy import _compute_clip_factors

@pytest.mark.parametrize('damping', [LAMBDAS[0], LAMBDAS[-1]])
def test_synthetic_batch_independence_and_private_update(damping):
    from exp15.verification import CheckedSyntheticKLBFGS
    torch.set_num_threads(4)
    torch.manual_seed(9)
    c = configuration(damping, 42, smoke=True)
    model = GradSampleModule(SimpleCNN(), loss_reduction='sum')
    state = CheckedSyntheticKLBFGS(c, torch.device('cpu'))
    state.refresh(model, 1)  # Explicit before-forward reset and pair retention assertions.
    for factors in state.factors.values():
        for factor in factors:
            assert type(factor) is InverseLBFGS
            assert factor.params['Kron_BFGS_H_epsilon'] == math.sqrt(damping)
    assert state.config['lbfgs']['factor_damping'] == math.sqrt(damping)
    assert len(state.before_pair_counts) == 3
    assert state.before_pair_counts[0] == 0
    assert state.before_pair_counts[2] > state.before_pair_counts[1]

    # Independently aggregate all preconditioned CNN parameters after refresh,
    # then draw identical noise and compare the actual private SGD update.
    x, y = torch.randn(2, 1, 28, 28), torch.tensor([2, 3])
    torch.nn.functional.cross_entropy(model(x), y, reduction='sum').backward()
    state.apply(model, c['p'])
    params = list(model.parameters())
    norms = sum(p.grad_sample.flatten(1).square().sum(1) for p in params).sqrt()
    clips = _compute_clip_factors(norms.square(), 1.)
    grads = [(p.grad_sample.flatten(1) * clips[:, None]).sum(0) for p in params]
    torch.manual_seed(99)
    expected = [p.detach() - .5 * ((g + 1.3*torch.randn_like(g))/len(x)).view_as(p)
                for p, g in zip(params, grads)]
    torch.manual_seed(99)
    result = private_step(model, build_optimizer(model, c['training']), x, y, c, state, 1.3)
    # Explicit sum-of-squares vs the utility's float32 norm reduction.
    torch.testing.assert_close(torch.tensor(result['preclip_norm_mean']), norms.mean(),
                               rtol=1e-5, atol=1e-6)
    for actual, reference in zip(params, expected):
        torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)
