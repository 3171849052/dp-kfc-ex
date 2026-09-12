import csv
import json
from unittest.mock import patch
import pytest
import torch
from torch.utils.data import TensorDataset
import yaml
from opacus import GradSampleModule
from dp_kfac.models import SimpleCNN
from dp_kfac.standalone.config import load_config
from dp_kfac.standalone.run_logging import prepare_run
from dp_kfac.standalone import trainer
from test_standalone_config import ROOT


def smoke_config(tmp_path, algorithm, optimizer='sgd'):
    c = load_config(ROOT / f"configs/standalone/mnist_{algorithm}{'_adamw' if optimizer == 'adamw' else ''}.yaml")
    c['runtime'].update(device='cpu', threads=1)
    c['training']['epochs'] = 3
    c['data'].update(batch_size=2, eval_batch_size=2)
    c['synthetic'].update(samples=4, batch_size=2)
    c['equil']['probes'] = 2
    c['output']['root'] = str(tmp_path / 'outputs')
    path = tmp_path / f'{algorithm}.yaml'
    path.write_text(yaml.safe_dump(c))
    return load_config(path), path


@pytest.mark.parametrize('algorithm', ['dp_sgd','dp_kfc','dp_equil'])
@pytest.mark.parametrize('optimizer', ['sgd', 'adamw'])
def test_smoke(tmp_path, algorithm, optimizer):
    c, path = smoke_config(tmp_path, algorithm, optimizer)
    generator = torch.Generator().manual_seed(1)
    data = TensorDataset(torch.randn(5,1,28,28,generator=generator), torch.arange(5))
    directory = prepare_run(c, path)
    original = trainer.build_preconditioner
    def isolated(*args):
        before = torch.random.get_rng_state().clone()
        state = original(*args)
        assert torch.equal(before, torch.random.get_rng_state())
        return state
    with patch.object(trainer, 'load_data', return_value=(data,data)), \
         patch.object(trainer, 'build_preconditioner', side_effect=isolated) as build, \
         patch.object(trainer, 'pink_batches', wraps=trainer.pink_batches) as pink:
        summary = trainer.train(c,directory)
    assert build.call_count == (0 if algorithm == 'dp_sgd' else 3)
    assert pink.call_count == (0 if algorithm == 'dp_sgd' else 3)
    stored = json.loads((directory / 'summary.json').read_text())
    assert summary == stored
    assert stored['optimizer'] == optimizer
    assert stored['completed_epochs'] == 3
    assert stored['global_step'] == 9  # Includes the final partial minibatch.
    assert stored['max_peak_allocated_mb'] is None
    assert stored['training_total_seconds'] > 0
    assert stored['final_epsilon'] > 0
    assert 0 <= stored['final_test_accuracy'] <= 1
    with (directory/'metrics.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    assert all(row['peak_allocated_mb_epoch'] == '' for row in rows)
    resolved = yaml.safe_load((directory/'resolved_config.yaml').read_text())
    assert resolved['sample_rate'] == 2/5
    assert resolved['total_steps'] == 9
    if algorithm == 'dp_sgd':
        assert all(float(row['precond_build_seconds']) == 0 for row in rows)
        assert all(float(row['gain_median']) == 1 for row in rows)
    else:
        assert stored['max_preconditioner_storage_mb'] > 0


@pytest.mark.parametrize('optimizer_name', ['sgd', 'adamw'])
def test_privacy_order(tmp_path, optimizer_name):
    c, _ = smoke_config(tmp_path, 'dp_kfc')
    model = GradSampleModule(SimpleCNN(), loss_reduction='sum')
    c['training']['optimizer'] = optimizer_name
    optimizer = trainer.build_optimizer(model, c['training'])
    accountant = trainer.RDPAccountant()
    events = []
    original_clip = trainer.clip_and_noise_gradients
    def precondition(model, *state):
        events.append('precondition')
        for p in model.parameters():
            p.grad_sample.mul_(2)
    def clip(*args):
        events.append('clip')
        original_clip(*args)
    original_noise = torch.randn_like
    def noise(tensor):
        events.append('noise')
        return original_noise(tensor)
    with patch.object(trainer, 'precondition_per_sample_gradients', side_effect=precondition), \
         patch.object(trainer, 'clip_and_noise_gradients', side_effect=clip), \
         patch('torch.randn_like', side_effect=noise), \
         patch.object(optimizer, 'step', side_effect=lambda: events.append('sgd')), \
         patch.object(accountant, 'step', side_effect=lambda **kw: events.append('accountant')):
        trainer.private_step(model,optimizer,accountant,torch.randn(2,1,28,28),torch.tensor([0,1]),c,({},{}),1.,.1)
    assert events[:2] == ['precondition','clip']
    assert set(events[2:-2]) == {'noise'}
    assert events[-2:] == ['sgd','accountant']


def test_failure_no_completed_summary(tmp_path):
    c,path = smoke_config(tmp_path,'dp_sgd')
    directory = prepare_run(c,path)
    with patch.object(trainer,'load_data',side_effect=RuntimeError('dataset failure')):
        with pytest.raises(RuntimeError,match='dataset failure'):
            trainer.train(c,directory)
    assert not (directory/'summary.json').exists()


def test_global_clipping_then_noise():
    # Two parameter blocks distinguish global clipping from layerwise clipping.
    model = torch.nn.Linear(1,1)
    model.weight.grad_sample = torch.tensor([[[3.]], [[0.]]])
    model.bias.grad_sample = torch.tensor([[4.], [0.]])
    with patch('torch.randn_like', side_effect=lambda t: torch.ones_like(t)):
        trainer.clip_and_noise_gradients(model, 2., 1., 2)
    factor = 1 / (5 + 1e-6)
    torch.testing.assert_close(model.weight.grad, torch.tensor([[(3*factor + 2)/2]]))
    torch.testing.assert_close(model.bias.grad, torch.tensor([(4*factor + 2)/2]))


@pytest.mark.parametrize('algorithm', ['dp_kfc', 'dp_equil'])
def test_exp6_preconditioner_regression(tmp_path, algorithm):
    # Execute the historical builder unchanged as the mathematical reference.
    import importlib.util
    spec = importlib.util.spec_from_file_location('exp6_reference', ROOT/'exp6/run_exp6.py')
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    c, _ = smoke_config(tmp_path, algorithm)
    # Match the historical reference, independent of experiment tuning.
    c['equil'].update(tau=.01)
    reference.PRECOND_STEPS = 2
    reference.PROBES = 2
    device = torch.device('cpu')
    torch.manual_seed(c['seed'])
    model = GradSampleModule(SimpleCNN(), loss_reduction='sum')
    actual = trainer.build_preconditioner(model, c, device, 1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(c['seed'] + 10001)
        if algorithm == 'dp_kfc':
            expected = reference.kfac_factors(model, 2, device)
        else:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(c['seed'] + 30001)
                probes = reference.rademacher(list(model.parameters()), 2)
            _, e = reference.fisher_statistics(model, reference.pink_batches(2, device), device, probes)
            # Keep the historical estimator, but normalization now has no hard bounds.
            values = torch.cat([v.flatten() for v in e.values()])
            raw = {p: (v + .01 * values.median()).rsqrt() for p, v in e.items()}
            gm = (sum(v.log().sum() for v in raw.values()) / values.numel()).exp()
            expected = {p: v / gm for p, v in raw.items()}
    for left, right in zip(actual if algorithm == 'dp_kfc' else (actual,),
                           expected if algorithm == 'dp_kfc' else (expected,)):
        assert left.keys() == right.keys()
        for key in left:
            torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
