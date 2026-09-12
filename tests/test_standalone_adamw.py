import pytest
import torch
import yaml
from dp_kfac.standalone.config import load_config
from dp_kfac.standalone.trainer import build_optimizer
from dp_kfac.standalone.run_logging import format_run_name
from test_standalone_config import ROOT


@pytest.mark.parametrize('algorithm', ['dp_sgd', 'dp_kfc', 'dp_equil'])
def test_adamw_config_and_update(algorithm):
    c = load_config(ROOT / f'configs/standalone/mnist_{algorithm}_adamw.yaml')
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(2.)
    optimizer = build_optimizer(model, c['training'])
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.defaults['betas'] == (.9, .999)
    model.weight.grad = torch.zeros_like(model.weight)
    optimizer.step()
    # Decoupled decay applies even with a zero sanitized gradient.
    torch.testing.assert_close(model.weight, torch.tensor([[2 * (1 - .001 * .01)]]))
    assert optimizer.state[model.weight]['step'].item() == 1
    assert '_adamw_' in format_run_name(c)
    assert '_mom' not in format_run_name(c)


@pytest.mark.parametrize('training', [
    {'optimizer': 'adam'}, {'betas': [.9]}, {'betas': [.9, 1.]},
    {'betas': [True, .999]}, {'betas': [float('nan'), .999]},
    {'eps': 0}, {'eps': float('inf')}, {'weight_decay': -1},
])
def test_invalid_adamw(tmp_path, training):
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump({'algorithm': 'dp_sgd',
                                   'training': {'optimizer': 'adamw', **training}}))
    with pytest.raises(ValueError):
        load_config(path)
