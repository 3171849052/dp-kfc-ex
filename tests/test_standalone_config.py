from pathlib import Path
import pytest
import yaml
from dp_kfac.standalone.config import load_config

ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('algorithm', ['dp_sgd', 'dp_kfc', 'dp_equil'])
def test_configs(algorithm):
    c = load_config(ROOT / f'configs/standalone/mnist_{algorithm}.yaml')
    assert c['algorithm'] == algorithm
    assert c['training']['epochs'] == 5
    assert c['data']['batch_size'] == 256
    assert c['data']['root'] == 'exp1/data'
    assert c['synthetic']['samples'] == (2560 if algorithm == 'dp_sgd' else 256)
    assert c['runtime']['device'] == 'cuda'


def test_default_data_root(tmp_path):
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump({'algorithm': 'dp_sgd'}))
    assert load_config(path)['data']['root'] == 'exp1/data'

@pytest.mark.parametrize('change', [{'algorithm': 'adam'}, {'training': {'epochs': 0}},
    {'privacy': {'delta': 2}}, {'data': {'dataset': 'cifar10'}}, {'extra': 1},
    {'synthetic': {'samples': 3}}, {'equil': {'tau': float('nan')}}])
def test_invalid(tmp_path, change):
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump({'algorithm': 'dp_sgd', **change}))
    with pytest.raises(ValueError):
        load_config(path)
