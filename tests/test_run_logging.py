import csv
from datetime import datetime
import re
from unittest.mock import patch
import pytest
from dp_kfac.standalone.config import load_config
from dp_kfac.standalone.run_logging import (format_run_name, prepare_run,
    format_tmux_session_name, MetricsCSVWriter, METRICS_FIELDS)
from test_standalone_config import ROOT

@pytest.mark.parametrize('algorithm,suffix', [('dp_sgd',''), ('dp_kfc','_M2560_U1_damp0.001'),
    ('dp_equil','_M2560_U1_K8_tau0.01')])
def test_exact_names(algorithm, suffix):
    c = load_config(ROOT / f'configs/standalone/mnist_{algorithm}.yaml')
    # Fix the naming fixture independently of editable experiment settings.
    c['training'].update(learning_rate=.5 if algorithm == 'dp_sgd' else .1,
                         momentum=0. if algorithm == 'dp_sgd' else .9)
    c['synthetic'].update(samples=2560, refresh_every_epochs=1)
    c['equil'].update(probes=8, tau=.01)
    learning = 'lr0.5_mom0' if algorithm == 'dp_sgd' else 'lr0.1_mom0.9'
    assert format_run_name(c, datetime(2026,9,11,17,15,0)) == (
        f'0911-171500_simple_cnn_mnist_{algorithm}_s42_ep5_bs256_{learning}_eps1_d1e-5_C1' + suffix)


def test_prepare_collision_metrics(tmp_path):
    path = ROOT / 'configs/standalone/mnist_dp_sgd.yaml'
    c = load_config(path)
    c['output']['root'] = str(tmp_path)
    now = datetime(2026,12,31,23,59,59)
    first = prepare_run(c, path, now)
    second = prepare_run(c, path, now)
    assert first.name.startswith('1231-235959_')
    assert second.name.startswith('0101-000000_')
    assert {p.name for p in first.iterdir()} == {'config.yaml','resolved_config.yaml','metrics.csv','train.log'}
    assert (first / 'config.yaml').read_bytes() == path.read_bytes()
    writer = MetricsCSVWriter(first / 'metrics.csv')
    with patch('dp_kfac.standalone.run_logging.os.fsync') as fsync:
        for epoch in range(1,4):
            writer.append(dict.fromkeys(METRICS_FIELDS, epoch))
        assert fsync.call_count == 3
    with (first / 'metrics.csv').open() as f:
        rows = list(csv.reader(f))
    assert len(rows) == 4
    assert rows[0] == list(METRICS_FIELDS)
    assert re.fullmatch('[A-Za-z0-9_-]+', format_tmux_session_name('/tmp/a b.c:你好'))


def test_prepare_cli(tmp_path):
    import subprocess
    import sys
    import yaml
    path = tmp_path / 'cpu.yaml'
    path.write_text(yaml.safe_dump({'algorithm':'dp_sgd', 'runtime':{'device':'cpu'},
                                    'output':{'root':str(tmp_path/'runs')}}))
    result = subprocess.run([sys.executable, str(ROOT/'scripts/train.py'),
        '--config',str(path),'--prepare-run'],check=True,capture_output=True,text=True)
    directory = __import__('pathlib').Path(result.stdout.strip())
    assert {p.name for p in directory.iterdir()} == {'config.yaml','resolved_config.yaml','metrics.csv','train.log'}
    assert len((directory/'metrics.csv').read_text().splitlines()) == 1
    resolved = yaml.safe_load((directory/'resolved_config.yaml').read_text())
    assert resolved['total_steps'] == 1175
    assert resolved['noise_multiplier'] is None
