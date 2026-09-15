"""Check the six completed tiny runs and all numerical log fields."""
import csv
import json
import math
from pathlib import Path

root = Path(__file__).resolve().parents[1] / 'results/smoke'
rows = list(csv.DictReader((root / 'summary.csv').open()))
assert len(rows) == 6
assert {(r['mode'], float(r['q'])) for r in rows} == {
    *[(m, q) for q in (.25, .5) for m in ('fisher', 'nested_ag', 'nested_a')]}
for row in rows:
    assert row['seed'] == '42' and row['steps'] == '4'
paths = sorted(root.glob('*/summary.json'))
assert len(paths) == 6
for path in paths:
    config = json.loads((path.parent / "config.json").read_text())
    assert config["lbfgs"]["factor_damping"] == 0.1
    assert config["kfac"]["damping"] == 0.001
    assert config["actual_device"] == "cuda:0"
    summary = json.loads(path.read_text())
    assert summary['smoke'] and summary['steps'] == 4
    assert all(math.isfinite(v) for v in summary.values() if isinstance(v, (int, float)))
    assert all(math.isfinite(v) for v in summary['diagnostics'].values())
    metrics = list(csv.DictReader((path.parent / 'metrics.csv').open()))
    epochs = list(csv.DictReader((path.parent / 'training.csv').open()))
    assert len(metrics) == 4 and len(epochs) == 2
    assert {'hessian_stage_norm_mean', 'final_preconditioned_norm_mean',
            'preclip_norm_mean', 'preclip_norm_p50', 'preclip_norm_p90',
            'preclip_norm_p99', 'clip_fraction', 'update_norm', 'train_loss'} <= metrics[0].keys()
    assert {'test_loss', 'test_accuracy', 'epsilon_spent'} <= epochs[0].keys()
    assert {f'{layer}/{side}_damped_condition' for layer in ('conv1', 'conv2', 'fc1', 'fc2')
            for side in ('A', 'G')} <= metrics[0].keys()
    assert all(math.isfinite(float(v)) for row in metrics + epochs for v in row.values())
    print(path.parent.name, 'PASS: 4 finite DP steps, 2 evaluations')
print('PASS: six-case summary, seed=42 only')
