"""Check the seven completed tiny runs and all numerical log fields."""
import csv
import json
import math
from pathlib import Path

root = Path(__file__).resolve().parents[1] / 'results/smoke'
rows = list(csv.DictReader((root / 'summary.csv').open()))
assert len(rows) == 7
assert {(r['mode'], float(r['q'])) for r in rows} == {
    ('hessian', 0.), *[(m, q) for q in (.25, .5) for m in ('hessian', 'fisher', 'nested')]}
for row in rows:
    assert row['seed'] == '42' and row['steps'] == '4'
for path in sorted(root.glob('*/summary.json')):
    config = json.loads((path.parent / "config.json").read_text())
    assert config["lbfgs"]["factor_damping"] == 0.1
    assert config["kfac"]["damping"] == 0.001
    assert config["actual_device"] == "cuda:0"
    summary = json.loads(path.read_text())
    assert summary['smoke'] and summary['steps'] == 4
    metrics = list(csv.DictReader((path.parent / 'metrics.csv').open()))
    epochs = list(csv.DictReader((path.parent / 'training.csv').open()))
    assert len(metrics) == 4 and len(epochs) == 2
    assert all(math.isfinite(float(v)) for row in metrics + epochs for v in row.values())
    print(path.parent.name, 'PASS: 4 finite DP steps, 2 evaluations')
print('PASS: seven-case summary, seed=42 only')
