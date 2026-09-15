from pathlib import Path
import json
import numpy as np
import pandas as pd
root = Path(__file__).resolve().parents[1]/'results/smoke'
summary = json.loads((root/'summary.json').read_text())
config = json.loads((root/'config.json').read_text())
assert summary['steps'] == 4 and summary['synthetic_batches'] == 3
assert config['actual_device'] == 'cuda:0'
assert config['lbfgs']['factor_damping'] == .1 and config['kfac']['damping'] == .001
for name, count in [('factors',16),('operator_metrics',32),('transformed_conditions',32),('training',2)]:
    frame = pd.read_csv(root/f'{name}.csv')
    assert len(frame) == count
    assert np.isfinite(frame.select_dtypes('number')).all().all()
factors = pd.read_csv(root/'factors.csv')
assert factors.groupby('epoch').private_step.first().tolist() == [0,2]
assert np.isfinite(pd.read_csv(root/'generalized_spectra.csv').select_dtypes('number')).all().all()
assert len(list((root/'matrices').glob('epoch*/*.pt'))) == 8
print('CUDA smoke PASS: 2 epochs, 4 private steps, 2 diagnostics, all CSV metrics finite; five checkpoints covered by pytest.')
