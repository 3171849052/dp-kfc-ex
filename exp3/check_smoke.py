"""Core scientific invariants for the single tiny end-to-end run."""
from pathlib import Path
import numpy as np
import pandas as pd
from run_exp3 import METHODS, LAYERS, interval_ends

output = Path(__file__).resolve().parent / 'results/smoke'
m = pd.read_csv(output / 'metrics.csv')
c = pd.read_csv(output / 'controller_metrics.csv')
assert interval_ends(234) == [47, 94, 141, 188]
assert len(m) == 16 and len(c) == 64
assert set(m.method) == set(c.method) == set(METHODS)
assert set(m.seed) == set(c.seed) == {42}
assert (m.base_reset_error == 0).all()
for method in METHODS:
    assert (m[m.method == method].privacy_valid == ('Noisy' in method)).all()
    part = c[c.method == method]
    assert set(part.feedback_type) == {'grad' if 'Noisy' in method else 'summed_grad'}
    for layer in LAYERS:
        p = part[part.layer == layer]
        assert p.global_update_index.tolist() == list(range(1, 9))
        assert m[(m.method == method) & (m.layer == layer)].epoch.tolist() == [1, 2]
        for epoch in (1, 2):
            e = p[p.epoch == epoch]
            assert e.interval.tolist() == [1, 2, 3, 4]
            assert e.interval_steps.tolist() == [2, 2, 2, 2]
            assert e.update_after_step.tolist() == [2, 4, 6, 8]
            assert e.applies_from_step.tolist() == [3, 5, 7, 9]
fields = [f'{prefix}_W_{side}' for prefix in ('base', 'corrected') for side in ('A', 'G', 'kron')]
fields += [f'gain_{side}' for side in ('A', 'G', 'kron')]
assert np.isfinite(m[fields].to_numpy()).all()
for side in ('A', 'G', 'kron'):
    assert np.allclose(m[f'gain_{side}'],
                       (m[f'base_W_{side}']-m[f'corrected_W_{side}']) / m[f'base_W_{side}'])
for prefix in ('base', 'corrected'):
    values = m[f'{prefix}_log_kappa_kron'].to_numpy()
    assert (np.isfinite(values) | np.isposinf(values)).all()
print('PASS: two methods × two epochs; 16 metric rows, 64 controller rows; reset, schedule, feedback source, privacy labels and whitening gains')
