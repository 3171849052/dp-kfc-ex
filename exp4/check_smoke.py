"""Core scientific invariants for the single tiny end-to-end run."""
from pathlib import Path
import numpy as np
import pandas as pd
from run_exp4 import METHODS, LAYERS, interval_ends

output = Path(__file__).resolve().parent / 'results/smoke'
m = pd.read_csv(output / 'metrics.csv')
c = pd.read_csv(output / 'controller_metrics.csv')
assert interval_ends(234) == [47, 94, 141, 188]
assert len(m) == 16 and len(c) == 64
assert set(m.method) == set(c.method) == set(METHODS)
assert set(m.seed) == set(c.seed) == {42}
assert (m.base_reset_error == 0).all()
for method in METHODS:
    assert (m[m.method == method].privacy_valid == True).all()
    part = c[c.method == method]
    assert set(part.feedback_type) == {'wiener_filtered_grad'}
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

w = pd.read_csv(output / 'wiener_metrics.csv')
assert len(w) == 80 and set(w.method) == set(METHODS)
fields = ['tau2', 'alpha', 'alpha_over_tau2', 'wiener_gain_mean',
          'wiener_gain_median', 'wiener_gain_p10', 'wiener_gain_p90',
          'cos_raw', 'cos_filtered', 'NSR_raw', 'NSR_filtered']
assert np.isfinite(w[fields].to_numpy()).all()
assert (w.tau2 > 0).all()
assert w.first_batch_identity.all()
assert (w.fitted_batches == w.interval_steps - 1).all()
for _, part in w.groupby(['method', 'seed', 'epoch', 'layer']):
    assert part.interval.tolist() == [1, 2, 3, 4, 5]
    assert part.interval_steps.tolist() == [2, 2, 2, 2, 1]
for col in [k for k in w if k.startswith('wiener_gain_')]:
    assert w[col].between(0, 1).all()
assert (c.correction_A_norm > 0).all()
assert (c[c.method == 'Wiener-Residual-AG'].correction_G_norm > 0).all()
assert (c[c.method == 'Wiener-Residual-AOnly'].correction_G_norm == 0).all()
print('PASS: 2 methods × 2 epochs; 16 metrics, 64 controller, 80 Wiener rows; '
      'schedule, identity warmup, interval resets, A/G updates, privacy and whitening gains. '
      'Runtime smoke assertions checked unchanged optimizer gradients and filtered controller EMA.')
