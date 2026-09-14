"""Validate artifacts after running run_exp14d.py --smoke."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from PIL import Image


def check():
    output = Path(__file__).resolve().parents[1]/'results/smoke'
    metrics = pd.read_csv(output/'metrics.csv')
    layers = pd.read_csv(output/'whitening.csv')
    final = pd.read_csv(output/'summary.csv')
    beta = pd.read_csv(output/'beta_summary.csv')
    profile = pd.read_csv(output/'whitening_profile.csv')
    config = json.loads((output/'config.json').read_text())
    assert config['betas'] == [0., .25, .5] and config['seeds'] == [42]
    assert config['epochs'] == 1 and config['accountant_steps'] == 0
    assert len(metrics) == len(final) == len(beta) == 3
    assert set(metrics.beta) == {0., .25, .5}
    assert (metrics.seed == 42).all() and (metrics.epoch == 1).all()
    assert (metrics.probe_sample_count == 256).all()
    assert (metrics.samples == 256).all()
    assert len(layers) == 12 and len(profile) == 3003
    pd.testing.assert_frame_equal(metrics, final)
    for field in ['test_accuracy', 'test_loss', 'train_loss', 'clip_fraction',
                  'transformed_norm_p50', 'transformed_norm_p90', 'transformed_norm_p99',
                  'scale_match', 'W_global', 'top10_energy_share', 'directions_for_90pct']:
        assert np.isfinite(metrics[field]).all()
        np.testing.assert_allclose(beta[field+'_mean'], final.sort_values('beta')[field])
        assert beta[field+'_std'].isna().all()
    assert np.isfinite(layers.select_dtypes('number')).all().all()
    assert ((layers.W > 0) & (layers.W <= 1)).all()
    for b, group in layers.groupby('beta'):
        row = metrics.loc[metrics.beta == b].iloc[0]
        for local, global_key in [('W', 'W_global'), ('top10_energy_share', 'top10_energy_share'),
                                  ('directions_for_90pct', 'directions_for_90pct')]:
            np.testing.assert_allclose(row[global_key], np.average(group[local], weights=group.d))
    assert np.isfinite(profile.normalized_energy).all()
    for _, group in profile.groupby('beta'):
        assert group.direction_percentile.iloc[0] == 0
        assert group.direction_percentile.iloc[-1] == 100
        assert (np.diff(group.normalized_energy) <= 0).all()
    for name in ['isotropy_vs_beta', 'accuracy_vs_isotropy', 'clip_fraction_vs_isotropy',
                 'energy_concentration_vs_beta', 'ranked_energy_profile']:
        with Image.open(output/(name+'.png')) as image:
            image.verify()
    print('Smoke artifacts passed: 3 runs, all CSVs and 5 PNGs.')


if __name__ == '__main__':
    check()
