"""Single-seed descriptive power sweep, strictly separated smoke outputs."""
import os
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
os.environ['MPLCONFIGDIR'] = str(HERE/'.cache/matplotlib')
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from exp20.config import POWERS, EPOCHS


def analyze(output, smoke):
    frames, layers, configs = [], [], []
    for p in POWERS:
        directory = output/'runs'/f'{p}_42'
        frame = pd.read_csv(directory/'metrics.csv')
        assert frame.epoch.tolist() == list(range(1, (1 if smoke else EPOCHS)+1))
        assert frame.p.eq(p).all() and frame.seed.eq(42).all()
        np.testing.assert_allclose(frame.scale_match**2*frame.m_p, frame.m_reference, rtol=1e-12)
        assert frame.builder_vjp_calls.eq(0).all() and frame.builder_reverse_vectors.eq(0).all()
        assert frame.batches.eq(1 if smoke else 234).all()
        frames.append(frame)
        layers.append(pd.read_csv(directory/'layer_norm_diagnostics.csv'))
        configs.append(json.loads((directory/'config.json').read_text()))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(output/'metrics.csv', index=False)
    pd.concat(layers, ignore_index=True).to_csv(output/'layer_norm_diagnostics.csv', index=False)
    spectral = [c for c in df if c.startswith(('operator_gain_', 'transformed_eig_')) or c in ('transformed_condition_proxy', 'm_p', 'm_reference', 'scale_match')]
    df[['p', 'seed', 'epoch']+spectral].to_csv(output/'spectrum_summary.csv', index=False)
    rows = []
    for p, g in df.groupby('p', sort=True):
        rows.append(dict(p=p, seed=42, final_accuracy=g.test_accuracy.iloc[-1],
            accuracy_auc=np.trapezoid(g.test_accuracy, g.epoch), clip_fraction=g.clip_fraction.mean(),
            norm_p99=g.transformed_norm_p99.mean(), scale_match=g.scale_match.mean(),
            transformed_condition_proxy=g.transformed_condition_proxy.mean()))
    summary = pd.DataFrame(rows)
    ref = summary[summary.p.eq(.5)].iloc[0]
    for col in ('final_accuracy', 'accuracy_auc', 'clip_fraction', 'norm_p99'):
        summary['delta_'+col] = summary[col]-ref[col]
    summary.to_csv(output/'summary.csv', index=False)
    summary.to_csv(output/'power_summary.csv', index=False)
    (output/'config.json').write_text(json.dumps(dict(smoke=smoke, runs=configs,
        accuracy_auc='Trapezoid integral over observed epochs 1..5; smoke one epoch = 0',
        epoch_reduction='clip_fraction, norm_p99, scale and spectral spread: arithmetic mean over epochs',
        spectrum='Unweighted concatenation of augmented activation eigenvalues over layers; condition proxy = p90/p10 (infinity if p10=0)',
        comparison='Single seed=42 descriptive comparison; reference p=0.5; no bootstrap or significance claims'), indent=2)+'\n')
    charts = [('final_accuracy_vs_p', 'p', 'final_accuracy'), ('accuracy_auc_vs_p', 'p', 'accuracy_auc'),
        ('clip_fraction_vs_p', 'p', 'clip_fraction'), ('norm_p99_vs_p', 'p', 'norm_p99'),
        ('accuracy_vs_clip_fraction', 'clip_fraction', 'final_accuracy'),
        ('accuracy_vs_norm_p99', 'norm_p99', 'final_accuracy'), ('scale_factor_vs_p', 'p', 'scale_match'),
        ('transformed_spectral_spread_vs_p', 'p', 'transformed_condition_proxy')]
    for filename, x, y in charts:
        plt.figure(figsize=(7, 4))
        plt.plot(summary[x], summary[y], 'o-')
        if x != 'p':
            for _, row in summary.iterrows():
                plt.annotate(f'p={row.p:g}', (row[x], row[y]))
        plt.xlabel(x); plt.ylabel(y)
        plt.title('Smoke only' if smoke else 'Seed 42')
        plt.tight_layout(); plt.savefig(output/f'{filename}.png', dpi=160); plt.close()
    for p, g in df.groupby('p'):
        plt.plot(g.epoch, g.test_accuracy, 'o-', label=f'p={p:g}')
    plt.xlabel('Epoch'); plt.ylabel('Test accuracy'); plt.legend()
    plt.tight_layout(); plt.savefig(output/'accuracy_vs_epoch_by_p.png', dpi=160); plt.close()
    best = summary.loc[summary.final_accuracy.eq(summary.final_accuracy.max()), 'p'].tolist()
    aucbest = summary.loc[summary.accuracy_auc.eq(summary.accuracy_auc.max()), 'p'].tolist()
    identity = summary[summary.p.eq(0)].iloc[0]
    low = summary[summary.p.eq(.25)].iloc[0]
    def direction(values):
        changes = np.diff(values)
        if np.all(changes == 0):
            return 'constant'
        if np.all(changes >= 0):
            return 'nondecreasing'
        if np.all(changes <= 0):
            return 'nonincreasing'
        return 'nonmonotonic'
    positive_best = summary[summary.p.gt(0)].final_accuracy.max()-identity.final_accuracy
    neighborhood = summary[summary.p.between(.25, .5)]
    lines = ['# Exp20', '', 'SMOKE ONLY: one zero-noise batch. These observations do not answer formal utility questions.' if smoke else
        'Single seed=42 descriptive comparison. No bootstrap confidence intervals or significance claims.', '',
        '## 1. Best final accuracy', f'p={best}; accuracy={summary.final_accuracy.max():.6f} (all ties listed).',
        '## 2. Best accuracy AUC', f'p={aucbest}; AUC={summary.accuracy_auc.max():.6f}. AUC integrates observed epochs; smoke AUC is zero.',
        '## 3. Clipping and norm tails', f'Clipping fraction is {direction(summary.clip_fraction)}; p99 norm is {direction(summary.norm_p99)} as p increases. Ordered p / mean epoch clip fraction / mean epoch p99 norm:']
    lines += [f'- {r.p:g}: {r.clip_fraction:.6f} / {r.norm_p99:.6f}' for r in summary.itertuples()]
    lines += ['## 4. Is anisotropic A geometry useful?',
        f'Scale-matched identity accuracy={identity.final_accuracy:.6f}. Positive-p accuracy minus identity: '+
        ', '.join(f'{r.p:g}: {r.final_accuracy-identity.final_accuracy:+.6f}' for r in summary.itertuples() if r.p > 0)+'.',
        ('The best anisotropic operator improves on scale-matched identity in this trajectory, supporting geometry utility for this seed.' if positive_best > 0 else 'The anisotropic operators do not improve final accuracy over scale-matched identity in this trajectory; this run does not support a utility benefit.')+' Scale matching controls the synthetic global RMS, not every private norm or optimization effect.',
        '## 5. Above p=0.5: over-preconditioning?',
        'Differences versus p=0.5 (accuracy / p99 / clip fraction):']
    lines += [f'- {r.p:g}: {r.delta_final_accuracy:+.6f} / {r.delta_norm_p99:+.6f} / {r.delta_clip_fraction:+.6f}; '+
        ('higher tail and lower accuracy are consistent with over-preconditioning.' if r.delta_norm_p99 > 0 and r.delta_final_accuracy < 0 else 'no joint higher-tail/lower-accuracy pattern.') for r in summary.itertuples() if r.p > .5]
    lines += ['## 6. Interpreting the Exp19 p=0.25 / p=0.5 proximity',
        f'Here p=0.25 minus p=0.5: final accuracy {low.delta_final_accuracy:+.6f}; AUC {low.delta_accuracy_auc:+.6f}.',
        'Dense neighboring results (p / final accuracy / AUC): '+', '.join(f'{r.p:g} / {r.final_accuracy:.6f} / {r.accuracy_auc:.6f}' for r in summary.itertuples() if .125 <= r.p <= .625)+'.',
        f'Within p=0.25..0.5, accuracy is {direction(neighborhood.final_accuracy)}, with range {neighborhood.final_accuracy.max()-neighborhood.final_accuracy.min():.6f}; the intermediate p=0.375 differs from p=0.5 by {neighborhood[neighborhood.p.eq(.375)].final_accuracy.iloc[0]-ref.final_accuracy:+.6f}. Endpoint proximity alone therefore does not establish a flat optimum. This single seed cannot establish that the Exp19 observation generalizes.', '',
        'Smoke cannot establish any of these geometry/utility interpretations.' if smoke else 'Interpretations are restricted to this single paired trajectory.', '',
        'Scale uses each run’s current synthetic A every epoch. p=0 is scalar identity, not ordinary DP-SGD. Biases use augmented activations. Spectral quantiles weight each activation eigenvalue once; RMS moments additionally weight layer output dimension.',
        'Timing: algorithm = build + private training; CPU norm diagnostics and evaluation are separate. CUDA memory peaks cover build/train. Layer diagnostics describe fractions of total transformed squared norm.',
        'RDP accounting follows Exp19’s shuffled fixed-batch convention. Private clipping diagnostics are unnoised research measurements.']
    (output/'report.md').write_text('\n\n'.join(lines)+'\n')
    print(f'All 8 powers validated; scale matching passed. Analysis saved: {output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    analyze(HERE/'results'/'smoke' if args.smoke else HERE/'results', args.smoke)
