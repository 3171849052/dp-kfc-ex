"""Historical references stay separate from newly trained rows."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]


def baselines():
    records = []
    for name, source, beta in [('full', 'exp14b/results/summary.csv', .25),
                               ('identity', 'exp14/results/summary.csv', 0.)]:
        df = pd.read_csv(ROOT / source)
        row = df[(df.seed == 42) & (df.beta == beta)].iloc[0]
        records.append(dict(method=name, source=source, seed=42, beta=beta,
                            test_accuracy=row.test_accuracy, total_algorithm_seconds=row.total_algorithm_seconds))
    return pd.DataFrame(records)


def frontier(df, cost):
    return [not any((df[cost] <= r[cost]) & (df.test_accuracy >= r.test_accuracy) &
                   ((df[cost] < r[cost]) | (df.test_accuracy > r.test_accuracy))) for _, r in df.iterrows()]


def save(rows, diagnostics, output, smoke=False):
    df = pd.DataFrame(rows)
    df.to_csv(output / 'metrics.csv', index=False)
    pd.DataFrame(diagnostics, columns=['method','seed','epoch','layer','side','rank','dimension','tau',
                                      'captured_trace_fraction']).to_csv(output / 'lowrank_diagnostics.csv', index=False)
    summary = df.groupby('method', sort=False).tail(1).set_index('method')
    for col in ['preconditioner_build_seconds', 'private_train_seconds', 'evaluation_seconds', 'algorithm_epoch_seconds']:
        summary['total_'+col.replace('epoch_', '')] = df.groupby('method')[col].sum()
    summary['peak_cuda_allocated_bytes'] = df.groupby('method').total_peak_cuda_allocated_bytes.max()
    for col in ['builder_forward_calls','builder_vjp_calls','builder_reverse_vectors']:
        summary['total_'+col] = df.groupby('method')[col].sum()
    summary.to_csv(output / 'summary.csv')
    summary.to_csv(output / 'method_summary.csv')
    base = baselines()
    base.to_csv(output / 'baseline_references.csv', index=False)
    full, identity = base.test_accuracy.tolist()
    pareto = summary.reset_index()
    pareto['utility_retention'] = (pareto.test_accuracy-identity)/(full-identity)
    pareto['memory_frontier'] = frontier(pareto, 'operator_state_bytes')
    pareto['runtime_frontier'] = frontier(pareto, 'total_algorithm_seconds')
    pareto.to_csv(output / 'pareto_summary.csv', index=False)
    for cost, label in [('operator_state_bytes','Operator state bytes'),('total_algorithm_seconds','Algorithm seconds')]:
        fig, ax = plt.subplots(figsize=(9,5))
        ax.scatter(pareto[cost], pareto.test_accuracy)
        for _, r in pareto.iterrows():
            ax.annotate(r.method, (r[cost], r.test_accuracy), fontsize=8)
        for _, r in base.iterrows():
            ax.axhline(r.test_accuracy, linestyle='--', label=f'{r.method} historical seed=42')
        ax.set(xlabel=label, ylabel='Test accuracy', title='SMOKE — not utility evidence' if smoke else 'Lightweight DP-KFC')
        ax.legend()
        fig.tight_layout(); fig.savefig(output / f'accuracy_vs_{cost}.png'); plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(13,4))
    summary[['clip_fraction','mean_clip_factor']].plot.bar(ax=axes[0])
    summary[['transformed_norm_p50','transformed_norm_p90','transformed_norm_p99']].plot.bar(ax=axes[1], logy=True)
    fig.tight_layout(); fig.savefig(output / 'clipping_norms.png'); plt.close(fig)
    lines = ['# Pareto summary', '', 'SMOKE ONLY: no scientific conclusions.' if smoke else
             'Single seed=42; differences are descriptive, without uncertainty estimates.', '',
             'Historical references (read only; never inserted into training metrics):']
    lines += [f'- {r.method}: {r.test_accuracy:.4f}, seed=42, beta={r.beta}, `{r.source}`.' for _,r in base.iterrows()]
    lines += ['', 'Identity uses Exp14 beta=0: Exp14b beta=0 is globally rescaled and is not identity.',
              'Utility retention = (accuracy − identity) / (full − identity).',
              'Pareto flags compare newly trained methods. Historical runtime has separate-run hardware/timing caveats.',
              'Historical operator bytes are not measured in those CSVs; baseline horizontal lines compare utility only.', '',
              pareto[['method','test_accuracy','utility_retention','operator_state_bytes','total_algorithm_seconds',
                      'memory_frontier','runtime_frontier']].to_string(index=False), '']
    if not smoke and len(pareto) == 10:
        p = pareto.set_index('method')
        lines += [f"Diagonal retains {p.loc['diag','utility_retention']:.1%} of the historical gain.",
          f"A-only − C-only accuracy: {p.loc['a_only','test_accuracy']-p.loc['c_only','test_accuracy']:+.4f}; "
          f"fullA/diagC − diagA/fullC: {p.loc['fullA_diagC','test_accuracy']-p.loc['diagA_fullC','test_accuracy']:+.4f}.",
          f"Rank 4→8 gain: {p.loc['rank8','test_accuracy']-p.loc['rank4','test_accuracy']:+.4f}; "
          f"8→16 gain: {p.loc['rank16','test_accuracy']-p.loc['rank8','test_accuracy']:+.4f}. "
          'A smaller latter gain suggests saturation; one seed cannot establish it.',
          *[f"{m} − historical full: {p.loc[m,'test_accuracy']-full:+.4f}; retention {p.loc[m,'utility_retention']:.1%}." for m in ['refresh2','frozen']]]
    (output / 'pareto_summary.md').write_text('\n'.join(lines)+'\n')
