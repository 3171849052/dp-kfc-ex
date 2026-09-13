"""Explicitly opt into curvature diagnostics for every recorded checkpoint."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import argparse
import json
import subprocess
import pandas as pd


def summarize_checkpoint(directory, checkpoint_row):
    errors = pd.read_csv(directory/'curvature_error.csv')
    whitening = pd.read_csv(directory/'whitening.csv')
    budget = pd.read_csv(directory/'compute_budget.csv')
    keys = ['estimator', 'layer', 'label_seed']
    out = errors[keys+['kron_relative_error', 'C_relative_error_vs_KFLR']].merge(
        whitening[keys+['floored_condition_number']], on=keys, validate='one_to_one')
    out = out.merge(budget[['estimator', 'label_seed', 'build_median_seconds']], on=['estimator', 'label_seed'], validate='many_to_one')
    out = out.groupby(['estimator', 'layer']).mean(numeric_only=True).drop(columns='label_seed').reset_index()
    out = out.rename(columns={'kron_relative_error': 'kron_relative_error_vs_private_oracle'})
    out.insert(0, 'checkpoint', Path(checkpoint_row.checkpoint_path).stem)
    for key in ['step', 'epoch']:
        out[key] = getattr(checkpoint_row, key)
    state = json.loads((directory/'state_metrics.json').read_text())
    for key in ['mean_max_probability', 'mean_prediction_entropy', 'normalized_entropy', 'mean_kl_to_uniform']:
        out['mnist_'+key] = getattr(checkpoint_row, key)
        out['synthetic_'+key] = state['synthetic_'+key]
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trajectory', type=Path, default=ROOT/'exp12/checkpoints/trajectory.csv')
    p.add_argument('--output', type=Path, default=ROOT/'exp12/results/trajectory')
    args, diagnostic_args = p.parse_known_args(argv)
    assert '--checkpoint' not in diagnostic_args
    trajectory = pd.read_csv(args.trajectory)
    assert not trajectory.duplicated(['seed', 'step']).any(), 'Trajectory (seed, step) must be unique'
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for row in trajectory.itertuples(index=False):
        checkpoint = ROOT/row.checkpoint_path
        directory = args.output/checkpoint.stem
        subprocess.run([sys.executable, str(ROOT/'exp12/run_budget_sweep.py'),
                        *diagnostic_args, '--checkpoint', str(checkpoint), '--output', str(directory)], check=True)
        summaries.append(summarize_checkpoint(directory, row))
        pd.concat(summaries, ignore_index=True).to_csv(args.output/'trajectory_summary.csv', index=False)


if __name__ == '__main__':
    main(sys.argv[1:])
