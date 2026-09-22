"""Two new points; optionally read Exp35 for the full five-point curve."""
import argparse
import json
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exp35b.runtime import configure
ROOT = Path(__file__).resolve().parent
configure(ROOT)
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--include-exp35', action='store_true')
    args = parser.parse_args()
    frames = []
    for dataset in ('mnist', 'vit'):
        points = [(ROOT, .25, 'dp_kfc_a_alpha025'), (ROOT, .75, 'dp_kfc_a_alpha075')]
        if args.include_exp35:
            points += [(ROOT.parent/'exp35', 0., 'dp_adam' if dataset == 'mnist' else 'dp_adamw'),
                       (ROOT.parent/'exp35', .5, 'dp_kfc_a_mix05'),
                       (ROOT.parent/'exp35', 1., 'dp_kfc_a_norm')]
        for origin, alpha, method in points:
            directory = origin/'results'/dataset/'runs'/method
            configuration = json.loads((directory/'config.json').read_text())
            frame = pd.read_csv(directory/'metrics.csv')
            assert configuration['seed'] == 42 and configuration['epochs'] == 5
            assert frame.epoch.tolist() == [1,2,3,4,5]
            assert frame.method.eq(method).all()
            assert frame.parameters_finite.all() and frame.parameters_updated.all()
            expected = frame.epoch*((60000 if dataset == 'mnist' else 50000)//256)
            for field in ('logical_steps', 'optimizer_steps', 'noise_steps', 'accountant_steps'):
                assert frame[field].eq(expected).all()
            frame = frame.assign(dataset=dataset, alpha=alpha, source_path=str(directory))
            frames.append(frame)
    all_epochs = pd.concat(frames, ignore_index=True)
    summary = all_epochs[all_epochs.epoch == 5].sort_values(['dataset', 'alpha'])
    suffix = '_five_point' if args.include_exp35 else ''
    summary.to_csv(ROOT/'results'/f'summary{suffix}.csv', index=False)
    for dataset in ('mnist', 'vit'):
        destination = ROOT/'results'/dataset
        local = summary[summary.dataset == dataset]
        local.to_csv(destination/f'summary{suffix}.csv', index=False)
        for field in ('test_accuracy', 'clip_fraction', 'mean_clip_factor', 'transformed_norm_p90', 'transformed_norm_p99'):
            fig, ax = plt.subplots(figsize=(7,4))
            for epoch, frame in all_epochs[all_epochs.dataset == dataset].groupby('epoch'):
                frame = frame.sort_values('alpha')
                ax.plot(frame.alpha, frame[field], 'o-', label=f'epoch {epoch}')
            ax.set(xlabel='alpha', ylabel=field, title=dataset.upper())
            ax.legend()
            fig.tight_layout()
            fig.savefig(destination/f'{field}{suffix}.png', dpi=180)
            plt.close(fig)


if __name__ == '__main__':
    main()
