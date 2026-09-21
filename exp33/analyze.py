"""Analyze only four complete formal runs; never fabricate incomplete results."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from exp33 import config as cfg
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    runs = {}
    for method in cfg.METHODS:
        frame = pd.read_csv(cfg.RESULTS / 'runs' / method / 'metrics.csv')
        assert frame.epoch.tolist() == [1, 2, 3, 4, 5], method
        assert frame.iloc[-1].accountant_steps == frame.iloc[-1].optimizer_steps == frame.iloc[-1].noise_events == 975
        runs[method] = frame
    summary = pd.DataFrame([f.iloc[-1] for f in runs.values()])
    assert len(summary) == 4
    summary.to_csv(cfg.RESULTS / 'summary.csv', index=False)
    a, w, f, k = [runs[m].iloc[-1].test_accuracy for m in cfg.METHODS]
    pd.DataFrame([dict(dp_adamw_accuracy=a, dp_wiener_a_accuracy=w, dp_wiener_full_accuracy=f,
                       dp_kfc_a_01_accuracy=k, delta_wiener_a_vs_adamw=w-a,
                       delta_wiener_full_vs_adamw=f-a, delta_wiener_full_vs_a=f-w,
                       delta_wiener_a_vs_kfc_a=w-k)]).to_csv(cfg.RESULTS / 'paired.csv', index=False)
    plots = {
        'accuracy_by_method': ['test_accuracy'], 'test_loss_by_method': ['test_loss'],
        'clip_fraction_by_method': ['clip_fraction'],
        'wiener_gain_by_group': [f'group_wiener_gain_{g}' for g in ('attention_qkv','attention_out','mlp','patch_head')],
        'wiener_norm_ratio_by_group': [f'group_wiener_norm_ratio_{g}' for g in ('attention_qkv','attention_out','mlp','patch_head')],
        'nsr_raw_vs_filtered': ['nsr_raw', 'nsr_filtered'],
        'cos_raw_vs_filtered': ['cos_raw_clean', 'cos_filtered_clean'],
    }
    for filename, columns in plots.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        for method, frame in runs.items():
            for column in columns:
                if column in frame:
                    ax.plot(frame.epoch, frame[column], marker='o', label=f'{cfg.LABELS[method]} {column}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(filename)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(cfg.RESULTS / (filename + '.png'), dpi=160)
        plt.close(fig)

if __name__ == '__main__':
    main()
