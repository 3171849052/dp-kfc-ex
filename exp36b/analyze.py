"""Read measured results; never fabricate missing epochs or historical runs."""
import json
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exp36b.config import cfg, privacy
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS = {'dp_adamw':'DP-AdamW', 'oracle_dp_kfc_a':'Oracle DP-KFC-A',
          'oracle_dp_kfc':'Oracle Full DP-KFC'}


def family_columns(frame):
    cols = [c for c in frame if c.startswith('layer_norm_')]
    return {
        'attention_qkv': [c for c in cols if any(f'attn_{p}_proj' in c for p in ('q','k','v'))],
        'attention_out_proj': [c for c in cols if 'attn_out_proj' in c],
        'mlp_fc1': [c for c in cols if 'mlp_fc1' in c],
        'mlp_fc2': [c for c in cols if 'mlp_fc2' in c],
        'patch_head': [c for c in cols if c in ('layer_norm_patch_embed','layer_norm_head')],
    }


def main():
    figures = cfg.RESULTS / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    runs = {LABELS[m]: cfg.RESULTS / 'runs' / m for m in cfg.METHODS}
    runs.update({'Pink DP-KFC-A': cfg.ROOT.parent / 'exp35/results/vit/runs/dp_kfc_a_raw',
                 'Pink Full DP-KFC': cfg.ROOT.parent / 'exp22/results/runs/dp_kfc_42'})
    frames, summaries = {}, []
    for label, directory in runs.items():
        frame = pd.read_csv(directory / 'metrics.csv')
        config = json.loads((directory / 'config.json').read_text())
        assert config.get('model', config.get('MODEL_NAME')) == cfg.MODEL_NAME
        assert config['seed'] == 42 and config['epochs'] == 5
        assert config.get('logical_batch_size', config.get('LOGICAL_BATCH_SIZE')) == 256
        assert config.get('physical_batch_size', config.get('PHYSICAL_BATCH_SIZE')) == 128
        assert len(frame) == 5
        frames[label] = frame
        summaries.append(dict(label=label, source=str(directory.relative_to(cfg.ROOT.parent)),
            privacy_status='oracle_non_private_geometry' if label.startswith('Oracle') else 'standard_dp',
            nominal_dp_epsilon=3, final_accuracy=frame.test_accuracy.iloc[-1],
            best_accuracy=frame.test_accuracy.max(), final_test_loss=frame.test_loss.iloc[-1],
            epochs=len(frame), seed=42, model=cfg.MODEL_NAME))
    summary = pd.DataFrame(summaries)
    summary.iloc[:3].to_csv(cfg.RESULTS / 'summary.csv', index=False)
    summary.to_csv(cfg.RESULTS / 'oracle_vs_synthetic.csv', index=False)
    fields = ['test_accuracy','test_loss','clip_fraction','mean_clip_factor',
              'transformed_norm_p90','transformed_norm_p99']
    fields += [c for c in frames['DP-AdamW'] if c.startswith('group_norm_')]
    for field in fields:
        fig, ax = plt.subplots(figsize=(8,5))
        for label, frame in frames.items():
            ax.plot(frame.epoch, frame[field], marker='o', label=label)
        ax.set(xlabel='Epoch', ylabel=field, title='Oracle geometry uses private data; not end-to-end DP')
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figures / f'{field}.png', dpi=150)
        plt.close(fig)
    baseline = frames['DP-AdamW'].set_index('epoch')
    ratios = []
    for label, frame in frames.items():
        for family, cols in family_columns(frame).items():
            assert cols, family
            for _, row in frame.iterrows():
                # Layer metrics are RMS norms: add squared RMS before sqrt.
                numerator = np.sqrt(np.square(row[cols].astype(float)).sum())
                denominator = np.sqrt(np.square(baseline.loc[row.epoch, cols].astype(float)).sum())
                ratios.append(dict(label=label, epoch=int(row.epoch), family=family,
                                   transformed_norm=numerator, dp_adamw_norm=denominator,
                                   ratio=numerator / denominator))
    pd.DataFrame(ratios).to_csv(cfg.RESULTS / 'layer_family_norm_ratios.csv', index=False)
    print(summary.to_string(index=False))
    for oracle, pink in [('Oracle DP-KFC-A','Pink DP-KFC-A'), ('Oracle Full DP-KFC','Pink Full DP-KFC')]:
        print(f'{oracle} vs {pink}: final accuracy difference = '
              f'{frames[oracle].test_accuracy.iloc[-1] - frames[pink].test_accuracy.iloc[-1]:.6f}')
    print('Oracle nominal epsilon describes Gaussian training only; geometry is non-private.')


if __name__ == '__main__':
    main()
