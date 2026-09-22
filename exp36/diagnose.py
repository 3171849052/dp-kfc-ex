"""A research-only process, loading each frozen reference checkpoint."""
import os
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exp36.runtime import ROOT, setup


def private_indices():
    import torch
    indices = torch.randperm(50000, generator=torch.Generator().manual_seed(42))
    return indices[:2560], indices[2560:5120]


def white_stream(epoch, device):
    import torch
    generator = torch.Generator(device=device).manual_seed(42+10000+epoch)
    for _ in range(10):
        images = torch.randn(256, 3, 224, 224, device=device, generator=generator)
        yield images/(images.flatten(1).std(dim=1).view(-1,1,1,1)+1e-8)*.5


def covariance_builder():
    from exp35.adapters import bind
    from exp22.geometry import build_a_operator

    class Covariances:
        def __init__(self, factors, power, damping):
            self.factors = factors
            self.operator_state_bytes = 0
            self.moments, self.diagnostics = {}, {}

    # Keep Exp22's exact hooks, bias augmentation, token flattening, counts and
    # physical forward splitting; stop before construction of any optimizer P.
    return bind(build_a_operator.__wrapped__, AOnlyOperator=Covariances)


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '3'
    setup()
    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader, Subset
    from exp35 import vit
    from exp22.geometry import synthetic_stream
    from exp36.metrics import decompose, compare, family
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    checkpoints = pd.read_csv(ROOT/'results/checkpoints.csv')
    assert checkpoints.epoch.tolist() == list(range(6))
    train, _ = vit.load_data()
    ref, replica = private_indices()
    assert len(set(ref.tolist()) & set(replica.tolist())) == 0
    pd.DataFrame({'private_ref':ref.tolist(), 'private_replica':replica.tolist()}).to_csv(ROOT/'results/private_indices.csv', index=False)
    model = vit.initialize(42, device)
    build = covariance_builder()
    rows = []
    spectra_dir = ROOT/'results/spectra'
    spectra_dir.mkdir(parents=True, exist_ok=True)
    for checkpoint in checkpoints.itertuples():
        epoch = checkpoint.epoch
        model.load_state_dict(torch.load(ROOT/checkpoint.checkpoint, map_location=device, weights_only=True))
        model.eval()
        sources = {
            'private_ref': (x.to(device) for x, _ in DataLoader(Subset(train, ref.tolist()), batch_size=256, shuffle=False)),
            'private_replica': (x.to(device) for x, _ in DataLoader(Subset(train, replica.tolist()), batch_size=256, shuffle=False)),
            'pink': synthetic_stream(42, epoch, device),
            'white': white_stream(epoch, device),
        }
        reference, spectra = {}, {}
        for source, batches in sources.items():
            with torch.no_grad():
                covariances, stats = build(model, batches, .4, .001)
                assert stats['builder_samples'] == 2560
                for name, factor in covariances.factors.items():
                    # Decompose one layer at a time on CPU, retaining only the
                    # current checkpoint's reference matrices/eigenvectors.
                    item = decompose(factor['A'].cpu())
                    spectra[source+'::'+name] = item['values'].numpy()
                    if source == 'private_ref':
                        reference[name] = item
                    else:
                        group, block = family(name)
                        rows.append(dict(epoch=epoch, layer=name, family=group, block_index=block,
                                         source=source, samples=2560, **compare(reference[name], item)))
                del covariances
            print(f'epoch={epoch} source={source}: {len(reference)} Linear layers', flush=True)
        np.savez_compressed(spectra_dir/f'epoch_{epoch}.npz', **spectra)
    frame = pd.DataFrame(rows)
    frame.to_csv(ROOT/'results/layer_metrics.csv', index=False)
    metric_columns = [c for c in frame.select_dtypes('number').columns if c not in ('epoch', 'block_index', 'samples')]
    grouped = frame.groupby(['epoch', 'source', 'family'])
    summary = grouped[metric_columns].mean()
    summary['layer_count'] = grouped.size()
    summary.reset_index().to_csv(ROOT/'results/group_metrics.csv', index=False)


if __name__ == '__main__':
    main()
