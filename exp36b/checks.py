"""CPU checks with real CIFAR inputs, tiny calibration model, no training run."""
import ast
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exp36b.config import cfg
from exp36b.oracle_geometry import indices, save_indices, calibration, build
from exp35.vit import load_data, reference
from exp22.geometry import inverse_sqrt
from exp22.handlers import flatten_linear_input
import pandas as pd
import torch
from torch import nn


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(3, 10)
        self.num_classes = 10
    def forward(self, x):
        return self.head(x.mean((2,3)))


def main():
    torch.set_num_threads(4)
    for path in cfg.ROOT.glob('*.py'):
        compile(path.read_text(), str(path), 'exec')
    assert cfg.METHODS == ('dp_adamw','oracle_dp_kfc_a','oracle_dp_kfc')
    assert (cfg.SEED,cfg.EPOCHS,cfg.A_POWER,cfg.DAMPING,cfg.PHYSICAL_BATCH_SIZE) == (42,5,.4,.001,128)
    assert len(indices()) == len(set(indices())) == 2560
    assert indices() == indices()
    save_indices()
    assert pd.read_csv(cfg.RESULTS / 'oracle_indices.csv')['index'].tolist() == indices()
    train, test = load_data(download=False)
    assert Path(train.root) == cfg.ROOT.parent / 'data'
    assert len(train) == 50000 and len(test) == 10000
    x, y = next(calibration(train, 'cpu'))
    assert torch.equal(x[0], train[indices()[0]][0])
    assert y.tolist() == [train.targets[i] for i in indices()[:256]]
    # Check complete calibration stream count/order without a ViT forward.
    seen = 0
    for xb, yb in calibration(train, 'cpu'):
        assert len(xb) == len(yb) == 256
        seen += len(xb)
    assert seen == 2560
    model = Tiny()
    original = {k:v.clone() for k,v in model.state_dict().items()}
    a = flatten_linear_input(x.mean((2,3)), model.head)
    covariance = a.T @ a / len(x)
    aop, stats = build(model,'dp_kfc_a_bk',[(x,y)])
    values, vectors = torch.linalg.eigh(((covariance+covariance.T)/2).double())
    expected = ((vectors*(values.clamp_min(0)+.001).pow(-.4))@vectors.T).float()
    torch.testing.assert_close(aop.data['head'], expected)
    assert aop.scale == 1 and stats['builder_samples'] == 256
    full, stats = build(model,'dp_kfc',[(x,y)])
    logits = model(x).detach()
    errors = logits.softmax(-1) - torch.nn.functional.one_hot(y,10)
    g = errors.T @ errors / len(y)
    torch.testing.assert_close(full.factors['head']['A'], covariance)
    torch.testing.assert_close(full.factors['head']['G'], g)
    torch.testing.assert_close(full.data['head'][0], inverse_sqrt(g,.001))
    torch.testing.assert_close(full.data['head'][1], inverse_sqrt(covariance,.001))
    assert stats['builder_reverse_vectors'] == 256
    assert all(torch.equal(v, original[k]) for k,v in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters())
    tokens = torch.randn(2,7,3)
    assert flatten_linear_input(tokens,model.head).shape == (14,4)
    assert reference.Clipper.__module__ == 'exp22.methods'
    from exp36b.analyze import family_columns
    old = pd.read_csv(cfg.ROOT.parent / 'exp22/results/runs/dp_kfc_42/metrics.csv')
    assert all(family_columns(old).values())
    print('PASS: imports/compile, 3 fixed runs, root data/download=False, fixed 2560 indices,')
    print('private images/true labels, unchanged parameters, cleared gradients, exact A/G operators,')
    print('token flatten, reference BK loop, historical layer families. No full training started.')


if __name__ == '__main__':
    main()
