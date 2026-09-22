"""CPU-only checks of actual data and geometry; never writes formal results."""
import csv
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exp36c.config import cfg, ALPHAS, ORACLE_INDICES
from exp36c.run import run
from exp36c.geometry import builder, calibration, bounded_spectral
from exp36c.analyze import RUNS, read_run, family_columns
from exp35.vit import load_data, reference
from exp22.handlers import flatten_linear_input
from exp35.adapters import bind
from exp22.geometry import build_a_operator
import torch
from torch import nn


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(3, 4)

    def forward(self, x):
        return self.head(x)


def main():
    torch.set_num_threads(4)
    for path in cfg.ROOT.glob('*.py'):
        compile(path.read_text(), str(path), 'exec')
    assert cfg.METHODS == ('oracle_dp_kfc_a_alpha025', 'oracle_dp_kfc_a_alpha05')
    assert cfg.grid('vit') == cfg.METHODS
    assert set(cfg.EXP22_METHOD.values()) == {'dp_kfc_a_bk'}
    assert (cfg.SEED, cfg.EPOCHS, cfg.DAMPING, cfg.A_POWER) == (42, 5, .001, .4)
    assert (cfg.LOGICAL_BATCH_SIZE, cfg.PHYSICAL_BATCH_SIZE, cfg.MAX_GRAD_NORM) == (256, 128, 1)
    assert (cfg.LEARNING_RATE, cfg.WEIGHT_DECAY, cfg.BETAS, cfg.ADAM_EPS) == (1e-4, .01, (.9, .999), 1e-8)
    with ORACLE_INDICES.open() as stream:
        indices = [int(row['index']) for row in csv.DictReader(stream)]
    assert len(indices) == len(set(indices)) == 2560
    assert indices == torch.randperm(50000, generator=torch.Generator().manual_seed(42))[:2560].tolist()
    assert calibration.__globals__['cfg'].RESULTS / 'oracle_indices.csv' == ORACLE_INDICES
    train, test = load_data(download=False)
    assert Path(train.root) == Path(test.root) == cfg.ROOT.parent / 'data'
    assert (len(train), len(test)) == (50000, 10000)
    count = 0
    for x, y in calibration(train, 'cpu'):
        assert len(x) == len(y) == 256
        assert y.tolist() == [train.targets[i] for i in indices[count:count+256]]
        torch.testing.assert_close(x[0], train[indices[count]][0], rtol=0, atol=0)
        count += len(x)
    assert count == 2560
    torch.manual_seed(42)
    model = Tiny()
    tokens = torch.randn(256, 7, 3)
    flattened = flatten_linear_input(tokens, model.head)
    assert flattened.shape == (256*7, 4)
    assert torch.equal(flattened[:, -1], torch.ones(256*7))
    covariance = flattened.T @ flattened / len(flattened)
    captured = {}

    def capture(factors, power, damping):
        captured.update(factors)

    # Use the real accumulation code independently of the spectral transform.
    class Captured:
        moments = diagnostics = {}
        operator_state_bytes = 0
        def __init__(self, factors, power, damping):
            capture(factors, power, damping)
    with torch.no_grad():
        bind(build_a_operator.__wrapped__, AOnlyOperator=Captured)(model, [tokens[:128], tokens[128:]])
    torch.testing.assert_close(captured['head']['A'], covariance)
    original = {k: v.clone() for k, v in model.state_dict().items()}
    for method, alpha in ALPHAS.items():
        operator, stats = builder(method)(model, 'dp_kfc_a_bk', [(tokens, torch.zeros(256))])
        values, vectors = torch.linalg.eigh(covariance.double())
        raw = (values+.001).pow(-.4)
        expected_gains = 1-alpha+alpha*raw/raw.max()
        expected = ((vectors*expected_gains)@vectors.T).float()
        torch.testing.assert_close(operator.data['head'], expected)
        assert operator.scale == 1 and stats['builder_samples'] == 256
        assert stats['builder_vjp_calls'] == 0 and stats['builder_reverse_vectors'] == 0
        assert all(torch.equal(v, original[k]) for k, v in model.state_dict().items())
        assert all(p.grad is None for p in model.parameters())
        for a in (covariance, torch.diag(torch.tensor([0., 1e-5, 1., 1000.]))):
            matrix, gains = bounded_spectral(a, alpha)
            eigenvalues = torch.linalg.eigvalsh(matrix.double())
            assert eigenvalues.min() >= 1-alpha-2e-6 and eigenvalues.max() <= 1+2e-6
            torch.testing.assert_close(torch.linalg.matrix_norm(matrix, 2), torch.tensor(1.), atol=2e-6, rtol=0)
    assert reference.Clipper.__module__ == 'exp22.methods'
    for label, path in RUNS.items():
        if not path.startswith('exp36c/'):
            frame = read_run(path)
            assert all(family_columns(frame).values())
            assert all('group_norm_'+g in frame for g in ('attention_qkv', 'attention_out', 'mlp', 'patch_head', 'identity'))
    assert not list((cfg.RESULTS / 'runs').glob('*/metrics.csv'))
    print('PASS: imports/compile; exactly 2 runs; fixed protocol; root data/download=False;')
    print('same 2560 Exp36b indices and 10x256 calibration; token flatten/bias/A formula;')
    print('spectral normalization/mixing and both gain bounds; calibration leaves parameters/gradients unchanged;')
    print('reference BK loop; all 4 historical comparisons and layer/group norms; no formal results created.')


if __name__ == '__main__':
    main()
