"""CPU-only mathematical, integration and checkpoint-boundary checks."""
import os
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exp36.runtime import ROOT, setup


def main():
    cfg = setup()
    import subprocess
    import tempfile
    from unittest.mock import patch
    import pandas as pd
    import torch
    from exp36 import diagnose, metrics, analyze, train
    from exp35 import vit
    torch.set_num_threads(2)
    for path in ROOT.glob('*.py'):
        compile(path.read_text(),str(path),'exec')
    subprocess.run(['bash','-n',str(ROOT/'run_all.sh')],check=True)
    assert cfg.GPU == 3 and 'export CUDA_VISIBLE_DEVICES=3' in (ROOT/'run_all.sh').read_text()
    assert cfg.DATA_ROOT == ROOT.parent/'data'
    ref, replica = diagnose.private_indices()
    assert len(ref) == len(replica) == 2560
    assert len(set(ref.tolist()) | set(replica.tolist())) == 5120
    assert torch.equal(ref,diagnose.private_indices()[0])
    with patch.object(vit.datasets,'CIFAR10',wraps=vit.datasets.CIFAR10) as constructor:
        dataset, test = vit.load_data()
        assert len(dataset) == 50000 and len(test) == 10000
        for call in constructor.call_args_list:
            assert Path(call.args[0]) == ROOT.parent/'data' and call.kwargs['download'] is False
    model = torch.nn.Sequential(torch.nn.Linear(3,4))
    x = torch.randn(2,5,3)
    with torch.no_grad():
        factors, stats = diagnose.covariance_builder()(model,[x],.4,.001)
    augmented = torch.cat((x.reshape(-1,3),torch.ones(10,1)),dim=1)
    torch.testing.assert_close(factors.factors['0']['A'],augmented.T@augmented/10)
    assert stats['builder_samples'] == 2
    values = torch.arange(1,41,dtype=torch.double)
    a = torch.diag(values)
    q, _ = torch.linalg.qr(torch.randn(40,40,dtype=torch.double))
    b = q@a@q.T
    r, e = metrics.decompose(a), metrics.decompose(b)
    same, rotated = metrics.compare(r,r), metrics.compare(r,e)
    assert abs(same['cos_A']-1) < 1e-12 and same['rel_frob_A'] == 0
    assert abs(same['cos_P']-1) < 1e-12 and same['rel_frob_P'] == 0
    for k in (16,32):
        qr, qe = r['vectors'][:,-k:], e['vectors'][:,-k:]
        expected = torch.trace((qr@qr.T)@(qe@qe.T))/k
        assert abs(rotated[f'top{k}_overlap']-expected.item()) < 1e-12
        assert abs(same[f'top{k}_overlap']-1) < 1e-12
    assert rotated['cos_A'] < .99 and rotated['cos_P'] < .999
    p = torch.diag((values+.001).pow(-.4))
    torch.testing.assert_close(r['P'],p)
    expected_p = q@p@q.T
    torch.testing.assert_close(e['P'],expected_p)
    for label, matrix_ref, matrix_est in (('A',a,b),('P',p,expected_p)):
        expected_cos = torch.trace(matrix_ref.T@matrix_est)/(torch.linalg.matrix_norm(matrix_ref)*torch.linalg.matrix_norm(matrix_est))
        expected_rel = torch.linalg.matrix_norm(matrix_est-matrix_ref)/torch.linalg.matrix_norm(matrix_ref)
        assert abs(rotated[f'cos_{label}']-expected_cos.item()) < 1e-12
        assert abs(rotated[f'rel_frob_{label}']-expected_rel.item()) < 1e-12
    scaled = metrics.compare(r,metrics.decompose(2*a))
    assert abs(scaled['cos_A']-1) < 1e-12 and abs(scaled['rel_frob_A']-1) < 1e-12
    assert abs(scaled['trace_ratio']-2) < 1e-12
    probabilities = values/values.sum()
    assert abs(r['effective_rank']-torch.exp(-(probabilities*probabilities.log()).sum()).item()) < 1e-12
    assert metrics.family('blocks.11.attn.q_proj') == ('attention_qkv',11)
    # Exercise the actual train.py injection boundaries with a tiny model and
    # mocked reference forward functions, never calling the training loop.
    with tempfile.TemporaryDirectory(dir=ROOT/'.cache/tmp') as tmp:
        temporary = Path(tmp)
        state = torch.random.get_rng_state().clone()
        toy = torch.nn.Linear(2,2)
        before = torch.random.get_rng_state().clone()
        def exercise(method):
            namespace = vit.reference.run.__globals__
            actual = namespace['initialize'](42,'cpu')
            for _ in range(5):
                namespace['evaluate'](actual,None,'cpu')
        with patch.object(train,'ROOT',temporary), patch.object(train,'setup',lambda: cfg), \
             patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'3'}), \
             patch.object(vit.reference,'initialize',lambda seed,device: toy), \
             patch.object(vit.reference,'evaluate',lambda model,loader,device: (0.,0.)), \
             patch.object(vit,'run',exercise), patch.object(vit.reference,'run',vit.reference.run):
            train.main()
        assert torch.equal(before,torch.random.get_rng_state())
        table = pd.read_csv(temporary/'results/checkpoints.csv')
        assert table.epoch.tolist() == list(range(6))
        for row in table.itertuples():
            saved = torch.load(temporary/row.checkpoint,weights_only=True)
            for name,tensor in toy.state_dict().items():
                torch.testing.assert_close(saved[name],tensor)
        torch.random.set_rng_state(state)
    assert vit.model_cfg.ROOT == ROOT
    assert Path(os.environ['HF_HOME']).is_relative_to(ROOT)
    print('PASS: compile/import/shell, GPU 3, root data/download=False, disjoint fixed indices, Exp22 flattened covariance, A/P matrix metrics, subspace overlap, effective rank, checkpoint boundaries and RNG preservation, local cache')


if __name__ == '__main__':
    main()
