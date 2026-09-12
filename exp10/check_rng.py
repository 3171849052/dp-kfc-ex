"""Check paired synthetic/probe draws and preservation of private RNG state."""
import hashlib
import torch
import run_exp10 as exp


def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()


def main():
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    original_batches, original_probes = exp.pink_batches, exp.rademacher
    draws = []

    def batches(n, device):
        for x, y in original_batches(n, device):
            draws.append((digest(x), digest(y)))
            yield x, y

    def probes(params, k):
        result = original_probes(params, k)
        draws.append(tuple(digest(z) for z in result.values()))
        return result

    exp.pink_batches, exp.rademacher = batches, probes
    for seed in exp.SEEDS:
        paired = []
        for method in exp.METHODS[1:]:
            torch.manual_seed(seed)
            model = exp.GradSampleModule(exp.SimpleCNN().to(device), loss_reduction='sum')
            initial = tuple(digest(p) for p in model.parameters())
            cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state(device)
            draws.clear()
            state, gains = exp.build_preconditioner(model, method, seed, 1, device)
            assert torch.equal(cpu_rng, torch.get_rng_state())
            assert torch.equal(cuda_rng, torch.cuda.get_rng_state(device))
            assert all(torch.isfinite(s).all() and (s >= .1).all() and (s <= 10).all()
                       for s in state['scales'].values())
            paired.append((initial, list(draws)))
            model.remove_hooks()
        assert paired[0] == paired[1]
        print(f'seed={seed}: identical initialization, 10 synthetic batches/labels, K=8 probes; CPU/CUDA RNG restored')


if __name__ == '__main__':
    main()
