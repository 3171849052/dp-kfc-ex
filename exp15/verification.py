"""Small assertions shared by the reset regression test and the tiny smoke."""
import torch
from exp15.preconditioner import SyntheticKLBFGS


class CheckedSyntheticKLBFGS(SyntheticKLBFGS):
    def refresh(self, model, epoch):
        self.theta_t = {n: p.detach().clone() for n, p in model._module.state_dict().items()}
        self.before_pair_counts = []
        super().refresh(model, epoch)
        assert len(self.before_pair_counts) >= 2
        assert self.before_pair_counts[1] > self.before_pair_counts[0]
        assert len(self.factors['conv1'][0].pairs['s']) > 1
        for n, p in model._module.state_dict().items():
            torch.testing.assert_close(p, self.theta_t[n], rtol=0, atol=0)

    def _synthetic_pair(self, probe, x, y):
        # Observe the actual before-forward, not just entry to the function.
        forward_count = 0
        def check_before(module, inputs):
            nonlocal forward_count
            if forward_count == 0:
                for n, p in module.state_dict().items():
                    torch.testing.assert_close(p, self.theta_t[n], rtol=0, atol=0)
            forward_count += 1
        handle = probe.register_forward_pre_hook(check_before)
        count = sum(f.accepted for fs in self.factors.values() for f in fs)
        self.before_pair_counts.append(count)
        if len(self.before_pair_counts) > 1:
            # The preceding accepted pair itself, not only its counter, persists.
            torch.testing.assert_close(self.factors['conv1'][0].pairs['s'],
                                       self.previous_pairs, rtol=0, atol=0)
        super()._synthetic_pair(probe, x, y)
        handle.remove()
        assert forward_count == 2
        assert any(not torch.equal(p, self.theta_t[n]) for n, p in probe.state_dict().items())
        self.previous_pairs = self.factors['conv1'][0].pairs['s'].clone()
