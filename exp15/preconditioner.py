"""Thin adapter to the sibling implementation, plus a compact spectral power."""
from pathlib import Path
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'kbfgs_neurips2020_public'))
from kbfgs_utils import (LBFGS_Hv, Kron_LBFGS_append_s_y,
                        get_BFGS_PowellHDamping, get_BFGS_ModifiedDamping)
from dp_kfac.models import SimpleCNN
from dp_kfac.recorder import KFACRecorder
from dp_kfac.standalone.trainer import pink_batches


class InverseLBFGS:
    def __init__(self, device, config):
        self.params = dict(device=device, Kron_BFGS_number_s_y=config['memory'],
                           Kron_BFGS_action_a='LBFGS',
                           Kron_BFGS_H_epsilon=config['factor_damping'])
        self.gamma = config['initial_scale']
        self.pairs = {k: [] for k in ('s', 'y', 'R_inv', 'yTy', 'D_diag',
                                     'left_matrix', 'right_matrix')}
        self.pairs['gamma'] = self.gamma
        self.accepted = self.rejected = self.powell = self.modified = 0
        self.spectrum = None

    def hv(self, v):
        return LBFGS_Hv(v, self.pairs, self.params)

    def append(self, s, y, gradient=None, damp=False):
        s, y = s.double(), y.double()
        if damp:
            data = {'Kron_BFGS_matrices': [{}],
                    'Kron_LBFGS_s_y_pairs': {'a': [self.pairs]}}
            s, y, ratio = get_BFGS_PowellHDamping(s, y, .2, 0, data, self.params)
            self.powell += int(ratio <= .2)
            self.modified += int((s @ y / (s @ s)) <= self.params['Kron_BFGS_H_epsilon'])
            s, y = get_BFGS_ModifiedDamping(s, y, 0, data, self.params)
        gradient = [] if gradient is None else gradient.double()
        # The upstream append replaces s on acceptance (also when memory is full).
        previous = self.pairs['s']
        self.pairs = Kron_LBFGS_append_s_y(s, y, self.pairs, gradient,
                                         self.gamma, self.params)
        accepted = self.pairs['s'] is not previous
        self.accepted += int(accepted)
        self.rejected += int(not accepted)
        self.spectrum = None

    def power(self, v, p):
        """Apply on the final dimension. No n-by-n Hessian/inverse is formed."""
        if p == 0 or len(self.pairs['s']) == 0:
            return v
        if self.spectrum is None:
            left = self.pairs['left_matrix']
            q, _ = torch.linalg.qr(left, mode='reduced')
            small = q.T @ self.hv(q)
            # Remove floating-point antisymmetry, without altering eigenvalues.
            eigenvalues, eigenvectors = torch.linalg.eigh((small + small.T) / 2)
            assert torch.isfinite(eigenvalues).all() and (eigenvalues > 0).all(), eigenvalues
            self.spectrum = (q @ eigenvectors, eigenvalues)
        basis, eigenvalues = self.spectrum
        gamma = self.pairs['gamma']
        # Spectral work is float64; private gradient GEMMs retain model dtype.
        basis = basis.to(v.dtype)
        correction = (eigenvalues.pow(p) - gamma ** p).to(v.dtype)
        return gamma ** p * v + ((v @ basis) * correction) @ basis.T

    def diagnostics(self):
        return dict(accepted=self.accepted, rejected=self.rejected,
                    powell_damping=self.powell, modified_damping=self.modified)


def rows(t):
    return t.movedim(1, -1).reshape(-1, t.shape[1]) if t.ndim == 4 else t


class SyntheticKLBFGS:
    def __init__(self, config, device):
        self.config, self.device = config, device
        self.factors, self.moments, self.activation_cov = {}, {}, {}

    def refresh(self, model, epoch):
        # A fresh ordinary CNN avoids copying Opacus hooks or private gradients.
        with torch.random.fork_rng(devices=[self.device.index] if self.device.type == 'cuda' else []):
            torch.manual_seed(self.config['seed'] + 10000 + epoch)
            probe = SimpleCNN().to(self.device)
            probe.load_state_dict(model._module.state_dict())
            for x, y in pink_batches(self.config, self.device):
                self._synthetic_pair(probe, x, y)

    def _synthetic_pair(self, probe, x, y):
        c = self.config['lbfgs']
        recorder = KFACRecorder(probe)
        recorder.enable()
        outputs = {}
        handles = [m.register_forward_hook(
            lambda module, inputs, out, name=n: outputs.__setitem__(name, out.detach()))
            for n, m in probe.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
        probe.zero_grad(set_to_none=True)
        F.cross_entropy(probe(x), y, reduction='sum').backward()
        before = {n: (outputs[n].clone(), recorder.backprops[n].clone()) for n in outputs}
        with torch.no_grad():
            for name, module in probe.named_modules():
                if name not in outputs:
                    continue
                a = recorder.activations[name]
                if isinstance(module, nn.Conv2d):
                    a = F.unfold(a, module.kernel_size, padding=module.padding,
                                 stride=module.stride).transpose(1, 2).flatten(0, 1)
                a = torch.cat((a, torch.ones_like(a[:, :1])), dim=1).double()
                cov = a.T @ a / len(a)
                if name not in self.factors:
                    self.factors[name] = (InverseLBFGS(self.device, c), InverseLBFGS(self.device, c))
                    self.activation_cov[name] = cov
                else:
                    self.activation_cov[name].lerp_(cov, 1 - c['activation_decay'])
                ha, _ = self.factors[name]
                s = ha.hv(a.mean(0))
                ha.append(s, self.activation_cov[name] @ s + c['factor_damping'] * s)
            # Virtual SGD step only on this disposable synthetic model.
            for parameter in probe.parameters():
                parameter.add_(parameter.grad, alpha=-c['synthetic_lookahead_lr'] / len(x))
        probe.zero_grad(set_to_none=True)
        F.cross_entropy(probe(x), y, reduction='sum').backward()
        for name, (old_z, old_backprop) in before.items():
            s = (rows(old_z).mean(0) - rows(outputs[name]).mean(0)).double()
            g = rows(old_backprop).mean(0).double()
            y_pair = g - rows(recorder.backprops[name]).mean(0).double()
            old_s, old_y = self.moments.get(name, (torch.zeros_like(s), torch.zeros_like(y_pair)))
            decay = c['pair_decay']
            s, y_pair = decay * old_s + (1-decay) * s, decay * old_y + (1-decay) * y_pair
            self.moments[name] = (s, y_pair)
            self.factors[name][1].append(s, y_pair, g, damp=True)
        recorder.remove()
        for handle in handles:
            handle.remove()

    @torch.no_grad()
    def apply(self, model, p):
        if p == 0:
            return
        for name, module in model._module.named_modules():
            if name not in self.factors:
                continue
            ha, hg = self.factors[name]
            weight = module.weight.grad_sample
            matrix = torch.cat((weight.flatten(2), module.bias.grad_sample.unsqueeze(-1)), dim=-1)
            matrix = hg.power(ha.power(matrix, p).transpose(-1, -2), p).transpose(-1, -2)
            module.weight.grad_sample = matrix[..., :-1].reshape_as(weight)
            module.bias.grad_sample = matrix[..., -1]

    def diagnostics(self):
        return {f'{name}/{side}': factor.diagnostics()
                for name, factors in self.factors.items()
                for side, factor in zip(('a', 'g'), factors)}
