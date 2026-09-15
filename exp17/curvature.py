"""Same-base-pass Fisher and exp15 secants; dense factor geometry in float64."""
import csv
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from dp_kfac.models import SimpleCNN
from dp_kfac.recorder import KFACRecorder
from dp_kfac.standalone.trainer import pink_batches
from exp15.preconditioner import SyntheticKLBFGS, InverseLBFGS, rows


class TinyCNN(SimpleCNN):
    def __init__(self):
        nn.Module.__init__(self)
        self.conv1 = nn.Conv2d(1, 2, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(2, 2, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(98, 8)
        self.fc2 = nn.Linear(8, 10)


class Diagnostic(SyntheticKLBFGS):
    def __init__(self, config, device, factory=SimpleCNN):
        super().__init__(config, device)
        self.factory = factory

    def refresh(self, model, epoch):
        self.totals = {}
        with torch.random.fork_rng(devices=[self.device.index] if self.device.type == 'cuda' else []):
            torch.manual_seed(self.config['seed'] + 10000 + epoch)
            probe = self.factory().to(self.device)
            theta = {n: v.detach().clone() for n, v in model._module.state_dict().items()}
            batches = list(pink_batches(self.config, self.device))
            for x, y in batches:
                probe.load_state_dict(theta)
                self._synthetic_pair(probe, x, y)
        self.fisher = {n: tuple(v / len(batches) for v in pair) for n, pair in self.totals.items()}

    @torch.no_grad()
    def collect(self, probe, recorder, x, y):
        for name, module in probe.named_modules():
            if not isinstance(module, (nn.Conv2d, nn.Linear)):
                continue
            a = recorder.activations[name]
            if isinstance(module, nn.Conv2d):
                a = F.unfold(a, module.kernel_size, padding=module.padding,
                             stride=module.stride).transpose(1, 2).flatten(0, 1)
            a = torch.cat((a, torch.ones_like(a[:, :1])), 1).double()
            g = rows(recorder.backprops[name]).double()
            pair = (a.T @ a / len(a), g.T @ g / len(g))
            if name not in self.totals:
                self.totals[name] = pair
            else:
                for total, value in zip(self.totals[name], pair):
                    total.add_(value)

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
        self.collect(probe, recorder, x, y)
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


def sym(x):
    return (x + x.T) / 2


def spectral_power(values, vectors, power):
    return (vectors * values.pow(power)) @ vectors.T


def condition(x):
    v = torch.linalg.eigvalsh(sym(x))
    assert torch.isfinite(v).all() and (v > 0).all()
    return (v[-1] / v[0]).item()


def cosine(a, b):
    return ((a * b).sum() / (a.norm() * b.norm())).item()


def spectrum(prefix, matrix, values):
    p = values.clamp_min(0)
    p = p / p.sum()
    positive = p[p > 0]
    return {prefix + key: value for key, value in dict(
        lambda_min=values[0].item(), lambda_median=values.quantile(.5).item(),
        lambda_max=values[-1].item(), trace=matrix.trace().item(),
        frobenius_norm=matrix.norm().item(),
        effective_rank=(-(positive * positive.log()).sum()).exp().item()).items()}


@torch.no_grad()
def geometry(fisher, factor, damping, seed):
    fisher = sym(fisher)
    eye = torch.eye(len(fisher), dtype=torch.float64, device=fisher.device)
    Q = sym(factor.hv(eye))
    qv, qu = torch.linalg.eigh(Q)
    assert torch.isfinite(qv).all() and (qv > 0).all()
    B = sym(spectral_power(qv, qu, -1))
    fv, fu = torch.linalg.eigh(fisher)
    bv, bu = torch.linalg.eigh(B)
    assert (bv > 0).all()
    dv = fv + damping
    assert (dv > 0).all()
    whitening = spectral_power(dv, fu, -.5)
    M = sym(whitening @ B @ whitening)
    gv = torch.linalg.eigvalsh(M)
    assert torch.isfinite(gv).all() and (gv > 0).all()
    alpha = (fisher * B).sum() / B.square().sum()
    k = min(10, len(fisher))
    base_condition = (dv[-1] / dv[0]).item()
    metrics = dict(**spectrum('F_', fisher, fv), **spectrum('B_', B, bv),
        condition_F=base_condition, condition_B=(bv[-1]/bv[0]).item(),
        matrix_cosine=cosine(fisher, B), optimal_scale=alpha.item(),
        shape_error=((fisher-alpha*B).norm()/fisher.norm()).item(),
        normalized_commutator=((fisher@B-B@fisher).norm()/(fisher.norm()*B.norm())).item(),
        top_k=k, top_k_overlap=((fu[:, -k:].T@bu[:, -k:]).square().sum()/k).item(),
        generalized_lambda_min=gv[0].item(), generalized_lambda_median=gv.quantile(.5).item(),
        generalized_lambda_max=gv[-1].item(), generalized_condition=(gv[-1]/gv[0]).item(),
        generalized_log10_width=(gv[-1].log10()-gv[0].log10()).item())
    generator = torch.Generator(device=fisher.device).manual_seed(seed)
    vectors = torch.randn(32, len(fisher), generator=generator, device=fisher.device, dtype=torch.float64)
    operators, transformed, extra = [], [], {}
    for q in (.25, .5):
        PH = spectral_power(qv, qu, q)
        torch.testing.assert_close(factor.power(vectors, q), vectors@PH, rtol=1e-7, atol=1e-8)
        PF = spectral_power(dv, fu, -q)
        action = F.cosine_similarity(vectors@PF, vectors@PH, dim=1)
        operators.append(dict(q=q, operator_matrix_cosine=cosine(PF, PH),
            operator_condition_F=condition(PF), operator_condition_H=condition(PH),
            operator_action_cosine_mean=action.mean().item(), operator_action_cosine_std=action.std(unbiased=False).item()))
        z = sym(PH @ fisher @ PH)
        zcondition = condition(z + damping*eye)
        transformed.append(dict(q=q, original_condition=base_condition,
                                transformed_condition=zcondition, condition_ratio=zcondition/base_condition))
        extra[f'Q_power_{q}'] = PH
        extra[f'F_transformed_{q}'] = z
    return metrics, gv, operators, transformed, dict(Q=Q, B=B, M=M, **extra)


def append_csv(path, records):
    exists = path.exists()
    with path.open('a') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        if not exists:
            writer.writeheader()
        writer.writerows(records)
        handle.flush()


def save_checkpoint(state, directory, epoch, private_step):
    target = directory / 'matrices' / f'epoch{epoch}'
    target.mkdir(parents=True, exist_ok=True)
    for layer, pair in state.fisher.items():
        matrices = {}
        for side, fisher, factor in zip(('A', 'G'), pair, state.factors[layer]):
            key = dict(epoch=epoch, private_step=private_step, layer=layer, side=side, dimension=len(fisher))
            metrics, values, ops, transformed, dense = geometry(fisher, factor, state.config['kfac']['damping'], 42000+epoch)
            append_csv(directory/'factors.csv', [dict(**key, **metrics)])
            append_csv(directory/'generalized_spectra.csv', [dict(**key, index=i, eigenvalue=v) for i, v in enumerate(values.tolist())])
            append_csv(directory/'operator_metrics.csv', [dict(**key, **r) for r in ops])
            append_csv(directory/'transformed_conditions.csv', [dict(**key, **r) for r in transformed])
            matrices[f'{side}_F'] = fisher.cpu()
            matrices[f'Q_{side}'] = dense.pop('Q').cpu()
            matrices[f'B_{side}'] = dense.pop('B').cpu()
            matrices[f'generalized_eigenvalues_{side}'] = values.cpu()
            matrices.update({f'{side}_{n}': v.cpu() for n, v in dense.items()})
        torch.save(matrices, target/f'{layer}.pt')
