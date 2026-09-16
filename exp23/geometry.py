"""Raw factors and research diagnostics, separate from the private update."""
import torch
from exp12.curvature import layers, activation_sum, estimate_kfac_with_labels
from exp13.operator import Operator as FullOperator
from exp19.methods import activation_forward_only
from exp20.methods import AOperator as PowerOperator
from exp23.config import A_POWER, RESEARCH_ONLY


class AOperator(PowerOperator):
    def __init__(self, factors):
        self.power = A_POWER
        super().__init__(factors, self.power)


class IdentityOperator(FullOperator):
    def __init__(self, model):
        self.data = dict.fromkeys(layers(model))

    def transform_activation(self, name, a):
        return a

    def transform_backprop(self, name, b):
        return b

    def transform_matrix(self, name, g):
        return g


@torch.no_grad()
def activation_factors(model, xs):
    factors, counts = {}, {}
    for x in xs:
        acts = activation_forward_only(model, x)
        for name, module in layers(model).items():
            a, count = activation_sum(acts[name], module)
            if name not in factors:
                factors[name] = dict(A=torch.zeros_like(a), output_dimension=module.weight.shape[0])
                counts[name] = 0
            factors[name]['A'].add_(a)
            counts[name] += count
    for name, factor in factors.items():
        factor['A'].div_(counts[name])
    return factors


def build(model, method, batches):
    stats = dict(builder_forward_calls=0, builder_vjp_calls=0, builder_backward_calls=0,
                 builder_reverse_vectors=0, builder_samples=0)
    if method == 'dp_sgd':
        return IdentityOperator(model), {}, stats
    xs = [x for x, _ in batches]
    if method.startswith('a_'):
        factors = activation_factors(model, xs)
        operator = AOperator(factors)
        stats.update(builder_forward_calls=len(xs), builder_samples=sum(map(len, xs)), **operator.moments)
    else:
        factors, counts = estimate_kfac_with_labels(model, xs, [y for _, y in batches])
        operator = FullOperator(factors)
        stats.update({f'builder_{key}': value for key, value in counts.items()})
    return operator, factors, stats


def alignment(model, factors, oracle_batches, full):
    xs = [x for x, _ in oracle_batches]
    if full:
        oracle, _ = estimate_kfac_with_labels(model, xs, [y for _, y in oracle_batches])
    else:
        oracle = activation_factors(model, xs)
    rows = []
    for name, factor in factors.items():
        row = dict(layer=name, diagnostic_scope=RESEARCH_ONLY)
        for key, label in [('A', 'A')] + ([('C', 'G')] if full else []):
            a, b = oracle[name][key].double(), factor[key].double()
            row[f'cos{label}'] = ((a*b).sum()/(a.norm()*b.norm())).item()
            row[f'relative_frobenius_{label}'] = ((a-b).norm()/a.norm()).item()
        if full:
            row['fisher_cosine_proxy'] = row['cosA']*row['cosG']
        rows.append(row)
    return rows
