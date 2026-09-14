"""Factorwise damped powers, with Exp13's augmented-bias interface."""
import torch
from exp13.operator import Operator as BaseOperator, inverse_sqrt
from exp14.config import DAMPING


def inverse_power(factor, beta):
    if beta == 0:
        return torch.eye(len(factor), device=factor.device, dtype=torch.float32)
    if beta == .5:
        return inverse_sqrt(factor)
    factor = factor.double()
    eigenvalues, q = torch.linalg.eigh(
        (factor + factor.T) / 2 + DAMPING * torch.eye(len(factor), device=factor.device))
    return ((q * eigenvalues.pow(-beta)) @ q.T).float()


class Operator(BaseOperator):
    def __init__(self, factors, beta):
        self.data = {name: (inverse_power(f['C'], beta), inverse_power(f['A'], beta))
                     for name, f in factors.items()}
