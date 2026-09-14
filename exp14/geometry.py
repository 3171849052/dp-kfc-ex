"""Covariance spectra from factors only; no Kronecker matrix."""
import torch
from exp14.config import DAMPING


def spectrum_stats(factor, beta):
    factor = factor.double()
    eigenvalues = torch.linalg.eigvalsh((factor + factor.T) / 2)
    # Estimated PSD factors can have small negative roundoff eigenvalues.
    eigenvalues = eigenvalues.clamp_min(0)
    transformed = eigenvalues * (eigenvalues + DAMPING).pow(-2 * beta)
    singular = bool((transformed == 0).any())
    condition = float('inf') if singular else (transformed.max() / transformed.min()).item()
    spread = float('inf') if singular else transformed.log().std(correction=0).item()
    return condition, spread


def diagnose(factors, beta, seed, epoch):
    rows = []
    for name, f in factors.items():
        ka, sa = spectrum_stats(f['A'], beta)
        kc, sc = spectrum_stats(f['C'], beta)
        rows.append(dict(beta=beta, seed=seed, epoch=epoch, layer=name,
                         kappa_A=ka, kappa_C=kc, kappa_block=ka * kc,
                         A_log_eigenvalue_spread=sa, C_log_eigenvalue_spread=sc))
    return rows
