"""Covariance spectra from factors only; no Kronecker matrix."""
import torch
from exp14.config import DAMPING


def spectrum_stats(factor, beta):
    factor = factor.double()
    eigenvalues = torch.linalg.eigvalsh((factor + factor.T) / 2)
    # Estimated PSD factors can have small negative roundoff eigenvalues.
    eigenvalues = eigenvalues.clamp_min(0)
    transformed = eigenvalues * (eigenvalues + DAMPING).pow(-2 * beta)
    maximum = transformed.max()
    floor = 1e-7 * maximum
    stats = dict(zero_eigenvalue_count=(transformed == 0).sum().item(),
                 dimension=len(transformed),
                 effective_rank=(transformed > floor).sum().item())
    if maximum == 0:
        return dict(**stats, floored_condition_number=float('nan'),
                    log_eigenvalue_spread=float('nan'), spectral_floor_ratio=float('nan'))
    floored = transformed.clamp_min(floor)
    return dict(**stats,
                floored_condition_number=(floored.max() / floored.min()).item(),
                log_eigenvalue_spread=floored.log().std(correction=0).item(),
                spectral_floor_ratio=(floor / maximum).item())


def diagnose(factors, beta, seed, epoch):
    rows = []
    for name, f in factors.items():
        a = spectrum_stats(f['A'], beta)
        c = spectrum_stats(f['C'], beta)
        rows.append(dict(beta=beta, seed=seed, epoch=epoch, layer=name,
                         **{f'A_{key}': value for key, value in a.items()},
                         **{f'C_{key}': value for key, value in c.items()},
                         floored_kappa_block=a['floored_condition_number'] * c['floored_condition_number']))
    return rows
