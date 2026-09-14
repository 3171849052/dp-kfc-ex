"""Relative-floor geometry is invariant to the global matched scale."""
from exp14b.geometry import diagnose as power_diagnose, spectrum_stats


def diagnose(factors, estimator, beta, seed, epoch):
    return [dict(estimator=estimator, **row)
            for row in power_diagnose(factors, beta, seed, epoch)]
