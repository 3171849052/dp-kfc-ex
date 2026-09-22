"""Spectral interpolation, with the Exp35 raw operator and normalization."""
ALPHAS = {'alpha025': .25, 'alpha075': .75}
GPU_BY_VARIANT = {'alpha025': 1, 'alpha075': 2}


def install():
    from exp35 import geometry
    original = geometry.spectral_operator

    def spectral_operator(covariance, variant):
        alpha = ALPHAS[variant]
        matrix, gain = original(covariance, 'norm')
        import torch
        matrix = (1-alpha)*torch.eye(len(matrix), device=matrix.device, dtype=matrix.dtype) + alpha*matrix
        gain = (1-alpha) + alpha*gain
        assert gain.min().item() >= 1-alpha-geometry.TOL
        assert gain.max().item() <= 1+geometry.TOL
        return matrix, gain

    geometry.spectral_operator = spectral_operator
    return spectral_operator
