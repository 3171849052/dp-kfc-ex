"""Reuse Exp24 operators, with exactly one auxiliary batch per rebuild."""
import torch
import torch.nn.functional as F
from exp24.geometry import AOnlyOperator, FullKFACOperator
from .config import A_POWER, DAMPING


def spectrum(matrix):
    eig = torch.linalg.eigvalsh((matrix.double() + matrix.double().T) * .5).clamp_min(0)
    return {'min': eig.min().item(), 'max': eig.max().item(),
            'trace': eig.sum().item(),
            'damped_condition': ((eig.max()+DAMPING)/(eig.min()+DAMPING)).item()}


def build(model, x, y, kind):
    if kind == 'identity':
        return None, {'calibration_batches': 0, 'calibration_samples': 0}
    with torch.no_grad():
        features = model.backbone(x)
        a = torch.cat((features, torch.ones_like(features[:, :1])), dim=1)
        factor = {'A': a.T @ a / len(a), 'output_dimension': model.classifier.out_features}
    diagnostics = {'A': spectrum(factor['A']), 'calibration_batches': 1,
                   'calibration_samples': len(x), 'damping': DAMPING}
    if kind == 'a_only':
        operator = AOnlyOperator({'classifier': factor}, power=A_POWER, damping=DAMPING)
        diagnostics.update(p=A_POWER, rms_scale=operator.scale, **operator.moments)
    elif kind == 'full':
        with torch.no_grad():
            logits = model.classifier(features)
        anchor = logits.requires_grad_(True)
        b, = torch.autograd.grad(F.cross_entropy(anchor, y, reduction='sum'), anchor)
        factor['G'] = b.T @ b / len(b)
        diagnostics['G'] = spectrum(factor['G'])
        operator = FullKFACOperator({'classifier': factor}, damping=DAMPING)
    else:
        raise ValueError(kind)
    return operator, diagnostics
