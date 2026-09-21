"""Exact Exp30 no-scale KFC-A reference with explicit synthetic chunk size."""
from exp22 import geometry as base
from exp33 import config as cfg
base.SYNTHETIC_PHYSICAL_BATCH_SIZE = cfg.SYNTHETIC_PHYSICAL_BATCH_SIZE
synthetic_stream = base.synthetic_stream

def remove_a_scale(operator, builder):
    """Use raw (A + damping I)^(-power), including unscaled diagnostics."""
    scale = operator.scale
    operator.scale = 1.0
    operator.moments["scale_match"] = 1.0
    for name in operator.diagnostics:
        if name.startswith("operator_gain_"):
            operator.diagnostics[name] /= scale
        elif name.startswith("transformed_eig_"):
            operator.diagnostics[name] /= scale ** 2
    builder.update(operator.moments)
    builder.update(operator.diagnostics)


def build_reference(model, seed, epoch, device):
    operator, stats = base.build_from_batches(
        model, "dp_kfc_a_bk", synthetic_stream(seed, epoch, device),
        seed, epoch, damping=0.1, power=0.4)
    remove_a_scale(operator, stats)
    return operator, stats
