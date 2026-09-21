"""Unmodified Exp22 operators, Exp30 no-scale A, and read-only spectra."""
from statistics import median
import torch
from exp22.geometry import build_from_batches, synthetic_stream
from exp22.methods import _group
from exp34 import config as cfg

GROUPS = ("attention_qkv", "attention_out", "mlp", "patch_head")
SPECTRUM_FIELDS = [f"{side}_{field}" for side in ("A", "G") for field in
                   ("trace_per_dim", "eig_p10", "eig_p50", "eig_p90", "eig_p99", "eig_min", "eig_max")]
GEOMETRY_COLUMNS = ["epoch", "layer", "group", "method", "damping", *SPECTRUM_FIELDS,
                    "relative_damping_A", "relative_damping_G"]


def remove_a_scale(operator, builder):
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


@torch.no_grad()
def spectrum_rows(operator, method, damping, epoch):
    rows = []
    if operator is None:
        return rows
    for name, factors in operator.factors.items():
        row = dict(epoch=epoch, layer=name, group=_group(name), method=method, damping=damping)
        for side in ("A", "G"):
            if side not in factors:
                continue
            covariance = factors[side].double()
            eigenvalues = torch.linalg.eigvalsh((covariance + covariance.T) * 0.5)
            trace = covariance.diagonal().mean().item()
            row[f"{side}_trace_per_dim"] = trace
            row[f"relative_damping_{side}"] = damping / trace if trace != 0 else float("inf")
            quantiles = eigenvalues.quantile(eigenvalues.new_tensor([.1, .5, .9, .99])).tolist()
            row.update({f"{side}_eig_p{p}": v for p, v in zip((10, 50, 90, 99), quantiles)})
            row[f"{side}_eig_min"] = eigenvalues.min().item()
            row[f"{side}_eig_max"] = eigenvalues.max().item()
        rows.append(row)
    return rows


def build_geometry(model, method, damping, seed, epoch, device):
    operator, stats = build_from_batches(
        model, cfg.EXP22_METHOD[method],
        synthetic_stream(seed, epoch, device, batches=cfg.SYNTHETIC_BATCHES,
                         batch_size=cfg.SYNTHETIC_BATCH_SIZE, alpha=cfg.SYNTHETIC_ALPHA),
        seed, epoch, damping=damping, power=cfg.A_POWER)
    if method == "dp_kfc_a":
        remove_a_scale(operator, stats)
    rows = spectrum_rows(operator, method, damping, epoch)
    for group in GROUPS:
        for side in ("A", "G"):
            for field in (f"{side}_trace_per_dim", f"relative_damping_{side}"):
                values = [r[field] for r in rows if r["group"] == group and field in r]
                stats[f"{group}_median_{field}"] = median(values) if values else None
    return operator, stats, rows
