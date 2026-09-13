"""Prediction statistics on the supplied input cache, without regenerating probes."""
import math
import torch


@torch.no_grad()
def synthetic_state_metrics(model, cache):
    entropy = confidence = 0.
    count = 0
    for x in cache:
        logp = model(x).double().log_softmax(-1)
        p = logp.exp()
        entropy += (-(p*logp).sum()).item()
        confidence += p.max(-1).values.sum().item()
        count += len(x)
    entropy /= count
    return dict(synthetic_mean_max_probability=confidence/count,
                synthetic_mean_prediction_entropy=entropy,
                synthetic_normalized_entropy=entropy/math.log(10),
                synthetic_mean_kl_to_uniform=math.log(10)-entropy)
