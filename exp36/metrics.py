"""Matrix alignment in float64; entropy effective rank and projector overlap."""
import torch


def decompose(covariance):
    a = (covariance.double() + covariance.double().T)*.5
    values, vectors = torch.linalg.eigh(a)
    values = values.clamp_min(0)
    gains = (values+.001).pow(-.4)
    operator = (vectors*gains)@vectors.T
    probabilities = values[values > 0]/values.sum()
    effective_rank = (-torch.sum(probabilities*probabilities.log())).exp().item()
    return dict(A=a, values=values, vectors=vectors, P=operator,
                gains=gains, effective_rank=effective_rank)


def alignment(reference, estimate):
    return ((reference*estimate).sum()/(reference.norm()*estimate.norm())).item(), ((estimate-reference).norm()/reference.norm()).item()


def overlap(reference_vectors, estimate_vectors, k):
    # tr(Qr Qr^T Qe Qe^T)/k = ||Qr^T Qe||_F^2/k.
    return ((reference_vectors[:, -k:].T@estimate_vectors[:, -k:]).square().sum()/k).item()


def compare(reference, estimate):
    cos_a, rel_a = alignment(reference['A'], estimate['A'])
    cos_p, rel_p = alignment(reference['P'], estimate['P'])
    result = dict(cos_A=cos_a, rel_frob_A=rel_a, cos_P=cos_p, rel_frob_P=rel_p,
                  trace_ratio=(estimate['A'].trace()/reference['A'].trace()).item(),
                  effective_rank=estimate['effective_rank'], effective_rank_ref=reference['effective_rank'])
    for k in (16, 32):
        result[f'top{k}_overlap'] = overlap(reference['vectors'], estimate['vectors'], k)
    for prefix, item in (('', estimate), ('ref_', reference)):
        gain = item['gains']
        for p in (10, 50, 90, 99):
            result[f'{prefix}operator_gain_p{p}'] = gain.quantile(p/100).item()
        result[f'{prefix}operator_gain_min'] = gain.min().item()
        result[f'{prefix}operator_gain_max'] = gain.max().item()
    return result


def family(name):
    if name == 'patch_embed':
        return 'patch_embed', -1
    if name == 'head':
        return 'head', -1
    block = int(name.split('.')[1])
    suffix = '.'.join(name.split('.')[2:])
    group = {'attn.q_proj':'attention_qkv', 'attn.k_proj':'attention_qkv',
             'attn.v_proj':'attention_qkv', 'attn.out_proj':'attention_out',
             'mlp.fc1':'mlp_fc1', 'mlp.fc2':'mlp_fc2'}[suffix]
    return group, block
