import torch


def relative(a, b):
    return ((a-b).norm()/b.norm()).item()


def compare(factors, reference, damping):
    errors, whitening = [], []
    for name, f in factors.items():
        a, c = f['A'].double(), f['C'].double()
        ar, cr = reference[name]['A'].double(), reference[name]['C'].double()
        norm = ar.square().sum()*cr.square().sum()
        err = (a.square().sum()*c.square().sum()+norm-2*(a*ar).sum()*(c*cr).sum()).clamp_min(0)
        row = {'layer': name, 'kron_relative_error': (err/norm).sqrt().item()}
        for key, v, r in [('A', a, ar), ('C', c, cr)]:
            row[key+'_relative_error'] = relative(v, r)
            row[key+'_cosine'] = ((v*r).sum()/(v.norm()*r.norm())).item()
        errors.append(row)
        logs, conds, ranks = [], [], []
        for v, r in [(a, ar), (c, cr)]:
            e, q = torch.linalg.eigh(v)
            inv = (q*(e.clamp_min(0)+damping).rsqrt()) @ q.T
            t = inv @ r @ inv.T
            e = torch.linalg.eigvalsh((t+t.T)/2)
            # Explicit relative spectral floor; tiny diagnostic oracles are singular.
            floor = e[-1]*1e-7
            ranks.append(int((e > floor).sum()))
            e = e.clamp_min(floor)
            conds.append((e[-1]/e[0]).item())
            logs.append(e.log())
        whitening.append({'layer': name, 'condition_number': conds[0]*conds[1],
                          'log_eigenvalue_spread': (logs[0].var(unbiased=False)+logs[1].var(unbiased=False)).sqrt().item(),
                          'A_rank': ranks[0], 'A_dim': len(a), 'C_rank': ranks[1], 'C_dim': len(c),
                          'relative_spectral_floor': 1e-7})
    return errors, whitening
