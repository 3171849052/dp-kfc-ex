"""Local CUDA norm-kernel sweep; no training or automatic router fitting."""
import argparse
import csv
import itertools
from pathlib import Path
import sys
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
import torch
from torch import nn
from exp21.handlers import Record, sample_norm_squared
from exp21.routing import norm_route
from exp12.runtime import runtime


@torch.no_grad()
def old_ghost(record, tile):
    total = record.b.new_zeros(len(record.b))
    for j in range(0, record.b.shape[1], tile):
        for k in range(0, record.b.shape[1], tile):
            bg = record.b[:, j:j+tile] @ record.b[:, k:k+tile].transpose(1, 2)
            zg = record.z[:, j:j+tile] @ record.z[:, k:k+tile].transpose(1, 2)
            total.add_((bg*zg).sum((1, 2)))
    return total.clamp_min(0)


def measure(fn, repeats):
    for _ in range(3):
        fn()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end)/repeats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--tile', type=int, default=64)
    parser.add_argument('--max-fast-temp-bytes', type=int, default=256*2**20)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(21)
    rows = []
    with runtime('cuda:0'), torch.no_grad():
        for batch, tokens, din, dout in itertools.product((4, 16), (16, 64, 128, 512), (128, 512, 1024), (128, 512, 1024)):
            module = nn.Linear(din, dout).cuda()
            r = Record('', module, torch.randn(batch, tokens, din, device='cuda'),
                       torch.randn(batch, tokens, dout, device='cuda'), None,
                       {id(p) for p in module.parameters()})
            def fast():
                return sample_norm_squared(r.fast())
            torch.testing.assert_close(r.ghost(args.tile), fast(), rtol=1e-4, atol=2e-5)
            fast_ms = measure(fast, args.repeats)
            ghost_ms = measure(lambda: r.ghost(args.tile), args.repeats)
            old_ms = measure(lambda: old_ghost(r, args.tile), args.repeats)
            route = norm_route(r, 'auto', args.max_fast_temp_bytes, args.tile)
            faster = 'fast' if fast_ms < ghost_ms else 'ghost'
            rows.append(dict(B=batch, T=tokens, d_in=din, d_out=dout, tile=args.tile,
                             **route, fast_ms=fast_ms, ghost_ms=ghost_ms, old_ghost_ms=old_ms,
                             ghost_speedup=old_ms/ghost_ms, actual_faster=faster,
                             prediction_matches=route['strategy'] == faster,
                             selected_over_best=(fast_ms if route['strategy']=='fast' else ghost_ms)/min(fast_ms, ghost_ms)))
    output = ROOT/'exp21/results/router_benchmark.csv'
    with output.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    mismatches = [r for r in rows if not r['prediction_matches']]
    ratios = torch.tensor([r['ghost_speedup'] for r in rows])
    print(f'{len(rows)} cases; {len(mismatches)} mismatches; old/new Ghost speedup median={ratios.median():.3f}, range={ratios.min():.3f}..{ratios.max():.3f}')
    for r in sorted(mismatches, key=lambda r: r['selected_over_best'], reverse=True)[:10]:
        print({k:r[k] for k in ('B','T','d_in','d_out','strategy','actual_faster','selected_over_best')})
    print(output)


if __name__ == '__main__':
    main()
