"""CUDA benchmark for full/chunked affine Fast and row-tiled Ghost."""
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
from exp21.handlers import Record
from exp21.routing import norm_route
from exp12.runtime import runtime


@torch.no_grad()
def old_row_ghost(record, tile):
    """The previous single-axis implementation, including its T-1 split."""
    total = record.b.new_zeros(len(record.b))
    rows = min(tile, max(1, record.b.shape[1]-1))
    for j in range(0, record.b.shape[1], rows):
        bg = record.b[:, j:j+rows] @ record.b.transpose(1, 2)
        zg = record.z[:, j:j+rows] @ record.z.transpose(1, 2)
        total.add_((bg*zg).sum((1, 2)))
    return total.clamp_min(0)


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
    for _ in range(2):
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
    base = list(itertools.product((4, 16), (16, 64, 128, 512),
                                  (128, 512, 1024), (128, 512, 1024)))
    large = [(16, t, 4096, 4096) for t in (128, 512)]
    cases = base + large
    rows = []
    with runtime('cuda:0'), torch.no_grad():
        for batch, tokens, din, dout in cases:
            module = nn.Linear(din, dout).cuda()
            r = Record('', module, torch.randn(batch, tokens, din, device='cuda'),
                       torch.randn(batch, tokens, dout, device='cuda'), None,
                       {id(p) for p in module.parameters()})
            route = norm_route(r, 'auto', args.max_fast_temp_bytes, args.tile)
            full_feasible = route['estimated_full_fast_bytes'] <= args.max_fast_temp_bytes
            chunk_feasible = route['candidate_output_chunk_size'] > 0
            full_ms = measure(r.fast, args.repeats) if full_feasible else None
            chunk = route['candidate_output_chunk_size']
            chunk_ms = measure(lambda: r.chunked_norm(chunk), args.repeats) if chunk_feasible else None
            ghost_ms = measure(lambda: r.ghost(args.tile), args.repeats)
            old_row_ms = measure(lambda: old_row_ghost(r, args.tile), args.repeats)
            old_double_ms = measure(lambda: old_ghost(r, args.tile), args.repeats)
            old_ms = old_row_ms if tokens <= args.tile else old_double_ms
            feasible = {k: v for k, v in {
                'full_fast': full_ms, 'chunked_fast': chunk_ms, 'ghost': ghost_ms
            }.items() if v is not None}
            actual_best = min(feasible, key=feasible.get)
            selected_name = route['strategy']
            selected_ms = feasible.get(selected_name, ghost_ms)
            rows.append(dict(
                B=batch, T=tokens, d_in=din, d_out=dout, tile=args.tile,
                strategy=route['strategy'], router_choice=route['strategy'],
                estimated_fast_bytes=route['estimated_fast_bytes'],
                estimated_full_fast_bytes=route['estimated_full_fast_bytes'],
                estimated_chunked_fast_bytes=route['estimated_chunked_fast_bytes'],
                estimated_fast_cost=route['estimated_fast_cost'],
                estimated_full_fast_cost=route['estimated_full_fast_cost'],
                estimated_chunked_fast_cost=route['estimated_chunked_fast_cost'],
                estimated_ghost_cost=route['estimated_ghost_cost'],
                estimated_fast_macs=route['estimated_fast_macs'],
                estimated_ghost_macs=route['estimated_ghost_macs'],
                routing_reason=route['routing_reason'], output_chunk_size=chunk,
                full_fast_ms=full_ms, chunked_fast_ms=chunk_ms,
                ghost_ms=ghost_ms, old_ghost_ms=old_ms,
                old_row_ghost_ms=old_row_ms, old_double_ghost_ms=old_double_ms,
                # Legacy aliases kept in the CSV for existing analysis files.
                fast_ms=full_ms, ghost_speedup=old_ms/ghost_ms,
                actual_faster=actual_best, actual_best_feasible=actual_best,
                prediction_matches=route['strategy'] == actual_best,
                selected_over_best=selected_ms/min(feasible.values())))
            del r, module
            torch.cuda.empty_cache()
    output = ROOT/'exp21/results/router_benchmark.csv'
    with output.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    mismatches = [r for r in rows if not r['prediction_matches']]
    ratios = torch.tensor([r['ghost_speedup'] for r in rows])
    print(f'{len(rows)} cases; {len(mismatches)} mismatches; '
          f'old/new Ghost speedup median={ratios.median():.3f}, '
          f'range={ratios.min():.3f}..{ratios.max():.3f}')
    for r in sorted(mismatches, key=lambda r: r['selected_over_best'], reverse=True)[:10]:
        print({k: r[k] for k in ('B', 'T', 'd_in', 'd_out', 'router_choice',
                                  'actual_best_feasible', 'selected_over_best')})
    print(output)


if __name__ == '__main__':
    main()
