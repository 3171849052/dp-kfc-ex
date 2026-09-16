# Exp21 BK optimization sanity results

GPU: NVIDIA GeForce RTX 3080 Ti, FP32; conda environment `curve`. Original implementation and final implementation each ran the existing fresh-process MNIST smoke (seed 42, 3 batches of 256). These are single smoke observations, not repeated performance estimates. No formal 25-job experiment was started.

## Correctness

`conda run -n curve python -B exp21/test_exp21.py`: **147 passed**, retaining all original numerical tolerances. The original 117 cases remain; obsolete whole-model VJP/forced tied-strategy assertions were updated for the requested execution changes. New cases cover bounded local VJP targets/chunks, stochastic replay, streaming baselines, exact row Gram, frozen subsets, allocation shapes, shared guards, 50,003-token ties, and seven-step memory stability.

BERT/GPT-2/LLaMA remain fallback=0, backward_calls=1 for BK/BK+GD. TinyViT identity and partial-A gradients/norms match independent per-example backward; B=2, K=2 uses 3 reverse calls (capture + one fallback VJP + fallback reconstruction). Its fallback has 320 parameter elements and a 2,560-byte VJP gradient workspace.

## MNIST smoke: original versus final

Wall time includes the same data loading, transfer, clipping and optimizer/noise work as before. Peak allocated memory uses the unchanged CUDA phase measurements.

| Method | Old ms/batch | New ms/batch | Time change | Old peak MiB | New peak MiB | Peak change |
|---|---:|---:|---:|---:|---:|---:|
| exact | 56.723 | 56.666 | -0.1% | 810.58 | 810.58 | +0.0% |
| fast2 | 38.917 | 57.488 | +47.7% | 663.73 | 259.89 | -60.8% |
| ghost2 | 55.186 | 60.041 | +8.8% | 127.00 | 150.39 | +18.4% |
| bk | 54.647 | 56.094 | +2.6% | 127.00 | 127.01 | +0.0% |
| bk_gd | 37.846 | 54.985 | +45.3% | 138.46 | 126.21 | -8.8% |

Final CUDA event times below are divided by the three batches. Fast2/Ghost2 perform norm work inside the first reverse pass; that work is included in first-pass time, not hidden by removing it from the step timer. Their separate norm phase is clipping-factor bookkeeping. The sparse/shared/tied completion also finishes before pass one ends.

| Method | First pass ms | Norm ms | Second pass ms | BK reconstruction ms | A transform ms | Temporary sample gradients MiB | First-pass parameter grads |
|---|---:|---:|---:|---:|---:|---:|---:|
| exact | 4.8934 | 10.2001 | 0.0000 | 0.0000 | 0.0000 | 202.0723 | 8 |
| fast2 | 7.6216 | 0.1086 | 2.2067 | 0.0000 | 0.1434 | 196.1250 | 0 |
| ghost2 | 13.1304 | 0.0092 | 1.2875 | 0.0000 | 0.1010 | 0.0000 | 0 |
| bk | 2.5996 | 4.8954 | 0.0000 | 0.7734 | 0.0000 | 4.5312 | 8 |
| bk_gd | 1.6848 | 5.2944 | 0.0000 | 0.7690 | 0.0000 | 4.5312 | 0 |

Fast2/Ghost2 record `bk_cache_bytes=0`, `first_pass_param_grad_disabled=True`, and `gd_applied=False`. BK+GD alone sets the formal GD flag. The fused Fast norm directly reduces strided sample gradients without a flatten copy or a full-size square tensor.

## Router and Ghost kernel

`conda run -n curve python -B exp21/benchmark_router.py` sweeps all 72 requested shapes and writes `router_benchmark.csv`. It measures Fast norm, row-tiled Ghost, and the original double-tiled Ghost on the same inputs, with CUDA events and 20 repetitions.

Final router mismatches: **0/72**. The fused Fast kernel wins all shapes in this particular sweep; compute-selected Ghost remains useful for T=1 SimpleCNN projections, and the hard memory cap forces Ghost independently of timing.

| T | Median old Ghost ms | Median row Ghost ms | Median paired speedup (old/new) |
|---:|---:|---:|---:|
| 16 | 0.1602 | 0.2610 | 0.610x |
| 64 | 0.1949 | 0.2906 | 0.637x |
| 128 | 0.6416 | 0.3177 | 2.033x |
| 512 | 9.2175 | 1.1570 | 7.830x |

Short sequences are slower in row Ghost: avoiding a full [B,T,T] matrix requires at least two row tiles when T>1 and tile>=T. Long sequences benefit from reducing Python loops from quadratic to linear in the number of tiles. In-place Gram multiplication avoids a third Gram buffer.

Router rule: first prohibit dense Fast when `B*d_aug*d_out*element_size > max_fast_temp_bytes` (default 256 MiB). Otherwise compare effective costs `B*d_aug*d_out*(T+8)+L` and `8*B*T*T*(d_aug+d_out)+n_row_tiles*L`, with `L=4.4e8` equivalent MACs. Reduction/skinny-GEMM/launch constants are simple manual calibration on this GPU; metadata also records the raw MAC counts. Norm stays Fast; Embedding uses vectorized sparse rows; tied heads use their own exact router. SimpleCNN remains conv1/conv2=Fast, fc1/fc2=Ghost.

## Bounded fallback and large vocabulary

Fallback VJP targets only fallback parameters and processes at most K examples per reverse call. Temporary returned parameter gradients scale as **O(K*M_fallback)**, replacing O(B*M_model); default K=2. BK layers retain direct reconstruction. The replayed weighted reverse requests only fallback parameters for BK/BK+GD. A generic shared component exceeding the configured per-sample guard joins fallback before any dense sample gradient is made.

Embedding norm uses one batched composite-key unique/index_add aggregation, including repeats and padding, with no Python per-example sparse loop. Tied head routing compares `1.25*B*T*V*d_aug` with `B*T*T*(V+d_aug)`, checks output-chunk capacity, and never chooses dense Fast. Only observed token rows contribute to the exact cross term.

The large-vocabulary oracle test uses V=50,003, B=2, d=3, T=9, output chunk=7: the chunk gradient is at most **168 bytes**, versus **1,200,072 bytes** for [B,V,d]. With the test token range and padding, merged token gradients plus cross-term rows occupy at most 180 bytes. These are analytical workspace bounds checked through runtime metadata/allocation guards, not whole-step CUDA peaks. A separate large-vocabulary test exercises Ghost under a 16-byte Fast cap; both routes match the independent oracle. Final [V,d] aggregate gradients remain necessary.

## Limits

- The router constants are GPU/kernel dependent; re-run the synthetic sweep after hardware or kernel changes. The smoke is only three batches, so wall-time deltas include substantial host/launch variability.
- Row Ghost trades larger O(B*tile*T) workspace for fewer launches, and can increase peak memory versus the old O(B*tile²) kernel.
- The fallback bound covers parameter-gradient tensors, not the retained forward graph or batched reverse activation workspace; smaller K trades memory for more reverse calls.
- Generic small shared components retain the exact per-sample reference loop. Non-analytic multi-owner Embedding patterns require explicit fallback registration unless the memory guard routes them.
- Existing supported scope remains FP32, sample-independent models with tensor inputs; checkpointing, cross-example operations and autocast are outside this implementation.

Core changes: `handlers.py`, `bk.py`, `routing.py`, `fallback.py`, `methods.py`; `run_one.py` only exports the additional requested metadata. Tests, router benchmark and algorithm documentation were extended. No geometry, training protocol, result-directory organization, or paper-analysis scripts were changed.
