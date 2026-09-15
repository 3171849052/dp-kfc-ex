# Exp19 results

SMOKE ONLY: one zero-noise batch, one seed. No formal utility or performance conclusion.

All clipping and layerwise norm statistics are unnoised research diagnostics, not a DP release.

## Attribution chain

### M1 − M0: Ghost backend

Final accuracy: 0.175781 → 0.175781; paired delta +0.0000 pp; 95% CI [nan, nan] pp.
Algorithm runtime: 0.194 → 0.203 s (-4.79% reduction). Private training: 0.065 → 0.070 s (-8.66% reduction).
Peak allocated memory: 811.03 → 152.69 MiB; train peak: 811.03 → 131.38 MiB (+83.80% reduction).
Mean epoch clipping fraction: 1.000000 → 1.000000; mean epoch norm p99: 1201.978149 → 1201.978149.

### M2 − M1: Remove C (including its scaling effect)

Final accuracy: 0.175781 → 0.187500; paired delta +1.1719 pp; 95% CI [nan, nan] pp.
Algorithm runtime: 0.203 → 0.196 s (+3.55% reduction). Private training: 0.070 → 0.077 s (-9.89% reduction).
Peak allocated memory: 152.69 → 159.27 MiB; train peak: 131.38 → 159.27 MiB (-21.23% reduction).
Mean epoch clipping fraction: 1.000000 → 1.000000; mean epoch norm p99: 1201.978149 → 72.315376.

### M3 − M2: Power .5 → .25 with self-contained A RMS matching

Final accuracy: 0.187500 → 0.214844; paired delta +2.7344 pp; 95% CI [nan, nan] pp.
Algorithm runtime: 0.196 → 0.200 s (-2.09% reduction). Private training: 0.077 → 0.080 s (-3.93% reduction).
Peak allocated memory: 159.27 → 159.27 MiB; train peak: 159.27 → 159.27 MiB (+0.00% reduction).
Mean epoch clipping fraction: 1.000000 → 1.000000; mean epoch norm p99: 72.315376 → 34.245144.

### M3 − M0: Headline M3 versus M0

Final accuracy: 0.175781 → 0.214844; paired delta +3.9062 pp; 95% CI [nan, nan] pp.
Algorithm runtime: 0.194 → 0.200 s (-3.18% reduction). Private training: 0.065 → 0.080 s (-24.10% reduction).
Peak allocated memory: 811.03 → 159.27 MiB; train peak: 811.03 → 159.27 MiB (+80.36% reduction).
Mean epoch clipping fraction: 1.000000 → 1.000000; mean epoch norm p99: 1201.978149 → 34.245144.

## Forward-only curvature budget

- M0_original: per-epoch forward/VJP/reverse-vector/sample counts 1/1/256/256; mean backward 0.001078s; build 0.129117s.
- M1_full_ghost: per-epoch forward/VJP/reverse-vector/sample counts 1/1/256/256; mean backward 0.000990s; build 0.132799s.
- M2_a_ghost_p05: per-epoch forward/VJP/reverse-vector/sample counts 1/0/0/256; mean backward 0.000000s; build 0.118648s.
- M3_a_ghost_p025: per-epoch forward/VJP/reverse-vector/sample counts 1/0/0/256; mean backward 0.000000s; build 0.119706s.
- M2_a_ghost_p05 versus M1: build time reduction +10.66%; curvature backward saved 0.000990 s/epoch (100%); 256 reverse vectors eliminated per epoch.
- M3_a_ghost_p025 versus M1: build time reduction +9.86%; curvature backward saved 0.000990 s/epoch (100%); 256 reverse vectors eliminated per epoch.

M2/M3 eliminate all synthetic reverse vectors and curvature backward calls. Actual build savings are reported above and include A accumulation/eigendecomposition.

Accuracy AUC integrates observed epoch 1–5 accuracies. Memory peaks exclude evaluation; reserved memory includes allocator caching within each fresh process. Private timing excludes diagnostic CPU transfer and postprocessing. diagnostic_postprocess_seconds is excluded from algorithm time (build + actual DP training). Internal breakdown uses deferred CUDA Events with no instrumentation synchronization inside batches. Each fresh subprocess uses disposable 256-sample action-path warmup and deterministic balanced method order. Allocated is the primary memory comparison; reserved is an allocator behavior diagnostic. Internal Exact/Ghost phases are explanatory, not cross-backend rankings. M3−M2 identifies power plus the required RMS matching jointly. Identity C is undamped identity. RDP accounting follows the repository shuffled fixed-batch convention; research CSVs are not privacy-protected releases.
