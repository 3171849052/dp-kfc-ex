# Exp19 results

Five paired seeds; descriptive estimates and paired percentile bootstrap intervals. No strong significance claims.

All clipping and layerwise norm statistics are unnoised research diagnostics, not a DP release.

## Attribution chain

### M1 − M0: Ghost backend

Final accuracy: 0.949120 → 0.949060; paired delta -0.0060 pp; 95% CI [-0.0140, 0.0000] pp.
Algorithm runtime: 39.007 → 44.462 s (-13.98% reduction). Private training: 38.164 → 43.617 s (-14.29% reduction).
Peak allocated memory: 812.51 → 159.58 MiB; train peak: 812.51 → 132.86 MiB (+83.65% reduction).
Mean epoch clipping fraction: 0.308284 → 0.308281; mean epoch norm p99: 11663.278525 → 11663.948379.

### M2 − M1: Remove C (including its scaling effect)

Final accuracy: 0.949060 → 0.960800; paired delta +1.1740 pp; 95% CI [1.0560, 1.3620] pp.
Algorithm runtime: 44.462 → 44.407 s (+0.12% reduction). Private training: 43.617 → 43.617 s (-0.00% reduction).
Peak allocated memory: 159.58 → 160.75 MiB; train peak: 132.86 → 160.75 MiB (-20.99% reduction).
Mean epoch clipping fraction: 0.308281 → 0.271424; mean epoch norm p99: 11663.948379 → 744.954581.

### M3 − M2: Power .5 → .25 with self-contained A RMS matching

Final accuracy: 0.960800 → 0.960520; paired delta -0.0280 pp; 95% CI [-0.0800, 0.0220] pp.
Algorithm runtime: 44.407 → 44.085 s (+0.73% reduction). Private training: 43.617 → 43.307 s (+0.71% reduction).
Peak allocated memory: 160.75 → 160.75 MiB; train peak: 160.75 → 160.75 MiB (+0.00% reduction).
Mean epoch clipping fraction: 0.271424 → 0.254286; mean epoch norm p99: 744.954581 → 628.428291.

### M3 − M0: Headline M3 versus M0

Final accuracy: 0.949120 → 0.960520; paired delta +1.1400 pp; 95% CI [0.9900, 1.3260] pp.
Algorithm runtime: 39.007 → 44.085 s (-13.02% reduction). Private training: 38.164 → 43.307 s (-13.48% reduction).
Peak allocated memory: 812.51 → 160.75 MiB; train peak: 812.51 → 160.75 MiB (+80.22% reduction).
Mean epoch clipping fraction: 0.308284 → 0.254286; mean epoch norm p99: 11663.278525 → 628.428291.

## Forward-only curvature budget

- M0_original: per-epoch forward/VJP/reverse-vector/sample counts 10/10/2560/2560; mean backward 0.004226s; build 0.168634s.
- M1_full_ghost: per-epoch forward/VJP/reverse-vector/sample counts 10/10/2560/2560; mean backward 0.004438s; build 0.168994s.
- M2_a_ghost_p05: per-epoch forward/VJP/reverse-vector/sample counts 10/0/0/2560; mean backward 0.000000s; build 0.158091s.
- M3_a_ghost_p025: per-epoch forward/VJP/reverse-vector/sample counts 10/0/0/2560; mean backward 0.000000s; build 0.155625s.
- M2_a_ghost_p05 versus M1: build time reduction +6.45%; curvature backward saved 0.004438 s/epoch (100%); 2560 reverse vectors eliminated per epoch.
- M3_a_ghost_p025 versus M1: build time reduction +7.91%; curvature backward saved 0.004438 s/epoch (100%); 2560 reverse vectors eliminated per epoch.

M2/M3 eliminate all synthetic reverse vectors and curvature backward calls. Actual build savings are reported above and include A accumulation/eigendecomposition.

Accuracy AUC integrates observed epoch 1–5 accuracies. Memory peaks exclude evaluation; reserved memory includes allocator caching within each fresh process. Private timing excludes diagnostic CPU transfer and postprocessing. diagnostic_postprocess_seconds is excluded from algorithm time (build + actual DP training). Internal breakdown uses deferred CUDA Events with no instrumentation synchronization inside batches. Each fresh subprocess uses disposable 256-sample action-path warmup and deterministic balanced method order. Allocated is the primary memory comparison; reserved is an allocator behavior diagnostic. Internal Exact/Ghost phases are explanatory, not cross-backend rankings. M3−M2 identifies power plus the required RMS matching jointly. Identity C is undamped identity. RDP accounting follows the repository shuffled fixed-batch convention; research CSVs are not privacy-protected releases.
