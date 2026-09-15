# Exp19 results

SMOKE ONLY: one zero-noise batch, one seed. No formal utility or performance conclusion.

All clipping and layerwise norm statistics are unnoised research diagnostics, not a DP release.

## Attribution chain

### M1 − M0: Ghost backend

Final accuracy: 0.175781 → 0.175781; paired delta +0.0000 pp; 95% CI [nan, nan] pp.
Algorithm runtime: 0.504 → 0.733 s (-45.52% reduction). Private training: 0.075 → 0.124 s (-65.58% reduction).
Peak allocated memory: 810.56 → 152.69 MiB; train peak: 810.56 → 131.58 MiB (+83.77% reduction).
Mean epoch clipping fraction: 1.000000 → 1.000000; mean epoch norm p99: 1201.978149 → 1201.978149.

### M2 − M1: Remove C (including its scaling effect)

Final accuracy: 0.175781 → 0.187500; paired delta +1.1719 pp; 95% CI [nan, nan] pp.
Algorithm runtime: 0.733 → 0.481 s (+34.38% reduction). Private training: 0.124 → 0.123 s (+1.08% reduction).
Peak allocated memory: 152.69 → 159.27 MiB; train peak: 131.58 → 159.27 MiB (-21.04% reduction).
Mean epoch clipping fraction: 1.000000 → 1.000000; mean epoch norm p99: 1201.978149 → 72.315376.

### M3 − M2: Power .5 → .25 with self-contained A RMS matching

Final accuracy: 0.187500 → 0.214844; paired delta +2.7344 pp; 95% CI [nan, nan] pp.
Algorithm runtime: 0.481 → 0.480 s (+0.21% reduction). Private training: 0.123 → 0.125 s (-1.73% reduction).
Peak allocated memory: 159.27 → 159.27 MiB; train peak: 159.27 → 159.27 MiB (+0.00% reduction).
Mean epoch clipping fraction: 1.000000 → 1.000000; mean epoch norm p99: 72.315376 → 34.245144.

### M3 − M0: Headline M3 versus M0

Final accuracy: 0.175781 → 0.214844; paired delta +3.9062 pp; 95% CI [nan, nan] pp.
Algorithm runtime: 0.504 → 0.480 s (+4.71% reduction). Private training: 0.075 → 0.125 s (-66.63% reduction).
Peak allocated memory: 810.56 → 159.27 MiB; train peak: 810.56 → 159.27 MiB (+80.35% reduction).
Mean epoch clipping fraction: 1.000000 → 1.000000; mean epoch norm p99: 1201.978149 → 34.245144.

## Forward-only curvature budget

- M0_original: per-epoch forward/VJP/reverse-vector/sample counts 1/1/256/256; mean backward 0.066148s; build 0.428942s.
- M1_full_ghost: per-epoch forward/VJP/reverse-vector/sample counts 1/1/256/256; mean backward 0.099900s; build 0.609212s.
- M2_a_ghost_p05: per-epoch forward/VJP/reverse-vector/sample counts 1/0/0/256; mean backward 0.000000s; build 0.358478s.
- M3_a_ghost_p025: per-epoch forward/VJP/reverse-vector/sample counts 1/0/0/256; mean backward 0.000000s; build 0.355353s.
- M2_a_ghost_p05 versus M1: build time reduction +41.16%; curvature backward saved 0.099900 s/epoch (100%); 256 reverse vectors eliminated per epoch.
- M3_a_ghost_p025 versus M1: build time reduction +41.67%; curvature backward saved 0.099900 s/epoch (100%); 256 reverse vectors eliminated per epoch.

M2/M3 eliminate all synthetic reverse vectors and curvature backward calls. Actual build savings are reported above and include A accumulation/eigendecomposition.

Accuracy AUC integrates observed epoch 1–5 accuracies. Memory peaks exclude evaluation; reserved memory includes allocator caching within each fresh process. Private timing includes the common research diagnostics. Internal Exact/Ghost phases are explanatory, not cross-backend rankings. M3−M2 identifies power plus the required RMS matching jointly. Identity C is undamped identity. RDP accounting follows the repository shuffled fixed-batch convention; research CSVs are not privacy-protected releases.
