# Inverse versus forward (single seed 42)

Research-only: unnoised private diagnostics; not a DP release.

Beta magnitude, lambda, per-layer trace, total expected noise energy, privacy budget and training protocol are matched.
Forward gives large-A directions larger clipping radii and noise standard deviations; inverse gives them smaller ones.
Equal beta has equal condition magnitude for the SAME A, with reversed spectral ordering. Trained models may yield different A.
Conditions below are epoch-five medians; AUC uses the inherited epoch-1-to-5 trapezoidal definition.
Single-seed observations do not establish statistical significance.

- public, beta=0.1: final-accuracy winner=forward; inverse−forward accuracy=-0.0077, AUC=-0.0629, clip fraction=+0.0505, mean clip factor=-0.0439, update relative error=+0.0076, update cosine=-0.0009.
- pink, beta=0.1: final-accuracy winner=forward; inverse−forward accuracy=-0.0019, AUC=-0.0175, clip fraction=+0.0139, mean clip factor=-0.0118, update relative error=+0.0050, update cosine=-0.0003.
- public, beta=0.2: final-accuracy winner=forward; inverse−forward accuracy=-0.0145, AUC=-0.1397, clip fraction=+0.1135, mean clip factor=-0.0971, update relative error=+0.0206, update cosine=-0.0019.
- pink, beta=0.2: final-accuracy winner=forward; inverse−forward accuracy=-0.0053, AUC=-0.0398, clip fraction=+0.0306, mean clip factor=-0.0266, update relative error=+0.0109, update cosine=-0.0006.
- public, beta=0.3: final-accuracy winner=forward; inverse−forward accuracy=-0.0231, AUC=-0.2541, clip fraction=+0.1960, mean clip factor=-0.1687, update relative error=+0.0525, update cosine=-0.0029.
- pink, beta=0.3: final-accuracy winner=forward; inverse−forward accuracy=-0.0084, AUC=-0.0771, clip fraction=+0.0578, mean clip factor=-0.0490, update relative error=+0.0190, update cosine=-0.0008.
- public, beta=0.4: final-accuracy winner=forward; inverse−forward accuracy=-0.0299, AUC=-0.4053, clip fraction=+0.3072, mean clip factor=-0.2692, update relative error=+0.1060, update cosine=-0.0038.
- pink, beta=0.4: final-accuracy winner=forward; inverse−forward accuracy=-0.0127, AUC=-0.1481, clip fraction=+0.1016, mean clip factor=-0.0874, update relative error=+0.0309, update cosine=-0.0011.
