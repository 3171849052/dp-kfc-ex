# Exp20 paired power sweep

SMOKE ONLY: no formal utility conclusions.

## Mean performance and clipping geometry

Best final accuracy: [0.25]. Best accuracy AUC: [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 1.0].

Lowest clip fraction: [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 1.0]; lowest norm p99: [0.0]. These are separate geometry criteria, not a single utility optimum.

p,final_accuracy_mean,final_accuracy_sample_std,best_accuracy_mean,best_accuracy_sample_std,accuracy_auc_mean,accuracy_auc_sample_std,clip_fraction_mean,clip_fraction_sample_std,norm_p99_mean,norm_p99_sample_std,scale_match_mean,scale_match_sample_std,transformed_condition_proxy_mean,transformed_condition_proxy_sample_std
0.0,0.20703125,,0.20703125,,0.0,,1.0,,13.634373664855955,,2.253616802384591,,inf,
0.125,0.2109375,,0.2109375,,0.0,,1.0,,20.73220443725586,,2.6915774471255904,,inf,
0.25,0.21484375,,0.21484375,,0.0,,1.0,,34.24514389038086,,2.3486504361541645,,inf,
0.375,0.1953125,,0.1953125,,0.0,,1.0,,51.23515319824219,,1.6164862477852333,,inf,
0.5,0.1875,,0.1875,,0.0,,1.0,,72.31537628173828,,1.0,,inf,
0.625,0.18359375,,0.18359375,,0.0,,1.0,,98.86940002441406,,0.5880331211876101,,inf,
0.75,0.1875,,0.1875,,0.0,,1.0,,132.1116943359375,,0.3350908607474391,,inf,
1.0,0.1875,,0.1875,,0.0,,1.0,,223.5239715576172,,0.1019879142089508,,inf,


## p=.375 − p=.5

Individual seed deltas, paired means, sample standard deviations and bootstrap 95% intervals:

p,reference_p,metric,paired_mean_delta,sample_std,ci95_low,ci95_high,delta_seed_42
0.375,0.5,final_accuracy,0.0078125,,0.0078125,0.0078125,0.0078125
0.375,0.5,accuracy_auc,0.0,,0.0,0.0,0.0
0.375,0.5,clip_fraction,0.0,,0.0,0.0,0.0
0.375,0.5,norm_p99,-21.080223083496094,,-21.080223083496094,-21.080223083496094,-21.080223083496094


Raw paired results:

p,seed,final_accuracy,best_accuracy,accuracy_auc,clip_fraction,norm_p99,scale_match,transformed_condition_proxy
0.375,42,0.1953125,0.1953125,0.0,1.0,51.23515319824219,1.6164862477852333,inf
0.5,42,0.1875,0.1875,0.0,1.0,72.31537628173828,1.0,inf


final_accuracy stability (smaller sample std): p=0.375: nan, p=0.5: nan

p=.375 improves final_accuracy in 1/1 seeds; mean delta +0.0078125, CI [+0.0078125, +0.0078125]. Positive in every observed seed.

accuracy_auc stability (smaller sample std): p=0.375: nan, p=0.5: nan

p=.375 improves accuracy_auc in 0/1 seeds; mean delta +0, CI [+0, +0]. Not consistently positive across observed seeds.

## Activation anisotropic geometry: p>0 versus p=0

Scale matching controls synthetic global RMS, not all private norms or optimization effects. p=0 is scalar identity, not ordinary DP-SGD.

p=0.125: final accuracy delta mean +0.00390625, CI [+0.00390625, +0.00390625]; positive in 1/1 seeds. Consistent observed support for geometry utility.

p=0.25: final accuracy delta mean +0.0078125, CI [+0.0078125, +0.0078125]; positive in 1/1 seeds. Consistent observed support for geometry utility.

p=0.375: final accuracy delta mean -0.0117188, CI [-0.0117188, -0.0117188]; positive in 0/1 seeds. No uniformly positive geometry benefit across seeds.

p=0.5: final accuracy delta mean -0.0195312, CI [-0.0195312, -0.0195312]; positive in 0/1 seeds. No uniformly positive geometry benefit across seeds.

p=0.625: final accuracy delta mean -0.0234375, CI [-0.0234375, -0.0234375]; positive in 0/1 seeds. No uniformly positive geometry benefit across seeds.

p=0.75: final accuracy delta mean -0.0195312, CI [-0.0195312, -0.0195312]; positive in 0/1 seeds. No uniformly positive geometry benefit across seeds.

p=1.0: final accuracy delta mean -0.0195312, CI [-0.0195312, -0.0195312]; positive in 0/1 seeds. No uniformly positive geometry benefit across seeds.

## Above p=.5: norm tail ↑, clipping ↑, accuracy ↓

p=0.625: joint pattern in 0/1 seeds; mean deltas {'final_accuracy': -0.00390625, 'accuracy_auc': 0.0, 'clip_fraction': 0.0, 'norm_p99': 26.55402374267578}. Pattern is not consistent across all seeds. This association does not establish causation.

p=0.75: joint pattern in 0/1 seeds; mean deltas {'final_accuracy': 0.0, 'accuracy_auc': 0.0, 'clip_fraction': 0.0, 'norm_p99': 59.79631805419922}. Pattern is not consistent across all seeds. This association does not establish causation.

p=1.0: joint pattern in 0/1 seeds; mean deltas {'final_accuracy': 0.0, 'accuracy_auc': 0.0, 'clip_fraction': 0.0, 'norm_p99': 151.2085952758789}. Pattern is not consistent across all seeds. This association does not establish causation.

## Moderate optimum and under/over-conditioning

Both mean-utility maxima lie in p≈.375–.5: False. The two mean-utility maxima do not jointly support that optimum region.

Using mean-final-accuracy winner p=0.25 as the candidate sweet spot: improvement over p=0 in 1/1 seeds; higher tail/clipping and lower accuracy at p=.625,.75,1 in [0, 0, 0] seeds respectively. Consistent full pattern across observed seeds: False. This selection is descriptive and made after observing the sweep.

The p=0 comparisons and above-.5 joint counts quantify the proposed under-conditioning → sweet spot → over-preconditioning pattern; mixed seed signs limit its stability. Mean maxima alone do not establish a universal optimum.

Timing: algorithm = builder + private training; diagnostic CPU transfers and evaluation excluded; CUDA internal breakdown uses deferred Events. Runtime is secondary; small single-machine fluctuations do not establish power-specific speedups.

Biases use augmented activations. Spectral quantiles weight each eigenvalue once; RMS moments additionally weight layer output dimension. RDP uses the unchanged shuffled fixed-batch convention; clipping diagnostics are unnoised research measurements.
