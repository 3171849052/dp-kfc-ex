# Exp20



SMOKE ONLY: one zero-noise batch. These observations do not answer formal utility questions.



## 1. Best final accuracy

p=[0.25]; accuracy=0.214844 (all ties listed).

## 2. Best accuracy AUC

p=[0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 1.0]; AUC=0.000000. AUC integrates observed epochs; smoke AUC is zero.

## 3. Clipping and norm tails

Clipping fraction is constant; p99 norm is nondecreasing as p increases. Ordered p / mean epoch clip fraction / mean epoch p99 norm:

- 0: 1.000000 / 13.634374

- 0.125: 1.000000 / 20.732204

- 0.25: 1.000000 / 34.245144

- 0.375: 1.000000 / 51.235153

- 0.5: 1.000000 / 72.315376

- 0.625: 1.000000 / 98.869400

- 0.75: 1.000000 / 132.111694

- 1: 1.000000 / 223.523972

## 4. Is anisotropic A geometry useful?

Scale-matched identity accuracy=0.207031. Positive-p accuracy minus identity: 0.125: +0.003906, 0.25: +0.007812, 0.375: -0.011719, 0.5: -0.019531, 0.625: -0.023438, 0.75: -0.019531, 1: -0.019531.

The best anisotropic operator improves on scale-matched identity in this trajectory, supporting geometry utility for this seed. Scale matching controls the synthetic global RMS, not every private norm or optimization effect.

## 5. Above p=0.5: over-preconditioning?

Differences versus p=0.5 (accuracy / p99 / clip fraction):

- 0.625: -0.003906 / +26.554024 / +0.000000; higher tail and lower accuracy are consistent with over-preconditioning.

- 0.75: +0.000000 / +59.796318 / +0.000000; no joint higher-tail/lower-accuracy pattern.

- 1: +0.000000 / +151.208595 / +0.000000; no joint higher-tail/lower-accuracy pattern.

## 6. Interpreting the Exp19 p=0.25 / p=0.5 proximity

Here p=0.25 minus p=0.5: final accuracy +0.027344; AUC +0.000000.

Dense neighboring results (p / final accuracy / AUC): 0.125 / 0.210938 / 0.000000, 0.25 / 0.214844 / 0.000000, 0.375 / 0.195312 / 0.000000, 0.5 / 0.187500 / 0.000000, 0.625 / 0.183594 / 0.000000.

Within p=0.25..0.5, accuracy is nonincreasing, with range 0.027344; the intermediate p=0.375 differs from p=0.5 by +0.007812. Endpoint proximity alone therefore does not establish a flat optimum. This single seed cannot establish that the Exp19 observation generalizes.



Smoke cannot establish any of these geometry/utility interpretations.



Scale uses each run’s current synthetic A every epoch. p=0 is scalar identity, not ordinary DP-SGD. Biases use augmented activations. Spectral quantiles weight each activation eigenvalue once; RMS moments additionally weight layer output dimension.

Timing: algorithm = build + private training; CPU norm diagnostics and evaluation are separate. CUDA memory peaks cover build/train. Layer diagnostics describe fractions of total transformed squared norm.

RDP accounting follows Exp19’s shuffled fixed-batch convention. Private clipping diagnostics are unnoised research measurements.
