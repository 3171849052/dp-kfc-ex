# Exp20



Single seed=42 descriptive comparison. No bootstrap confidence intervals or significance claims.



## 1. Best final accuracy

p=[0.375]; accuracy=0.960700 (all ties listed).

## 2. Best accuracy AUC

p=[0.5]; AUC=3.786300. AUC integrates observed epochs; smoke AUC is zero.

## 3. Clipping and norm tails

Clipping fraction is nonmonotonic; p99 norm is nondecreasing as p increases. Ordered p / mean epoch clip fraction / mean epoch p99 norm:

- 0: 0.315508 / 309.050729

- 0.125: 0.278943 / 474.797620

- 0.25: 0.266573 / 618.445947

- 0.375: 0.270870 / 685.258740

- 0.5: 0.283183 / 729.189557

- 0.625: 0.300244 / 783.906006

- 0.75: 0.318974 / 832.734631

- 1: 0.356564 / 900.640442

## 4. Is anisotropic A geometry useful?

Scale-matched identity accuracy=0.919500. Positive-p accuracy minus identity: 0.125: +0.026000, 0.25: +0.039200, 0.375: +0.041200, 0.5: +0.040400, 0.625: +0.039800, 0.75: +0.038600, 1: +0.034200.

The best anisotropic operator improves on scale-matched identity in this trajectory, supporting geometry utility for this seed. Scale matching controls the synthetic global RMS, not every private norm or optimization effect.

## 5. Above p=0.5: over-preconditioning?

Differences versus p=0.5 (accuracy / p99 / clip fraction):

- 0.625: -0.000600 / +54.716449 / +0.017061; higher tail and lower accuracy are consistent with over-preconditioning.

- 0.75: -0.001800 / +103.545074 / +0.035791; higher tail and lower accuracy are consistent with over-preconditioning.

- 1: -0.006200 / +171.450885 / +0.073381; higher tail and lower accuracy are consistent with over-preconditioning.

## 6. Interpreting the Exp19 p=0.25 / p=0.5 proximity

Here p=0.25 minus p=0.5: final accuracy -0.001200; AUC -0.016100.

Dense neighboring results (p / final accuracy / AUC): 0.125 / 0.945500 / 3.728000, 0.25 / 0.958700 / 3.770200, 0.375 / 0.960700 / 3.782250, 0.5 / 0.959900 / 3.786300, 0.625 / 0.959300 / 3.782450.

Within p=0.25..0.5, accuracy is nonmonotonic, with range 0.002000; the intermediate p=0.375 differs from p=0.5 by +0.000800. Endpoint proximity alone therefore does not establish a flat optimum. This single seed cannot establish that the Exp19 observation generalizes.



Interpretations are restricted to this single paired trajectory.



Scale uses each run’s current synthetic A every epoch. p=0 is scalar identity, not ordinary DP-SGD. Biases use augmented activations. Spectral quantiles weight each activation eigenvalue once; RMS moments additionally weight layer output dimension.

Timing: algorithm = build + private training; CPU norm diagnostics and evaluation are separate. CUDA memory peaks cover build/train. Layer diagnostics describe fractions of total transformed squared norm.

RDP accounting follows Exp19’s shuffled fixed-batch convention. Private clipping diagnostics are unnoised research measurements.
