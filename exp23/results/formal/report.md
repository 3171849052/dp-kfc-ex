# Exp23 descriptive results

No strong statistical significance claims; sample std uses ddof=1. Accuracy is a fraction.

research-only: unnoised oracle/clipping/norm diagnostics; not a DP release

AUC: trapezoid over observed epochs 1..5; one-epoch smoke AUC is zero.

Timing includes cold-start work; oracle time is separate. DP-SGD has no calibration.


Final accuracy paired comparisons:


        metric              comparison      mean  sample_std  n
final_accuracy               Penalty_A -0.001767    0.000115  3
final_accuracy            Penalty_Full  0.000667    0.002150  3
final_accuracy             Interaction  0.002433    0.002237  3
final_accuracy    Fashion_A_minus_Full  0.011867    0.001222  3
final_accuracy      CIFAR_A_minus_Full  0.014300    0.003045  3
final_accuracy    a_fashion_minus_pink  0.000133    0.001069  3
final_accuracy    a_cifar10_minus_pink  0.001900    0.001136  3
final_accuracy full_fashion_minus_pink  0.000400    0.002689  3
final_accuracy full_cifar10_minus_pink -0.000267    0.002957  3


Only three paired seeds: descriptive statistics.
