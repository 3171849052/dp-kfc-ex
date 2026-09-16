# Exp23 SMOKE — pipeline verification only

No strong statistical significance claims; sample std uses ddof=1. Accuracy is a fraction.

research-only: unnoised oracle/clipping/norm diagnostics; not a DP release

AUC: trapezoid over observed epochs 1..5; one-epoch smoke AUC is zero.

Timing includes cold-start work; oracle time is separate. DP-SGD has no calibration.


Final accuracy paired comparisons:


        metric              comparison      mean  sample_std  n
final_accuracy               Penalty_A -0.007812         NaN  1
final_accuracy            Penalty_Full -0.013672         NaN  1
final_accuracy             Interaction -0.005859         NaN  1
final_accuracy    Fashion_A_minus_Full  0.039062         NaN  1
final_accuracy      CIFAR_A_minus_Full  0.033203         NaN  1
final_accuracy    a_fashion_minus_pink -0.003906         NaN  1
final_accuracy    a_cifar10_minus_pink  0.003906         NaN  1
final_accuracy full_fashion_minus_pink -0.021484         NaN  1
final_accuracy full_cifar10_minus_pink -0.007812         NaN  1


Smoke results must not be used for formal conclusions.
