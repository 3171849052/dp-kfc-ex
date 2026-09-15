# Pareto summary

Single seed=42; differences are descriptive, without uncertainty estimates.

Historical references (read only; never inserted into training metrics):
- full: 0.9570, seed=42, beta=0.25, `exp14b/results/summary.csv`.
- identity: 0.9221, seed=42, beta=0.0, `exp14/results/summary.csv`.

Identity uses Exp14 beta=0: Exp14b beta=0 is globally rescaled and is not identity.
Utility retention = (accuracy − identity) / (full − identity).
Pareto flags compare newly trained methods. Historical runtime has separate-run hardware/timing caveats.
Historical operator bytes are not measured in those CSVs; baseline horizontal lines compare utility only.

     method  test_accuracy  utility_retention  operator_state_bytes  total_algorithm_seconds  memory_frontier  runtime_frontier
       diag         0.9322           0.289398                  8164                51.518302             True             False
     a_only         0.9588           1.051576               9998116                51.925706             True              True
     c_only         0.9221           0.000000                 71064                51.256185            False              True
fullA_diagC         0.9583           1.037249               9998860                51.934849            False             False
diagA_fullC         0.9325           0.297994                 78476                51.497906            False             False
      rank4         0.9513           0.836676                 32824                55.468753             True             False
      rank8         0.9492           0.776504                 65576                55.353399            False             False
     rank16         0.9535           0.899713                130384                52.887163             True             False
   refresh2         0.9537           0.905444              10069172                51.611979            False              True
     frozen         0.9425           0.584527              10069172                51.323522            False              True

Diagonal retains 28.9% of the historical gain.
A-only − C-only accuracy: +0.0367; fullA/diagC − diagA/fullC: +0.0258.
Rank 4→8 gain: -0.0021; 8→16 gain: +0.0043. A smaller latter gain suggests saturation; one seed cannot establish it.
refresh2 − historical full: -0.0033; retention 90.5%.
frozen − historical full: -0.0145; retention 58.5%.
