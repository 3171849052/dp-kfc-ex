# Pareto summary

SMOKE ONLY: no scientific conclusions.

Historical references (read only; never inserted into training metrics):
- full: 0.9570, seed=42, beta=0.25, `exp14b/results/summary.csv`.
- identity: 0.9221, seed=42, beta=0.0, `exp14/results/summary.csv`.

Identity uses Exp14 beta=0: Exp14b beta=0 is globally rescaled and is not identity.
Utility retention = (accuracy − identity) / (full − identity).
Pareto flags compare newly trained methods. Historical runtime has separate-run hardware/timing caveats.
Historical operator bytes are not measured in those CSVs; baseline horizontal lines compare utility only.

     method  test_accuracy  utility_retention  operator_state_bytes  total_algorithm_seconds  memory_frontier  runtime_frontier
       diag            0.5         -12.094556                  8164                 0.515325             True             False
     a_only            0.5         -12.094556               9998116                 0.237868            False             False
     c_only            0.5         -12.094556                 71064                 0.020446            False              True
fullA_diagC            0.5         -12.094556               9998860                 0.143108            False             False
diagA_fullC            0.5         -12.094556                 78476                 0.020821            False             False
      rank4            0.5         -12.094556                 32824                 0.032371            False             False
      rank8            0.5         -12.094556                 65576                 0.024334            False             False
     rank16            0.5         -12.094556                130384                 0.022856            False             False
   refresh2            0.5         -12.094556              10069172                 0.151540            False             False
     frozen            0.5         -12.094556              10069172                 0.131664            False             False

