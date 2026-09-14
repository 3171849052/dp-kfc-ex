# Exp14: DP-KFC preconditioner power ablation

唯一实验变量是 β ∈ {0, 0.25, 0.5, 0.75, 1}：
`P_beta(G) = (C + πI)^(-β) G (A + πI)^(-β)`，π=1e-3。
β=0 是 identity control；β=0.5 直接复用 Exp13 inverse_sqrt 数值路径；
β=1 是 factorwise KFAC inverse，不是严格的 `(F+πI)^-1`。
其他 power 同样在 float64 中对 symmetric damped factor 做 eigendecomposition，
转回 float32。不额外归一化或改变 clipping bound。

## 实现与固定设置

`operator.py` 继承 Exp13 Operator 的所有 transform 接口，仅替换 factor power。
`builders.py` 复用 Exp13 synthetic_cache 和 Exp12 estimate 的 KFAC-U（k=1）路径，
保持 bias augmentation、conv unfolding、A/C averaging，不引入其他 estimator。
`run_exp14.py` 沿用 Exp13 训练循环，直接复用 initialize、evaluate、timestamp，
Exp12 runtime / synthetic_state_metrics，以及 Exp13 的 GhostNorm、ghost_aggregate、
noise_and_step。训练没有 per-example parameter gradients；显式逐样本参考仅在测试中。
所有新增文件和输出都位于 exp14，原实验不变。

MNIST / dp_kfac.models.SimpleCNN；seeds=42,7；5 epochs；batch=256，shuffle、
drop_last=True；SGD lr=0.5，momentum=weight_decay=0；epsilon=1，delta=1e-5；
max_grad_norm=1。每 epoch 重建一次，10×256 synthetic pink-noise batches。
相同 seed 的各 β 使用完全相同初始化、独立同 seed shuffle generator；
同 seed+10000+epoch 重建相同 synthetic cache，uniform labels 使用独立
seed+20000+epoch generator；builder 保存/恢复 CPU 和所有 CUDA RNG。
Gaussian noise generator 固定 seed+40000，参数次序、形状和调用次数相同。
模型随 β 演化不同，因此之后的 curvature factors 可以不同。
统一 sigma 只计算一次，accounting 沿用 Exp13 的 RDP shuffled fixed-batch
convention（不是 Poisson sampling），正式共 1170 steps。
未加噪 train loss / clipping 研究日志不属于 DP 输出保证。

## Geometry 与输出

每 epoch 在更新前同一 factors 上，float64 eigvalsh，计算
`lambda_tilde = lambda * (lambda + π)^(-2β)`。
只处理 A、C，不构造 Kronecker matrix。估计 PSD factor 的负舍入 eigenvalues
截为零。True transformed spectrum 仍可能是 singular（奇异）的。为了跨 β 做有限、
稳定的描述性比较，使用 `floor = 1e-7 * transformed.max()`，并对
`floored = transformed.clamp_min(floor)` 计算 max/min condition number 和
自然对数的总体标准差（log-eigenvalue spread）。A/C 分别记录 spectral_floor_ratio。
`floored_kappa_block = A_floored_condition_number * C_floored_condition_number`。
同时分别保留 dimension、zero_eigenvalue_count 和
`effective_rank = (transformed > floor).sum()`：零 eigenvalue 数保留奇异信息，
effective rank 描述该阈值下的数值秩，不等同于精确代数秩。
Floored condition number 只是 diagnostic，不代表奇异情况下真实 condition number 有限。
若整个 transformed spectrum 为零，condition、spread、floor ratio 记 NaN，rank=0；
汇总保留该未定义值，不提供替代数值。

- `metrics.csv`：每 beta/seed/epoch 的 Exp13 指标，另含 beta、geometry_seconds。
- `geometry.csv`：每 beta/seed/epoch/layer 的 A/C floored condition、log spread、rank、dimension、zero count、floor ratio 和 floored block condition。
- `summary.csv`：每 beta/seed 最后 epoch，附累计时间和峰值 memory，
  transformed_block_floored_condition_number 是该 epoch 各层 floored_kappa_block 的等权算术均值。
- `paired_summary.csv`：同 seed 最后 epoch 各 β 减 β=0.5 的 accuracy、clip fraction、
  norm p90/p99、floored block condition 差值。
- `beta_summary.csv`：最后 epoch 的上述五个量，按 β 汇总跨 seed 均值与 sample std。
- `config.json`：固定设置、实际统一 sigma/accounting、RNG schedule 和计时定义。

Accuracy 为 [0,1]。保留 per-layer geometry 便于解读，层均值只是描述统计；
两 seeds 不做显著性检验。build 计时包含 cache、curvature、power；
geometry 与 synthetic state metrics、evaluation 分开同步计时；
algorithm=build+private train，wall 为上述五阶段之和，不含 CSV 和阶段间管理开销。
沿用 Exp13 build/train CUDA allocated peaks，total_peak 为这两个阶段最大值，
不表示 geometry/evaluation 全阶段峰值。正式训练前使用独立 disposable warmup，
不推进训练、noise generator 或 accountant。

## 执行

需要 conda curve、CUDA、已有 exp1/data MNIST。按顺序：

```bash
conda run -n curve pytest -q exp14/test_exp14.py
conda run -n curve python exp14/run_exp14.py --smoke
```

唯一 smoke：seed=42，全部五个 β，1 epoch，1 private batch=256，
1 synthetic batch=256，sigma=0；test 使用前256张。
输出到 `exp14/results/smoke/`。这是 correctness smoke，不提供 DP 保证，
也不是 hypothesis 正向 signal 的证据。epsilon 留空、accountant_steps=0。
测试仅覆盖 operator regression、identity、small-matrix reference、
全部 β 的 Ghost clipped aggregate、builder RNG 隔离和相同初始化。

重新分析：`conda run -n curve python exp14/analyze.py exp14/results/smoke`。
完整实验手动启动，输出到 `exp14/results/`：

```bash
conda run -n curve python exp14/run_exp14.py
```
