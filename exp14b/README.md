# Exp14b: synthetic-KFAC RMS scale matched power sweep

Exp14 是 raw power sweep：改变 β 同时改变 anisotropy 和整体梯度 scale。
Exp14b 是 synthetic-KFAC predicted RMS scale matched sweep，用于区分 overall
scale effect 与 exponent-dependent anisotropy。

## 唯一变动：全模型 scale matching

仍用 β ∈ {0, 0.25, 0.5, 0.75, 1}，π=1e-3：

```
P_beta(G) = (C + πI)^(-β) G (A + πI)^(-β)
m_beta = Σ_l Tr[C_l (C_l + πI)^(-2β)] Tr[A_l (A_l + πI)^(-2β)]
m_ref = m_0.5
s_beta = sqrt(m_ref / m_beta)
L = sqrt(s_beta) (C + πI)^(-β)
R = sqrt(s_beta) (A + πI)^(-β)
P_hat_beta(G) = L G R = s_beta P_beta(G)
```

每个当前 run/epoch 在自己的同一组 synthetic KFAC factors 上计算 raw 和 reference
second moment；m_ref 不取自另一个 β=0.5 run。先对所有层求和，再计算一个全模型
scale，不逐层匹配。不使用 private data、private gradients 或 private norms 调 scale。
这只控制 synthetic-KFAC predicted global second-moment scale，不强制 private
transformed norms 相同，也不保证后续各 run 的 m_ref 相同，因为模型会不同。

β=0.5 是 reference，scale=1，算子数值回归 Exp14 β=0.5。
β=0 的 raw power 是 identity，matched 算子是 s_0 × identity。
β=1 的 raw power 是 factorwise KFAC inverse，不是严格 `(F+πI)^-1`。
scale 用 float64 对称 eigvalsh 的 trace sums 计算；PSD eigenvalues 的负舍入误差
截为零，不使用 geometry 的 spectral floor 计算 moment。factor power 直接复用
Exp14 的 float64 eigendecomposition / float32 输出，再将 sqrt(scale) 乘入两侧。

## 复用与固定设置

`operator.py` 继承 Exp14 Operator，仅增加 scale；`builders.py` 沿用 Exp14 KFAC-U
builder，直接调用 Exp12 estimate，复用 Exp13 synthetic_cache。
`run_exp14b.py` 沿用 Exp14 训练循环、RNG/accounting、计时、memory 和 smoke，
直接调用 Exp13 initialize/evaluate、GhostNorm、ghost_aggregate、noise_and_step，
以及 Exp12 runtime/synthetic_state_metrics。训练不生成 per-example parameter
梯度；逐样本显式梯度仅用于 correctness test。
`geometry.py`、`analyze.py` 直接复用 Exp14 geometry 和 save 分析逻辑。
所有新增内容及输出在 exp14b，已有实验不修改。

MNIST / dp_kfac.models.SimpleCNN，seeds=42,7，epochs=5，batch=256，shuffle、
drop_last=True；SGD lr=0.5，momentum=weight_decay=0；epsilon=1，delta=1e-5，
max_grad_norm=1。每 epoch 重建一次，10×256 synthetic pink-noise，固定 KFAC-U k=1。
各 β 相同 seed 初始化；shuffle 独立 generator seed；synthetic cache 使用
seed+10000+epoch；uniform label generator 使用 seed+20000+epoch；Gaussian
noise generator 使用 seed+40000，同参数次序/形状/RNG schedule。builder 保存恢复
外部 CPU/所有 CUDA RNG。sigma 统一计算一次，正式共1170 accountant steps，沿用
Exp13 RDP shuffled fixed-batch convention（不是 Poisson sampling）。未加噪的研究
日志不属于 DP 输出保证。CUDA 和 disposable warmup 设置同 Exp14。

## Geometry 与结果

报告 exponent geometry：`lambda_tilde = lambda * (lambda + π)^(-2β)`。
每 factor 使用 `1e-7 * max(lambda_tilde)` spectral floor，记录 floored condition、
log-eigenvalue spread（自然对数总体标准差）、effective rank（高于 floor 的数量）、
dimension、zero eigenvalue count、spectral floor ratio。全零谱 condition/spread
为 NaN，rank=0。真实谱可能奇异；floored condition 不意味着真实 condition 有限。
全模型 scalar s_beta 不改变 condition、log spread 或相对 floor 下的 rank，
因此直接复用 Exp14 exponent geometry；不 materialize Kronecker matrix。

- `metrics.csv`：每 beta/seed/epoch 所有 Exp14 训练指标，以及
  predicted_second_moment_raw、predicted_second_moment_ref、scale_match、
  predicted_rms_raw、predicted_rms_matched。后者应等于 sqrt(m_ref)。
- `geometry.csv`：每层 A/C floored geometry、floored_kappa_block。
- `summary.csv`：各 beta/seed 最后 epoch，保留 scale diagnostics、累计时间和峰值；
  transformed_block_floored_condition_number 为各层 block condition 的等权均值。
- `paired_summary.csv`：同 seed 各 β 减 β=0.5 的最后 epoch accuracy、clip fraction、
  transformed norm p90/p99、floored block condition 差值。
- `beta_summary.csv`：上述指标按 β 跨 seed 均值及 sample std。
- `config.json`：实际设置、统一 sigma/accounting、RNG schedule 和 scale 定义。

build 时间包含 cache、curvature、power、scale matching；geometry、synthetic state
metrics、evaluation 分开计时。algorithm=build+private train，wall 为五阶段之和，
不含 CSV/阶段间管理。memory 沿用 build/train CUDA allocated peaks，total_peak 为
两阶段最大值，不含独立 geometry/evaluation 峰值。Accuracy 为 [0,1]；两 seeds
仅作描述统计，单 seed 的 sample std 留空。

## 执行

需要 conda curve、CUDA、已有 exp1/data MNIST：

```bash
conda run -n curve pytest -q exp14b/test_exp14b.py
conda run -n curve python exp14b/run_exp14b.py --smoke
```

唯一 smoke：seed42、全部五个 β、1 epoch、1 private batch=256、1 synthetic
batch=256、sigma=0，test 为前256张，写入 `exp14b/results/smoke/`。
Smoke 不提供 DP 保证，epsilon 留空、accountant_steps=0；也不是 hypothesis 的证据。
测试仅覆盖 β=0.5 回归、scale 等式、小矩阵显式 transform、所有 β 的 Ghost
aggregate、builder RNG 隔离与相同初始化。

重新分析：`conda run -n curve python exp14b/analyze.py exp14b/results/smoke`。
完整实验只手动启动，输出 `exp14b/results/`：

```bash
conda run -n curve python exp14b/run_exp14b.py
```
