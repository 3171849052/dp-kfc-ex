# Exp14c：KFLR / KFRA-block scale-matched power ablation

研究目标：控制 synthetic curvature predicted RMS scale 后，判断 exponent dependence
是否跨 curvature estimator 保持，是否呈现与此前 KFAC-U 实验类似的依赖。
Exp14c 只比较 KFLR 和 KFRA-block，不包含 KFAC-U / DP-KFC，也不使用 full KFRA。
Smoke 仅检查 correctness，不能据此回答研究问题。

## 算子与复用

`builders.py` 直接调用 `exp12.curvature.estimate`，只允许 KFLR、KFRA-block 和
SimpleCNN。KFLR 保持 Exp13 exact-output GGN/Fisher synthetic estimator（10 类输出的
9 个方向）；KFRA-block 保持 SimpleCNN recursive block approximation。
`operator.py` 直接复用 Exp14b power + global RMS matching：

```
beta ∈ {0, 0.25, 0.5, 0.75, 1.0}, π = 1e-3
P_beta(G) = (C + πI)^(-beta) G (A + πI)^(-beta)
m_beta = Σ_l Tr[C_l (C_l + πI)^(-2beta)] Tr[A_l (A_l + πI)^(-2beta)]
m_ref = m_0.5
s_beta = sqrt(m_ref / m_beta)
L = sqrt(s_beta) (C + πI)^(-beta)
R = sqrt(s_beta) (A + πI)^(-beta)
P_hat_beta(G) = L G R = s_beta P_beta(G)
```

每个 estimator、当前 run/epoch 在自身同一组 factors 上计算 beta=0.5 reference，
不是使用另一个 reference run 或另一个 estimator 的 factors。所有层求和后计算一个
全模型 scale。只用 synthetic factors，不用 private data、private gradients 或 private
norms 估计 scale。global scale matching 不保证真实 private transformed norms 相同；
模型轨迹分离后，不同 run 的 m_ref 也可能不同。

beta=0.5 时 scale_match 精确为 1，分别回归 Exp13 DP-KFLR / DP-KFRA-block operator。
beta=0 是 **scale-matched identity：s_0 × I**。factor power 使用 float64 对称
特征分解，最后转 float32，再对称乘入 sqrt(s_beta)。moment 使用 float64 谱，
负舍入特征值截零，不使用 geometry spectral floor。满足 s_beta² m_beta = m_ref。

`run_exp14c.py` 沿用 Exp14b 的训练循环，复用 Exp13 initialize/evaluate、Structured
Ghost Clipping（GhostNorm、ghost_aggregate、noise_and_step），以及 Exp12 runtime 和
synthetic_state_metrics。训练不生成 per-example parameter gradients；显式逐样本梯度
仅出现在测试。所有新增代码、测试、文档及结果位于 exp14c，不修改已有实验。

## 固定设置与随机数

MNIST / `dp_kfac.models.SimpleCNN`；seeds=42,7；epochs=5；batch size=256；
SGD lr=0.5、momentum=0、weight_decay=0；epsilon=1、delta=1e-5；
max_grad_norm=1；damping=1e-3。每 epoch rebuild，10×256 synthetic pink-noise batches。
`config.py` 复用 Exp14b / Exp13 设置。

所有 estimator × beta 使用同 seed 初始化、独立 private shuffle generator（seed）、
相同 synthetic cache schedule（seed+10000+epoch）以及 Gaussian noise generator
（seed+40000，同参数次序/形状）。builder 使用 seed+20000+epoch 并保存恢复外部
CPU/所有 CUDA RNG；这两个 estimator 不采样 pseudo-labels。
统一计算一次 sigma，沿用 Exp13 RDP shuffled fixed-batch accounting convention，
shuffle=True、drop_last=True；正式每 run 共1170 accountant steps。
CUDA deterministic 设置和 disposable warmup 沿用 Exp14b。
研究日志不属于 DP 输出保证。

## Geometry、指标与输出

`geometry.py` 复用 Exp14b / Exp14 floored geometry，按 estimator 标记：
`lambda_tilde = lambda * (lambda + damping)^(-2beta)`，spectral floor 为每 factor
最大值的 1e-7。报告 floored condition number、log eigenvalue spread（自然对数总体
标准差）、effective rank（高于 floor 的数量）、zero eigenvalue count、dimension 和
spectral floor ratio；全零谱 condition/spread 为 NaN、rank=0。不 materialize Kronecker
matrix。全模型 scalar 不改变这些相对谱指标；floored condition 不代表真实谱非奇异。

- `metrics.csv`：每 estimator/beta/seed/epoch 的 test accuracy/loss、train loss、
  clip_fraction、mean_clip_factor、transformed_norm_p50/p90/p99、epsilon/accountant、
  timing、CUDA memory、synthetic state metrics、builder budget，以及
  predicted_second_moment_raw/ref、scale_match、predicted_rms_raw/matched。
- `geometry.csv`：每 estimator/beta/seed/epoch/layer 的 A/C 谱指标和 floored_kappa_block。
- `summary.csv`：各 estimator/beta/seed 最后 epoch，保留 diagnostics、累计时间和峰值。
- `estimator_beta_summary.csv`：按 estimator/beta 跨 seed 汇总均值与 sample std。
- `exponent_effects.csv`：同 estimator、同 seed，非 reference beta 减 beta=0.5 的最终
  accuracy_difference、clip_fraction_difference、transformed_norm_p90/p99_difference，
  以及层等权平均 floored block condition 差值；明确写入 reference_beta。
- `config.json`：实际运行配置、统一 sigma/accounting、RNG schedule、scale 定义。

build 时间包括 cache、curvature、factor power、scale matching；geometry、synthetic
state metrics、evaluation 独立计时。algorithm=build+private train；wall 为五阶段之和，
不含 CSV 和阶段间管理。memory 为 build/train CUDA allocated peaks 的最大值，不含
独立 geometry/evaluation 峰值。accuracy 范围 [0,1]；单 seed sample std 留空。

## 运行

需要 conda `curve`、CUDA 和已有 `exp1/data` MNIST。

```bash
conda run -n curve pytest -q exp14c/test_exp14c.py
conda run -n curve python exp14c/run_exp14c.py --smoke
```

Smoke：seed42、两个 estimator × 全部5个 beta、1 epoch、1 private batch=256、
1 synthetic batch=256、sigma=0；test 为前256张，输出 `exp14c/results/smoke/`。
Smoke 只用于 correctness，不提供 DP 保证；epsilon 留空，accountant_steps=0。
测试覆盖两 estimator 的半次幂回归、所有 beta 的 scale 等式和 Ghost aggregate、
小矩阵显式 transform、builder CPU/CUDA RNG 隔离、两个 seed 的相同初始化。

重新分析：`conda run -n curve python exp14c/analyze.py exp14c/results/smoke`。
完整实验仅手动启动，输出 `exp14c/results/`：

```bash
conda run -n curve python exp14c/run_exp14c.py
```
