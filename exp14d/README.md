# Exp14d: exponent and empirical whitening

Hypothesis: **“DP utility may peak at intermediate empirical whitening rather than maximal whitening.”**

本实验研究 preconditioning exponent p（代码沿用 `beta`）与经验白化、clipping、accuracy 的关系。
主区间为 beta={0, .125, .25, .375, .5}；.75、1 是 over-whitening controls。
beta=.5 是理论 near-full-whitening reference；由于 damping、factor approximation 和有限 probe，不能保证 W=1。
beta>.5 不能简单解释为“更多 whitening”。

## 固定协议和复用

正式 sweep：7 beta × seeds {42, 7} = 14 runs。MNIST + SimpleCNN，5 epochs；
epsilon=1、delta=1e-5、max_grad_norm=1；SGD lr=.5、momentum=0、weight_decay=0；
train/eval batch_size=256；shuffle=True、drop_last=True，每 run 234×5=1170 private steps。
每 epoch 使用 10×256 pink-noise、uniform random labels 构造 synthetic KFAC-U，damping=1e-3。
全局 synthetic RMS scale matching 的 reference beta=.5。
复用 exp14b 的固定 batch shuffled RDP accounting，同一次 sigma calibration 用于全部 runs。

`run_exp14d.py` 直接调用 `exp14b.run_exp14b.run`。只在调用期间使用 scoped
`patch.object` 替换其 builder 引用为 observer；observer 调用原
`exp14b.builders.build_from_cache`，再测量 probe，返回原 operator、原 factors 和补充的 stats。
退出作用域恢复引用；不修改已有源码。训练循环、synthetic KFAC、RMS matching、
GhostNorm、clipping、DP noise/step、evaluation 和 accounting 均复用原实现。
这不是并发运行接口；当前 runner 顺序运行各 seed/beta。
原 builder 的计时会包含 probe diagnostic，不能将其直接视为纯 preconditioner 构造成本。
已有实现中的数值处理保持原样；新增代码没有 clamp、fallback 或 broad try/except。

## Held-out synthetic diagnostic

Probe 完全 synthetic、held-out、non-private；每 epoch 256 个 pink-noise inputs 和 uniform random labels。
input RNG 为 seed+50000+epoch，label RNG 为 seed+60000+epoch，均独立于 curvature
input RNG（seed+10000+epoch）和 label RNG（seed+20000+epoch）。Probe 不依赖 beta，
同 seed/epoch 的 inputs/labels 完全一致；生成过程保持全局 RNG 状态。
Probe 不读取 private MNIST train gradients，不参与 factor construction、scale matching、model update 或超参数选择。

每层梯度矩阵采用既有 convention：output channels 为行，flattened input weights 为列，bias 是最后一列。
每个 microbatch（8 samples）用 torch.func 求 per-sample 梯度，仅保留该 batch，然后累积统计量。
应用实际训练 operator：Gtilde = scale_match × (C+damping I)^(-beta) G (A+damping I)^(-beta)。
在当前 factors 的 eigenspaces 中计算 Z=U_C^T Gtilde U_A，e_j=mean_i Z_ij²。
这里是 **uncentered second moment**，不是 centered covariance variance。

每层保存 d、D_eff=(sum e)²/sum(e²)、W=D_eff/d（同时命名 isotropy_score 和
whitened_dimension_fraction）、top ceil(.1d) energy share，以及达到 90% energy 所需最少方向数/d。
全模型 W_global、top10_energy_share 和 directions_for_90pct 均为按层参数方向数量 d 加权的层指标平均值。
包括 bias，所有层 d 之和等于模型参数数量。

每层先计算 r=e/mean(e)，合并所有层后降序排列，保存 0–100 percentile 上均匀的
1001 个 quantile 点作为压缩 ranked profile；不保存 per-sample gradients/Z。
最终曲线在同一 percentile 对两 seeds 的 final-epoch profile 取均值，含 y=1 reference。

白化测量发生在每个 epoch factors 构造后、private updates 前；accuracy 是该 epoch 更新后的 test accuracy，
clipping 是该 epoch 的训练统计。两者关联是描述性的，不意味着因果结论。
whitening metrics 只用于 diagnostics，不反馈训练或选择超参数。

## 执行

```bash
conda activate curve
python exp14d/run_exp14d.py
```

默认输出 `exp14d/results/formal/`。MNIST 直接读取现有 `exp1/data`，download=False，不复制数据。
需要 CUDA，沿用 exp14b runtime。启动前请确认要执行正式实验；开发验证只运行下面的 smoke。

```bash
conda activate curve
PYTHONDONTWRITEBYTECODE=1 pytest -q exp14d/tests -o cache_dir=exp14d/results/.pytest_cache
PYTHONDONTWRITEBYTECODE=1 python exp14d/run_exp14d.py --smoke
python exp14d/analyze.py exp14d/results/smoke
```

Smoke：beta={0,.25,.5}、seed=42、1 epoch、1 private batch、1×256 curvature samples，
保留完整 256-sample probe；sigma=0，仅验证功能，不提供正式隐私或 utility 结论。
正式实验不由测试启动。

## 输出

每完成一个 run 写出：

- `metrics.csv`：epoch training/evaluation、epsilon、clipping、norm quantiles、scale_match 和 global whitening。
- `whitening.csv`：每层每 epoch whitening metrics、direction count、probe sample count。
- `summary.csv`：每 run final epoch。
- `beta_summary.csv`：final epoch 按 beta 聚合 mean/std（sample std，ddof=1；只有一个 seed 时 std 留空）。
- `whitening_profile.csv`：每 run/epoch 的压缩 normalized directional energy profile。
- `config.json`：协议、随机种子和 diagnostic metadata。
- `isotropy_vs_beta.png`、`accuracy_vs_isotropy.png`、`clip_fraction_vs_isotropy.png`、
  `energy_concentration_vs_beta.png`、`ranked_energy_profile.png`。

图使用 final epoch；折线为 seed 均值，前三图另显示各 seed 散点，accuracy/clipping 图标注 beta。
Smoke 输出在 `results/smoke/`，测试与 smoke 日志在 `results/`。
