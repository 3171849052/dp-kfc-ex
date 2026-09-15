# exp17: 同轨迹 Fisher / K-LBFGS factor 几何诊断

新增目录完全独立；只读复用现有模型、DP、pink-noise 工具和 exp15 的
InverseLBFGS。`Diagnostic._synthetic_pair` 是 exp15 secant 流程的本地副本，
唯一插入操作是在 base backward 后采集 Fisher。每个 checkpoint 固定
seed=42+10000+epoch，生成一次完整 `(x,y)` 列表；Fisher 直接使用 secant
base backward 的 activation/backprop，因此 probes、标签、参数逐元素一致。
每次 lookahead 重置到 checkpoint 参数；记忆和 activation/pair EMA 跨 epoch 保留。
Fisher 每 checkpoint 从零累计，不使用跨 epoch EMA。diagnostic 无 private loader
参数，在 fork_rng 内使用独立普通模型，不改变 private 模型或训练 RNG。

## 协议

MNIST + SimpleCNN，seed 42，batch 256，drop_last=true，5 epochs，
1170 private steps。checkpoint 位于每轮读取 minibatch 前：0/234/468/702/936。
Private identity DP-SGD：sum-loss backward → per-sample norm → clipping=1 →
Gaussian noise → SGD(lr=.5)。RDP target epsilon=1、delta=1e-5，使用
fixed-batch shuffled、q=batch/train_size 的既有 accounting convention。
无 curvature private preconditioning，accuracy 只作为训练记录，不比较优化方法。
每 checkpoint 2560 pink-noise samples / 256 batch，共 10 batches。
K-LBFGS memory=100、initial_scale=1、factor_damping=.1、两种 decay=.9、
lookahead lr=.5；Fisher damping=.001。

**Fisher G_F 是 gradient covariance；K-LBFGS B_G 是 secant/pre-activation
Hessian curvature。它们不是理论上相同的矩阵，本实验测量几何一致性和冲突。**
A 使用带 bias 的 unfolded activation covariance，G 使用空间位置展开后的
backprop covariance，与 exp16c 的 normalization 一致。

## 输出与指标

`results/formal/matrices/epoch{e}/{layer}.pt` 保存 A_F/G_F、Q_A/Q_G、
B_A/B_G、generalized matrix/eigenvalues、Q powers 和 transformed factors。
只构造 conv1/conv2/fc1/fc2 的逐层 factor，绝不构造完整 Kronecker 矩阵。
Q=sym(factor.hv(I))，B 用正特征值分解求逆；全部几何计算 float64。

- factors.csv：F_/B_ spectrum（min/median/max/trace/Frobenius/effective rank），
  condition_F 对 F+.001I，condition_B 对 B；cosine、optimal_scale、shape_error、
  normalized_commutator、top_k_overlap（k=min(10,n)）；generalized spectrum 摘要。
- generalized_spectra.csv：完整 F_lambda^(-1/2) B F_lambda^(-1/2) 特征值。
- transformed_conditions.csv：q=.25/.5，Q^q F Q^q 加 .001I 的 condition 和原始比值。
  重点观察 A ratio<1、G ratio>1、G 在 .5 时是否比 .25 更严重；不预设结论。
- operator_metrics.csv：F_lambda^(-q) 与 Q^q 的 cosine、conditions、固定种子
  32 Gaussian row vectors 的 action cosine 均值/总体标准差。
- training.csv / metrics.csv：epoch / private-step 训练与隐私日志。
- plots/：condition_vs_depth.png、alignment_vs_epoch.png、generalized_spectrum.png、
  transformed_condition_ratio.png。仅 formal 自动绘图，使用 matplotlib。

Effective rank=exp(entropy(normalized nonnegative eigenvalues))，仅此指标去掉
raw covariance 的浮点负尾；所有逆和 condition 都严格检查正定，不裁剪修复。
Median 使用 0.5 quantile。每次 geometry 检查 power API 与 dense power 一致。

## 验证与执行

使用 conda `curve`。所有新代码、数据下载、缓存和输出位于 exp17。
依赖既有 sibling `kbfgs_neurips2020_public`，不提供 fallback。

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n curve python -m pytest exp17/tests/test_exp17.py -q -o cache_dir=exp17/results/pytest_cache
PYTHONDONTWRITEBYTECODE=1 conda run -n curve python exp17/run_exp17.py --smoke
PYTHONDONTWRITEBYTECODE=1 conda run -n curve python exp17/tests/check_smoke.py
conda run -n curve bash exp17/run_all.sh
```

Smoke 使用 CUDA、缩小 CNN（2/2 channels，fc1=8）、train subset=8、test subset=32、
batch=4、2 epochs、synthetic=24/8；不用于研究结论。需求中的五 checkpoint
验证另由 pytest 在同一缩小模型协议上完成。正式实验使用原版 SimpleCNN。
输出目录必须不存在，避免混入前次 CSV；重复运行需显式移动旧结果。
根据用户最终指令，验证通过后将完整实验后台启动，不追踪其完成情况。
