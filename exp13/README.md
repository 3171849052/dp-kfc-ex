# Exp13: synthetic curvature in real DP training

研究问题：将 uniform-label KFAC curvature 换成 synthetic KFLR/GGN 或
KFRA-block，是否改善 clipping geometry，进而改善最终 DP utility？

- **DP-KFC**：uniform-label synthetic KFAC（Exp12 KFAC-U-1）。
- **DP-KFLR**：label-free synthetic exact-output GGN/Fisher within the same
  Kronecker factorization；softmax `diag(p)-ppᵀ` 的至多 9 个平方根方向传播。
- **DP-KFRA-block**：Exp12 的 SimpleCNN recursive block approximation，包含
  ReLU/MaxPool gates、same-site spatial channel blocks，丢弃跨位置 curvature；
  不是 full spatial KFRA。

直接调用 `exp12.curvature.estimate`，保持 bias augmentation、conv unfolding 和
A/C averaging convention。每层 `P(G)=L G R`，`L=(C+0.001I)^(-1/2)`、
`R=(A+0.001I)^(-1/2)`；double eigendecomposition 后转 float32，不做额外
geometric normalization、scale clamp。所有新增实现和结果仅在 exp13，
不修改 exp11/exp12。Exp12 Private-KFLR oracle 禁止用于本实验训练。

## 固定配置与配对

MNIST / `dp_kfac.models.SimpleCNN`，5 epochs，seeds **42、7**；batch 256，
shuffle、drop_last=True；SGD lr=0.5、momentum=0、weight_decay=0；epsilon=1、
delta=1e-5、max_grad_norm=1。每 epoch 重建一次，10×256 synthetic pink-noise。
初始模型由相同 seed 创建；DataLoader 使用独立同 seed generator；Gaussian
noise 使用同 seed+40000 的 device generator，同参数次序/形状抽样。

每个 seed/epoch 用 seed+10000+epoch 在 `fork_rng` 中一次生成完整 cache，
三个独立训练方法使用相同 input RNG schedule（各自模型 state 会随训练不同）。
KFC label 用独立 seed+20000+epoch generator；builder 保存/恢复 CPU 和所有
CUDA RNG。synthetic prediction statistics 在该 cache、更新前模型上计算。
**private MNIST examples 不参与 preconditioner construction**；builder 没有
private loader/input 参数。

Structured Ghost：先用 `Rᵀa`、`Lb` 的分块 spatial Gram 计算 transformed norm，
再 weighted ordinary backward，aggregate precondition 一次，加各向同性
Gaussian noise 后除 batch size，SGD step。训练不生成 per-example parameter
梯度，也不使用 GradSampleModule。空间 Gram tile=32，偏向简单实现而非速度最优。

三方法使用同一 clipping/noise mechanism 和 Exp11 的 RDP accountant /
shuffled fixed-batch convention；sigma 按相同 epsilon、delta、256/60000、
`5*floor(60000/256)=1170` steps **统一计算一次**。
这里沿用仓库固定批次的 accounting convention，不将其描述为 Poisson sampling。
训练 loss 和 clipping diagnostics 是未加噪的研究日志，不属于 DP 输出保证。

## 执行

需要 conda `curve`、CUDA，以及已有 `exp1/data/MNIST/raw` 数据。

```bash
conda run -n curve pytest -q exp13/test_exp13.py
conda run -n curve python exp13/run_exp13.py --smoke
```

仅一个 smoke：seed 42、三方法、1 epoch、1 private batch（256）、
1 synthetic batch（256）、sigma=0。检查有限 loss/norm/gradient/parameters 和
参数更新，写入 `exp13/results/smoke/`。零噪声 smoke **不提供 DP 保证**，
epsilon 留空，accountant_steps=0；单 seed 的 sample std 同样留空，不伪造数值。
测试仅包含共同 A、damped whitening、KFLR/KFRA-block Ghost aggregate 对
逐样本显式参考、builder RNG 隔离。按研究原型范围不新增 golden/bitwise regression、
hardening 或额外 smoke；初始化由同一函数保证。

正式完整实验（本次不自动启动）：

```bash
conda run -n curve python exp13/run_exp13.py
```

结果：`exp13/results/metrics.csv`（每 epoch），`summary.csv`（每 method/seed
最后 epoch，附累计时间和跨 epoch 峰值），`paired_summary.csv`（三对差值），
`method_summary.csv`（两 seed 描述统计，sample std，未做显著性检验）。
Accuracy 单位为 [0,1]；paired difference 均为名称前者减后者；build time mean
是每个 seed 累计 build 时间的均值；total runtime 含 build、training、evaluation
及 epoch 统计，不含 CSV 写出、数据加载初始化和 warmup。仅两 seeds，不能据此宣称
统计显著性。重新生成 summary：`python exp13/analyze.py [结果目录]`。

CUDA：benchmark=False、deterministic=True、两项 TF32=False；复用 Exp12 runtime
在计时前 warmup CUDA/cuBLAS。build（含 cache、state metrics、inverse sqrt）与
training 分别同步计时和重置 peak allocated memory；total peak 是两阶段最大值。
不在 batch 内 empty_cache。额外 state metric forward 单独记录，不混入 curvature
builder budget。

| Builder | 正式每 epoch forward / VJP / reverse vectors | smoke |
|---|---:|---:|
| DP-KFC | 10 / 10 / 2560 | 1 / 1 / 256 |
| DP-KFLR | 10 / 90 / 23040 | 1 / 9 / 2304 |
| DP-KFRA-block | 10 / 0 / 0 | 1 / 0 / 0 |

reverse vectors 按每个 sample 的 output reverse direction 计数；一次 batched
VJP 含 256 vectors。KFRA-block 使用解析递推，没有 autograd VJP，但仍有递推开销。
三个方法另有正式 10 / smoke 1 次 synthetic state metric forward。
