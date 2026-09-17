# Exp25: CrossViT CIFAR-100, BK+Ghost Differentiation

所有新增代码、下载、缓存、日志和结果都在本目录。运行环境为 conda `curve`。
复用 `src/dp_kfac/models.py::CrossViTClassifier` 和
`exp24/geometry.py::{AOnlyOperator,FullKFACOperator}`；单线性层 BK+GD
采用 `exp21/bk.py` 的真实参数禁用梯度、output anchor 求 backprop、BK 重建语义。
不实例化 Opacus GradSampleModule，不生成逐样本参数梯度。

固定协议：预训练 `crossvit_tiny_240`，backbone 冻结并保持 eval，唯一可训练层
`Linear(288,100)`，28,900 参数。CIFAR-100 train/test，Resize(240,240)、
ToTensor、ImageNet mean/std，与原 CrossViT 实验一致，无随机 augmentation。
Adam(lr=1e-3, betas=(0.9,0.999), eps=1e-8, weight_decay=0)，5 epochs，
logical/physical batch 均为 256，drop_last，C=1，195 steps/epoch，共 975 steps。
seeds=(42,7,91,23,58)，七种 condition，合计 35 runs。

每 epoch 起始只取 **1 个 256 样本辅助 batch** 并重新构建 geometry：

- identity: `dp_sgd_bk_gd`，无辅助数据。
- A-only: `dp_kfc_a_{public,pink,oracle}_bk_gd`，U_A=s(A+1e-3 I)^(-0.4)。
- Full: `dp_kfc_{public,pink,oracle}_bk_gd`，U_A=(A+1e-3 I)^(-1/2)，
  U_G=(G+1e-3 I)^(-1/2)。

把 bias 合并为 activation 的常数 1 列；A=a^T a/B。
Full 的 G=b^T b/B，其中 b 是 summed cross-entropy 对 logits 的导数
（即每样本 softmax-onehot，与 exp24 的 operator/factor 约定一致）。
按本实验给出的公式只加 1e-3 damping，不叠加旧 covariance helper 的 1e-5 jitter；
也不把 mean-loss 的 1/B 缩放引入 G。
A-only 完全不使用 label，RMS matching 直接复用 exp24：
`s² = Σ λ/(λ+d) / Σ λ(λ+d)^(-0.8)`（输出维数权重在单层中约掉）。

public 来自 CIFAR-10 train；Full 使用真实 0..9 label，与原实验的标签行为一致。
oracle 来自 CIFAR-100 train，Full 使用真实 0..99 label；它是 **非部署型 upper-bound**，
使用未私有化的私有 geometry，不能宣称整体 epsilon=8 DP。
pink 使用原 CrossViT 的 complex-white FFT、1/f 振幅、去 DC、每样本 std=0.5
输入空间噪声；不对已在模型输入空间的噪声再次做 ImageNet normalization。
Full pink 使用独立 RNG 的随机 0..99 label。

BK+GD 第一遍清空 grad，关闭 classifier 的 requires_grad，在 frozen features 上产生
output anchor，通过 autograd.grad(sum(loss), anchor) 得到 b，恢复参数标志。
转化 a'=a U_A^T、b'=b U_G^T 后，范数为 ||a'|| ||b'||，
clip factor=min(1,C/(norm+1e-6))，聚合为 (c b')^T a'。
不需要第二次参数 backward。七个条件共用此路径、同一个矩阵空间的各向同性
Gaussian noise、除以 batch size、Adam update 和 accountant step。

RNG 为 classifier 初始化、private shuffle、DP noise、pink image、pink label、
oracle calibration、public calibration 分离。shuffle 每 epoch 重置为固定独立 stream；
DP noise 在 run 内持续推进。每条 epoch 结果保存初始化和 minibatch 顺序 SHA256，
分析时校验相同 seed 的配对一致性。固定 CUDA deterministic algorithms，禁用 TF32。

Opacus RDP 使用 q=256/50000，显式 steps=975 校准 sigma，epsilon=8、delta=1e-5，
smoke 也使用正式 sigma，但只累计实际执行的 step。这里沿用原实验的 fixed-size
shuffle/drop_last + Opacus sampled-Gaussian RDP 记账约定；该名义数值并非针对
shuffle-without-replacement 的独立隐私证明。训练 loss、clipping/norm diagnostics
也来自未私有化私有统计；日志为研究用途，不属于训练 accountant 覆盖的发布物。

```bash
conda run -n curve python -m pytest exp25/test_exp25.py -q
conda run -n curve bash exp25/run_all.sh --smoke
conda run -n curve python -B exp25/analyze.py --smoke
```

smoke 使用真实预训练 CrossViT 和真实数据，seed=42，覆盖全部七个条件，
每条件 1 epoch、1 private step、256 test samples，辅助 batch 仍为 256。
它不是正式精度结果，也不能估计 5-seed std（输出 NA）。
单元测试的模型结构检查不下载预训练权重；显式 reference 仅在小矩阵测试中构建。

正式运行（35 runs，顺序执行）：

```bash
conda run -n curve bash exp25/run_all.sh
conda run -n curve python -B exp25/analyze.py
```

结果在 `results/{smoke,formal}/{method}_{seed}.jsonl`，每 epoch 一行，
日志在各自 `logs/`。重新运行同一 condition/seed 会覆盖本目录对应结果。
分析严格要求完整 seed/epoch/step 数，输出最终 epoch accuracy 和 best accuracy
的 mean±sample std、逐 seed 相对 DP-SGD 的最终 accuracy delta、10,000 次配对
bootstrap mean delta 的 percentile 95% CI，并写入 `summary.json`。
accuracy 在 JSON 中为 0..1，终端为百分比；delta 终端为百分点。
结果还包含 train/test loss、epsilon、sigma、step 数、clip fraction/factor、norm
p50/p90/p99/max、A/G 谱范围/trace/加 damping condition、p/RMS scale、校准时间
（含辅助数据准备）、private training wall time、CUDA peak allocated bytes 和参数数目。

已有项目 `data/` 若有完整 CIFAR 解压目录则只读复用，否则下载到 `exp25/data/`。
模型权重保存在 `exp25/cache/`，不修改已有实验或其结果。
