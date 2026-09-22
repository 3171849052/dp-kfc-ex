# Exp33d — Synthetic Wiener-A noise temperature

只改变 post-noise Wiener gain：`h = lambda / (lambda + beta * tau2)`，其中
`tau2 = (sigma * C / 256)^2`。beta 不进入 DP 噪声生成、clipping、accountant 或优化器。
不使用 gamma、RMS scaling、LR sweep、private gradient history 或自动 beta selection。

唯一正式启动命令（curve 环境）：

```bash
conda run --no-capture-output -n curve bash exp33d/run_all.sh
```

脚本固定 `CUDA_VISIBLE_DEVICES=3`，顺序运行 `dp_adamw`，再运行 beta 为
`1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7` 的七个 Wiener-A run。
每个 run 使用新的 Python 进程，独立初始化 pretrained model、optimizer、RDP accountant 和 RNG。
已有任一正式 run 目录时直接报错，不 resume、不跳过。失败即停止。

协议直接复用 Exp22 explicit ViT conversion 和 Exp33/Exp33c：
`vit_tiny_patch16_224.augreg_in21k_ft_in1k`，pretrained full-model fine-tuning，
74 Linear 右乘 Wiener-A（含 bias 列），LayerNorm、cls token、positional embedding 为 identity。
CIFAR-10 直接只读 `exp30/data/`，不下载、不复制；bicubic resize 224，ToTensor，
mean/std 均为 0.5，无 augmentation。
seed=42，5 epochs，epsilon=3，delta=1e-5，global C=1，logical/physical batch=256/128，
每 epoch 195 steps，共 975 noise/optimizer/accountant steps。
全参数 AdamW LR=1e-4，betas=(0.9,0.999)，eps=1e-8，weight_decay=0.01。

每 epoch 从当前模型重新构建 synthetic covariance，复用 Exp33 的 estimator、eigendecomposition
和 global clipping：10 个 logical batches，每个 256 个 alpha=1 pink-noise 图像，physical batch=128。
synthetic x/y seeds 分别为 seed+10000+epoch / seed+20000+epoch；shuffle seed=42；
DP noise seed=seed+40000。beta=1 与 Exp33 Wiener-A 等价。
DP-AdamW 完全不构建或应用 operator。

`metrics.csv` 每 epoch 记录训练、clipping、privacy、gain、spectrum/SNR、clean-shadow 和真实
AdamW 参数更新。谱统计按谱方向拼接（不是层均值的无权平均），group 字段命名为
`group_<metric>_<group>`。near-zero 定义 `lambda/tau2 < 1e-8`，gain fractions 使用严格大于。
`wiener_norm_ratio` 为每 step 全参数 filter 后/前 L2 范数比的 epoch mean，附 step 分位数。
update norm 是每 step 真正 `theta_after - theta_before` 的分组 L2 范数，再取 epoch mean；
relative update 除以该组更新前参数 L2 范数加 1e-30。NSR 沿用 Exp33c：误差 L2 / clean L2。
`logical_steps` 为当 epoch 步数，其余三个 step counters 为累计步数。

clean shadow 只用于 cosine/NSR diagnostics，不传入 builder、filter、optimizer 或参数选择。
这些原始私有梯度衍生诊断沿用研究协议，不作为训练决策输入。
DP-AdamW Wiener-specific CSV 字段为空，raw clean-shadow 和实际 update diagnostics 仍记录。

所有输出、cache 和临时文件限制在 exp33d/。8 runs 完成后 `analyze.py` 验证完整 grid，
生成 epoch-5 `summary.csv`（8 行）、`paired.csv`（7 行）和 12 张 log-beta 横轴图。
DP-AdamW 的实测指标或 identity gain/norm=1 显示为水平线；synthetic spectrum 对 identity
无定义，因此 spectrum 图不虚构 baseline。不会预先生成正式结果。

轻量检查（CPU，不加载 pretrained 权重、不训练正式模型）：

```bash
conda run --no-capture-output -n curve python -B exp33d/checks.py
```

检查独立 per-example clipping/covariance oracle、Exp33 beta=1 逐位一致、gain bounds/单调性、
固定 RDP、8 个方法的两步显式 noise/filter/AdamW 参考路径、真实更新范数、74 Linear 转换、
RNG、缓存位置、完整汇总与 12 张图以及不完整结果拒绝。
