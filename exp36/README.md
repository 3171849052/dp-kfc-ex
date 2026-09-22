# Exp36

从仓库根目录启动整个实验：

```bash
conda run --no-capture-output -n curve bash exp36/run_all.sh
```

固定 physical GPU 3，三个独立进程依次训练、诊断、作图。`train.py` 复用 Exp35 ViT DP-AdamW（最终为 Exp30 原训练循环）：pretrained ViT-Tiny，全模型 AdamW lr=1e-4、wd=.01，5 epochs，seed42，epsilon3/delta1e-5/C1，logical/physical batch256/128。初始化/每次 epoch 评估后只保存 state_dict，不采样随机数、不改变模型参数或训练 RNG。无 resume。日志、cache、checkpoints、results 全部在本目录；预训练 cache 从 Exp30 复制；数据仅用根目录 data/，download=False。

`diagnose.py` 在训练结束后逐个加载 epoch_0–5，模型 eval、no_grad。private_ref/private_replica 为 seed42 的同一次 50000 元素 permutation 的前两个不重叠 2560 样本段；跨 checkpoint 固定，indices 保存至 CSV。这里的“不重叠”仅指两组诊断集；两者均属于 reference trajectory 的训练集，不是 held-out 数据。

pink 严格复用 Exp22/35 `synthetic_stream(42, epoch)` 的 10×256 probes。white 为实数 IID Gaussian，shape 256×3×224×224，与 pink 一样逐图除以 std 再乘 .5，不做频域滤波；不额外施加 CIFAR transform。每个 epoch 的 probe seed 为 `42+10000+epoch`。真实图像采用 Exp35 的 resize/bicubic/Normalize(.5,.5)，无 augmentation。

A 累加直接复用 Exp22 `build_a_operator.__wrapped__`，仅将最终 operator 构造器替换为 covariance 容器；使用原 `flatten_linear_input` 的 `(B*T,d)` 和 bias augmentation，不引入 token-aware estimator。诊断矩阵不进入 optimizer 或自动参数选择。patch_embed 是 Exp22 转换后的 Linear，q/k/v 为独立 Linear，全层结果保留 block index（patch/head=-1）。

指标使用 float64，A/P 的 Frobenius cosine 与相对 Frobenius 误差比较完整矩阵。`P=(A+.001 I)^(-.4)`，不做谱归一化或 scale matching。effective rank 定义为 `exp(-sum(p log p))`，p 为非负特征值归一化。top-k overlap 为 `||Qr.T Qe||_F^2/k`；比较 A 的最大 k 个特征向量，k=16/32。重复特征值跨越截断边界时，子空间指标依赖 eigensolver 选择的基。记录 reference/estimate 的 operator gain 分位数及极值。

正式输出：

- `checkpoints/epoch_0.pt` 至 `epoch_5.pt`，`results/checkpoints.csv`。
- `results/private_indices.csv`、`layer_metrics.csv`、`group_metrics.csv`。
- `results/spectra/epoch_{0..5}.npz`：四种 source、全部层的升序 A 特征值，key 为 `source::layer`；P 谱可由固定公式直接获得。
- family summary 对层等权平均（attention_qkv 包括每个 block 的 q/k/v），同时记录 layer_count。
- `results/figures/`：pink cosA/cosP heatmap，三种 estimate 的 family-wise cosA/cosP/relative-Frobenius vs epoch，pink family alignment，epoch0/5 六个代表层的四种来源 spectrum。

这些包含 private/oracle 数据的输出仅为非 DP research diagnostics，不属于 DP 发布结果。未做可选 Exp35 norm-ratio 散点图。

轻量检查（不启动正式训练）：

```bash
conda run --no-capture-output -n curve python -B exp36/checks.py
```

检查 compile/import、shell、GPU、真实数据路径/download=False、固定且不重叠 indices、toy sequence bias-augmented covariance、A/P 矩阵公式、旋转但同谱矩阵、子空间 projector 公式、entropy effective rank，以及用 toy 模型调用实际 checkpoint 注入边界验证 epoch0–5 和 RNG 保持；toy checkpoint 仅写入自动删除的本目录临时目录。
