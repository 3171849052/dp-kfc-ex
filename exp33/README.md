# Exp33: Synthetic Wiener

在 pretrained `vit_tiny_patch16_224.augreg_in21k_ft_in1k` 上进行 CIFAR-10 全参数 DP 微调。直接复用 Exp22/Exp30 的权重转换、seeded classifier、图像 transform、BK 全模型 clipping、shuffle/noise RNG 和 RDP convention，不重新设计 ViT。

| GPU | Run |
|---|---|
| 0 | DP-AdamW |
| 1 | DP-Wiener-A |
| 2 | DP-Wiener-Full |
| 3 | DP-KFC-A-0.1 |

固定 seed=42、5 epochs、epsilon=3、delta=1e-5、C=1、logical batch=256、physical batch=128、accumulation=2。AdamW lr=1e-4、betas=(0.9,0.999)、eps=1e-8、weight_decay=0.01。每轮 195 steps，最终 accountant/optimizer/noise events 均为 975。metrics 中 logical_steps 为当轮计数，另三项为累计计数。RDP 沿用固定 batch 抽样率 256/50000 的既有 convention。

每个进程独立初始化 pretrained model、optimizer、accountant、shuffle generator(seed)、noise generator(seed+40000)。每轮独立 pink generator(seed+10000+epoch) 与 labels generator(seed+20000+epoch)。Synthetic 输入沿用 Exp30 pink FFT，3×224×224、alpha=1，10×256 samples/epoch；synthetic physical batch 明确取 128（原 Exp30 导入的 Exp22 常量实际是 256，此处按实验要求覆盖）。数据及已有 pretrained cache 复制到本目录，读取复用代码不会在旧实验目录写 cache。

## 算子

Synthetic Wiener 复用 `Clipper(operator=None)` 的全模型 raw per-example norm 与全局裁剪，然后取得无噪声 logical batch mean gradient。BK 使用因子精确重建 per-example norm 与 clipped aggregate，不保留全模型 per-example weight tensors。与 private clipping 完全相同，数值实现为 `min(1, C/(norm+1e-6))`，沿用 Exp30 稳定项，非逐层 clipping。

每个 Linear 将 weight/bias 合并成 Z，10 个 batch 平均构造 `SA=mean(Z.T@Z/m)`；Full 同时构造 `SG=mean(Z@Z.T/n)`。以 float64 累积与 eigendecomposition，先对称化，只将负数值特征值截到零；最终应用使用 float32。

A-only: `h=λA/(λA+tau2)`，`Z_filtered=((Z@UA)*h)@UA.T`。
Full: `alpha=max((mean(λA)+mean(λG))/2,1e-12)`，`s=λG[:,None]*λA[None,:]/alpha`，`H=s/(s+tau2)`，`Z_filtered=UG@((UG.T@Z@UA)*H)@UA.T`。
其中 `tau2=(sigma*C/256)^2`。没有 activation covariance、private gradient history、residual controller 或参数自适应。所有 gains 位于 [0,1]，每次应用检查每个 Linear 的 Frobenius norm 不增加（相对 2e-5、绝对 1e-8 数值容差）。LayerNorm、cls token、positional embedding 保持 identity。

Private Wiener 顺序严格为 raw per-example gradient → 全局 clipping → 聚合 → Gaussian noise → Wiener → AdamW。DP-AdamW 同样顺序但无 Wiener。KFC reference 直接复用 Exp30 的 no-scale helper 与 Exp22 A builder，damping=.1、power=.4、scale=1；作用于 per-example gradient，随后全局 clipping、聚合、noise、AdamW。

## Diagnostics 与隐私边界

metrics.csv 包含 loss/accuracy/AUC、accounting、clipping/raw norms、计时、operator storage、参数更新检查、Wiener gain/signal 分位数及全梯度 norm ratio。四组 attention_qkv、attention_out、mlp、patch_head 分别记录元素加权 mean gain 与组内合并 L2 norm ratio（每 epoch 取 step 平均）。KFC 的 norm 标记 transformed_norm，避免误称 raw identity norm。

`cos_raw_clean`、`cos_filtered_clean`、`nsr_raw`、`nsr_filtered` 使用 clipping 后加噪前的 clean shadow，**仅为 non-private research diagnostics，不是 DP-publishable outputs**。clean shadow 只流入诊断函数，不流入 fitting、operator、optimizer update 或 hyperparameter adaptation。DP-AdamW 记录 raw 字段，filtered/Wiener 字段为空。同样，private train loss、clip/norm statistics 等日志不能当成经 DP 保护的发布结果；这里 accountant 描述的是训练机制。

## 文件与执行

`config.py` 固定协议和四卡分配；`geometry.py` 复用 KFC；`wiener.py` synthetic estimator、spectral operator 和 post-noise boundary；`run.py` 训练及 epoch metrics；`worker.py` 每卡一个 run；`checks.py` 轻量验证；`analyze.py` 完整结果汇总及 7 张图；`run_all.sh` 并行启动并在成功后分析。

轻量检查：`conda run --no-capture-output -n curve python -B exp33/checks.py`。

完整实验唯一入口：

```bash
conda run --no-capture-output -n curve bash exp33/run_all.sh
```

已有任何正式 run 目录立即报错；无 fallback、batch 自动调整或 resume。日志在 `logs/`，所有 cache/TMP 在 `.cache/`，每个 run 的 config.json 在训练前写入，metrics.csv 每 epoch 更新。四个 worker 全部成功后生成严格 4 行 epoch-5 summary.csv、paired.csv 及所需七张图。失败返回非零状态并保留日志，不跳过失败结果。

后台启动时，launcher PID 保存在 `logs/launcher.pid`，总日志为 `logs/launcher.log`，各 GPU 日志为 `logs/gpu0.log` 至 `logs/gpu3.log`。
