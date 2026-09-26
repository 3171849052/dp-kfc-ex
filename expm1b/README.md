# ExpM1b: per-layer trace normalization

固定参考：`b5f56b55d198e9ddae866c7c5055fa779a5b4ed1`。
只比较 DP-KFM / DP-KFM-A，pink / public，MNIST / ViT，beta=.25/.5，seed=42，共 16 runs。
GPU 0/1 分别运行 KFM beta=.25/.5，GPU 2/3 分别运行 KFM-A beta=.25/.5；
每卡按 pink ViT、pink MNIST、public ViT、public MNIST 串行执行，四卡并行。

`mechanism.py` 的 `Shape` 为每个 affine block（包括 bias coordinate）设置
`tau_layer = layer_dimension / trace_R`。metric A factor 除以 sqrt(tau_layer)，
noise 乘以 sqrt(tau_layer)。identity 参数的 metric 和 noise 均为单位形状。
`bk.py` 仍先汇总全模型样本范数，每条样本只产生一个 clip factor，再用原始
activation/backprop 汇总 raw clipped gradients；不变换 optimizer signal。

数据、模型、geometry、BK、worker 协议代码基于 ExpM1 保留在本目录，避免导入
`expm1/__init__.py` 时创建其缓存。数据仅从根目录 `data/` 读取，download=False；
预训练权重与 ExpM1 一样从 Exp30 已有 checkpoint 复制到本实验缓存，离线使用。
A-only 不生成随机 pseudo-label、不构造 G、不在 geometry builder 中 backward；
Full ViT public 沿用独立确定性 10 类 pseudo-label。
保留 ExpM1 固定 private oracle 的只读 geometry alignment 诊断，不增加 oracle 实验条件。
这些未加噪诊断仅用于研究，不是 DP release。

`analyze.py` 只读引用 `expm1/results/runs/` 中对应 16 个 global runs 与两个
DP-SGD references；输出 final_summary.csv（34 个末轮结果）、paired_summary.csv
（16 对）、完整逐 epoch 诊断、accuracy/clipping 与 SNR/noise-share 图。
`vit_kfm_a_pink_beta025_*` 单独保存指定 ViT 条件的三方逐 epoch、geometry、group 诊断。
accuracy 使用第 5 epoch test_accuracy；best_accuracy 同时保留在 summary。

轻量验证（CPU toy，无正式模型训练）：

```bash
conda run --no-capture-output -n curve python -B expm1b/checks.py
bash -n expm1b/run_all.sh
```

正式启动（目录存在即失败；任一 GPU worker 失败后不执行分析）：

```bash
conda run --no-capture-output -n curve bash expm1b/run_all.sh
```
