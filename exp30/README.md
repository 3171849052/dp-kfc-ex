# Exp30：DP-KFC / DP-KFC-A damping 单变量 sweep

仅新增 Exp30 文件；复用 `exp22.model` 初始化及全参数训练模型、
`exp22.geometry` 层选择/合成数据/geometry、`exp22.methods.Clipper` 的 BK clipping
及 DP noise，以及 `exp22.analyze.accuracy_auc`。不更改 Exp22 文件。
`exp22.run_one` 导入时会创建并指定 Exp22 缓存，因此只在本地保留其简单的数据、
shuffle loader、评估和训练编排逻辑；预处理及计算公式相同，不复制核心算法。
模型导入前仅将进程内的 `exp22.config.ROOT` 指向 Exp30，以隔离其缓存路径副作用。

## 固定协议

- 模型：`vit_tiny_patch16_224.augreg_in21k_ft_in1k`，全部参数参与微调。
- CIFAR-10：训练集 50,000；完整测试集。Resize 到 224×224（bicubic），
  ToTensor，Normalize(mean=std=(0.5,0.5,0.5))，无数据增强，完全沿用 Exp22。
- seed=42，epochs=5，epsilon 目标=3，delta=1e-5，clipping C=1。
- AdamW：lr=1e-4，weight_decay=0.01，betas=(0.9,0.999)，eps=1e-8。
- logical batch=256，physical batch=128；shuffle=True、drop_last=True。
  每 epoch 195 个 logical/optimizer/noise steps，共 975 次 accountant steps。
- synthetic geometry：每 epoch 10×256 pink noise，alpha=1，synthetic physical batch=256。
- A_POWER=0.4。DP-KFC 使用 full geometry；DP-KFC-A 映射到 Exp22 `dp_kfc_a_bk`。
- 每 epoch 显式调用 `build_from_batches(..., damping=damping, power=cfg.A_POWER)`，
  不通过改默认常量进行 sweep。其他参数不 sweep。
- Exp22 的 RDP fixed logical-batch convention，sample_rate=256/50000；
  noise multiplier 按 975 steps 计算。epsilon 列记录逐 epoch 实际 accountant 值。
- 每 run 启动独立 Python 子进程，重新创建模型、optimizer、shuffle RNG、
  DP noise RNG 和 accountant。shuffle/init seed=42；DP noise seed=seed+40000；
  每 epoch synthetic X seed=seed+10000+epoch，synthetic Y seed=seed+20000+epoch。

## GPU 调度及启动

| GPU | damping（每个值依次运行 DP-KFC、DP-KFC-A） | run 数 |
| --- | --- | --- |
| 0 | 1e-4, 3e-4 | 4 |
| 1 | 1e-3, 3e-3 | 4 |
| 2 | 1e-2, 3e-2 | 4 |
| 3 | 1e-1 | 2 |

从仓库根目录执行：

```bash
conda run --no-capture-output -n curve bash exp30/run_all.sh
```

使用已有 `curve` 环境。入口先在 `exp30/data/` 下载/校验 CIFAR-10，随后并行启动
四个 worker；每卡顺序执行，子进程内使用映射后的 `cuda:0`。所有 worker 成功后
才执行分析；失败返回非零状态，不 fallback、不降低 batch、不 resume、不跳过。
已有 metrics 的 run 会直接报错，重跑应先自行归档旧结果。
HF、Torch、CUDA、Triton、matplotlib、XDG 缓存和临时目录均在 `exp30/.cache/`；
禁用 Python bytecode 写入，不使用 Exp22 缓存。首次运行需要下载预训练权重。

## 输出

- `logs/gpu0.log` 至 `logs/gpu3.log`：各 worker 的训练输出及错误。
- `results/runs/<method>_damping_<value>/metrics.csv`：逐 epoch 指标，例如
  `dp_kfc_damping_0.001`、`dp_kfc_a_damping_0.001`；每 run 同时保存 `config.json`。
- 指标包括 method、damping、epoch、train/test loss、test/best accuracy、accuracy AUC、
  clipping fraction/factor、transformed norm p50/p90/p99/max、group/layer norm、
  noise multiplier、epsilon、logical/optimizer/noise/accountant steps、builder/train 时间、
  operator bytes 及 Exp22 原有诊断指标。group/layer norm 是每组/层平方范数均值开根号。
  accuracy 使用 [0,1] 比例，AUC 沿用 Exp22 对已观测 epoch 点的梯形积分。
  logical/optimizer/noise steps 是当 epoch 计数，accountant steps 是累计计数。
- `results/summary.csv`：严格 14 行，每 run 取 epoch 5；缺少或不完整 run 直接报错。
- `results/dp_kfc_ranking.csv`、`results/dp_kfc_a_ranking.csv`：按 final accuracy 降序，
  同分按 damping 升序。
- `results/paired.csv`：7 行，包含 damping、dp_kfc_accuracy、dp_kfc_a_accuracy、
  delta_accuracy=dp_kfc_a_accuracy−dp_kfc_accuracy。
- 六张 final epoch 图：`damping_vs_accuracy.png`、`damping_vs_test_loss.png`、
  `damping_vs_clip_fraction.png`、`damping_vs_mean_clip_factor.png`、
  `damping_vs_norm_tail.png`（p90/p99/max）、`damping_vs_delta_accuracy.png`。
  文件位于 `results/`，横轴均为 log scale；前五图叠加两个方法，最后一图显示成对差值。

只做导入检查（不会启动训练）：

```bash
conda run --no-capture-output -n curve python -B -c 'import exp30.run, exp30.worker, exp30.analyze; from exp30.config import grid; print(grid())'
bash -n exp30/run_all.sh
```
