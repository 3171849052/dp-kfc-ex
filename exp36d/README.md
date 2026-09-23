# Exp36d: raw fractional DP-KFC-A exponent，C=2

仅在 pretrained full ViT-Tiny + CIFAR-10 上比较 seed=42 的 6 个 runs：
DP-AdamW 基线和 5 个 DP-KFC-A 指数实验。
Exp36d 内部比较统一使用 **C=2**，不能将旧 C=1 实验直接当作同条件 p comparison。
本目录不包含正式训练结果；轻量检查不启动正式实验。

## 启动

从仓库根目录启动全部正式实验的唯一命令：

```bash
conda run --no-capture-output -n curve bash exp36d/run_all.sh
```

四个后台 worker 固定绑定 physical GPU；每卡串行、卡间并行，等待全部完成且成功后运行分析。
失败时返回非零退出码，无动态调度、自动 resume 或自动降 batch。已有 run 目录不能覆盖。

| Physical GPU | 按执行顺序的 runs |
|---|---|
| 0 | `dp_adamw`, `dp_kfc_a_p01` |
| 1 | `dp_kfc_a_p02`, `dp_kfc_a_p05` |
| 2 | `dp_kfc_a_p03` |
| 3 | `dp_kfc_a_p04` |

## 算子与训练协议

固定 damping=1e-3，p ∈ {0.1, 0.2, 0.3, 0.4, 0.5}。
A-only 使用 `G_grad @ (A+dI)^(-p)`，外部 scale=1。
无 spectral normalization、RMS scale matching、alpha mixing 或 gain cap。
使用 Exp22 的对称化、float64 eigendecomposition 和负特征值截零约定。

固定模型 `vit_tiny_patch16_224.augreg_in21k_ft_in1k`，pretrained=true，full-model fine-tuning。
CIFAR-10 只读取仓库根目录 `data/`，`download=False`；224×224 bicubic resize，
mean/std 均为 (0.5,0.5,0.5)，无 augmentation。
AdamW：lr=1e-4，weight_decay=.01，betas=(.9,.999)，eps=1e-8；
5 epochs，seed=42，logical batch=256，physical batch=128，C=2，epsilon=3，delta=1e-5。
RDP 使用原 fixed shuffled logical-batch/drop_last 约定，975 总步数。
noise multiplier 由原 accountant 计算；Gaussian noise 在聚合后按 sigma*C 缩放再除以 batch size。

每 epoch 重建 geometry；pink probes 每 epoch 为 10×256 samples。
严格复用 Exp22 probes 的连续 RNG：seed+10000+epoch；A-only 不使用 synthetic labels。
初始化和 shuffle seed=42，DP noise seed=40042。
训练顺序保持 geometry build → private per-example gradients → geometry transform →
全局 C=2 clipping → aggregate → Gaussian noise → AdamW。
Calibration 只构建算子，不更新模型参数。

## 复用关系与文件

- `config.py`：固定 grid/protocol；复用 `exp35b.runtime.configure` 隔离所有输出与 cache。
- `geometry.py`：复用 `exp22.geometry` 的 A-only builder、pink probes、labels 与 transforms；
  复用 `exp35.geometry.SpectralAOperator`/diagnostics，仅注入 raw spectral exponent。
- `worker.py`：复用 `exp35.vit` 的数据和已有 Exp30 pretrained checkpoint 拷贝，
  `exp30.run` 的完整训练循环、`exp22.model` 的 pretrained ViT、
  `exp22.methods.Clipper` 的 BK clipping/noise、原 RDP accounting，
  `exp35.adapters.bind/prepare/MetricsSink` 的依赖注入和 metrics 边界。
- `analyze.py`：要求全部 6 runs 完整且计数一致后生成分析。
- `checks.py`：CPU 轻量数值与协议检查，不训练完整 ViT。
- `run_all.sh`：固定四 worker 启动器。

所有 runtime 写入均位于 `exp36d/`：日志在 `logs/`，共享 cache 在 `.cache/`，
每卡 cache 在 `workers/gpuN/.cache/`。
Pretrained checkpoint 从已有 `exp30/.cache/huggingface/hub/` 只读拷贝至本实验 cache，
保持原 pretrained 权重来源与 offline 配置。

## Metrics 与分析

每个 run 输出 `results/vit/runs/<method>/config.json` 和逐 epoch `metrics.csv`。
保留原训练/测试 loss、accuracy/best accuracy、epsilon/noise multiplier、clip fraction、
mean clip factor、transformed norm p50/p90/p99/max、累计 logical/optimizer/noise/accountant steps、
builder statistics、finite/update checks、operator state bytes，以及全部 `group_norm_*` 和 `layer_norm_*`。
A-only 额外保留 gain min/p10/p50/p90/p99/max 和每层 gain min/max。

正式完成后生成：

- `results/summary.csv`、`a_only_summary.csv`：epoch 5 全部指标，含 best accuracy。
- 按 p 列出 A-only accuracy、clip fraction、p90，打印到 `logs/analysis.log`。
- accuracy、test loss、clip fraction、mean clip factor、p90、p99、group norm vs p PNG 图：
  A-only 曲线加入本实验 C=2 DP-AdamW 水平基线。
- `results/layer_family_norms.csv`、`layer_family_norm_ratios.csv` 及 ratio 图：
  attention_qkv、attention_out、mlp_fc1、mlp_fc2、patch_head、identity。
  fc1/fc2 norm 为各层记录 norm 的平方和开根号，其余直接使用原 group RMS norm；
  ratio 分母为本实验 C=2 DP-AdamW 的 epoch 5 对应 family norm。

所有 vs-p 图与简表统一比较最后一个 epoch，best accuracy 另外保留在 summary 中。
缺失/未完成 runs 时分析直接失败，不输出伪正式结果。

## Lightweight checks

```bash
conda run --no-capture-output -n curve python -B exp36d/checks.py
```

检查 compile/import、shell syntax、6-run grid/GPU 分配、固定协议、真实本地 CIFAR-10 的
路径和 download=False、cache 隔离、所有 p 的 raw A 公式、
calibration 参数不变、小模型 BK clipping 与直接 per-example 梯度一致、
Gaussian noise 的 C=2 缩放和不完整分析拒绝。
检查记录：`logs/checks.log`。没有启动正式训练，也未生成正式 metrics/summary。
