# Exp31：冻结 ViT backbone 的 DP LoRA 对照

目标：在相同隐私预算下，比较 synthetic DP-KFC / DP-KFC-A 是否优于 DP-AdamW。实现完成后仅执行轻量检查，不运行正式实验，也不预先生成正式实验结论。

## 固定协议

- 模型：`vit_tiny_patch16_224.augreg_in21k_ft_in1k`，复用 Exp22 的 pretrained 初始化及显式 q/k/v 转换；新建 CIFAR-10 head。
- CIFAR-10：Resize 224×224 bicubic → ToTensor → Normalize(mean=std=(0.5,0.5,0.5))，无增强。
- seed=42，5 epochs；logical/physical batch 均为 256，accumulation=1；drop_last，每 epoch 195 steps，总计 975 accountant steps。
- ε=3，δ=1e-5，C=1；沿用 Exp22/Exp30 的 RDP accountant、noise multiplier 和固定 batch accounting 约定。
- AdamW：lr=1e-3，weight_decay=0.01，betas=(0.9,0.999)，eps=1e-8。optimizer 仅接收可训练参数。

## LoRA 和 geometry

冻结所有原 backbone 参数，仅在 12 个 block 的 `attn.q_proj`、`attn.v_proj` 加入 LoRA。rank=8，alpha=8，dropout=0（无需 dropout 模块），增量为 BA。A 使用 PyTorch Linear 默认 Kaiming uniform 初始化，B 全零，因此初始增量和输出差异严格为零。

唯一可训练参数为上述 LoRA A/B 的 weight，以及新 classifier head 的 weight/bias。geometry 显式选择以下 48 个 Linear，绝不包含 base Linear 或 head：

```
blocks.{0..11}.attn.{q_proj,v_proj}.{lora_A,lora_B}
```

- DP-AdamW-LoRA：全部 identity geometry。
- DP-KFC-A-LoRA：g(A+λI)^(-0.4)，scale 恒为 1；不计算或应用 RMS matching。
- DP-KFC-LoRA：(G+λI)^(-1/2) g (A+λI)^(-1/2)。
- head 始终使用 identity geometry，但与 LoRA 一同参与全局范数裁剪及 DP noise。

复用 Exp22 的 synthetic pink noise（alpha=1）、Full KFC、BK clipping、DP noise、accountant、evaluation 逻辑。`clipping.py` 仅适配 BK 初始化以排除冻结参数及 token hook；`geometry.py` 复用 A covariance 估计，直接构造无 scale matching 的 A-only operator。damping 作为显式参数传给 builder，不修改默认常量。每 epoch 用 10×256 synthetic 样本重建 operator，synthetic physical batch=256。

RNG：synthetic X 为 seed+10000+epoch，labels 为 seed+20000+epoch，DP noise 为 seed+40000，私有数据 shuffle 为 seed。

## 严格 7 runs 与 GPU 分配

| GPU | 顺序执行 |
|---|---|
| 0 | DP-KFC λ=1e-3 → DP-KFC-A λ=1e-3 |
| 1 | DP-KFC λ=1e-2 → DP-KFC-A λ=1e-2 |
| 2 | DP-KFC λ=1e-1 → DP-KFC-A λ=1e-1 |
| 3 | DP-AdamW-LoRA（无 damping） |

四个 worker 并行；每个 worker 内 subprocess 顺序执行，失败直接报错。已有正式 run 目录直接报错，不 resume，不调整 batch，不跳过失败 run。

启动完整实验的唯一命令（仓库根目录）：

```bash
conda run --no-capture-output -n curve bash exp31/run_all.sh
```

## 文件与输出

所有新增代码、下载数据、HF/Torch/CUDA/Triton/matplotlib/XDG 缓存、TMP、日志及结果均位于 `exp31/`。复用模块只在当前进程内重定向 Exp22 的 ROOT 以控制其初始化缓存路径，不修改已有实验文件。

- `config.py`：固定参数和 grid；`lora.py`：LoRA、白名单及参数诊断。
- `run.py`：单次正式训练；`worker.py`、`run_all.sh`：GPU 调度。
- `geometry.py`、`clipping.py`：冻结 backbone 所需的局部适配。
- `analyze.py`：要求全部 7 runs 的 5 个 epoch 齐全，汇总仅取 epoch 5。
- `checks.py`：真实 pretrained ViT 的 2 样本检查，三种 geometry 的 BK 对照逐样本 autograd，验证一次 DP 更新后冻结参数逐值不变。
- `results/runs/<run_name>/config.json`：在训练循环之前保存配置。
- `results/runs/<run_name>/metrics.csv`：逐 epoch 指标，包括所有要求的参数、范数、裁剪、时间、隐私计数、更新检查及 BK diagnostics。
- `results/summary.csv`：严格 7 行；`results/paired.csv`：每 damping 一行，共 3 行。
- `results/damping_vs_{accuracy,clip_fraction,mean_clip_factor,norm_p99}.png`：log damping 横轴，AdamW 水平参考线。
- `logs/gpu{0,1,2,3}.log`：worker 日志；`logs/checks.log`、`results/checks.json`：轻量检查证据。

`accountant_steps` 为累计计数；`logical_steps`、`optimizer_steps`、`noise_events` 为当 epoch 计数（正式协议均为 195）。LoRA 范数为相应 A/B 参数合并的 L2 范数，q/v 范数同样表示对应适配参数的合并范数。冻结检查对照初始化快照，LoRA/head 更新指标要求各参数张量均发生变化。

正式训练尚未启动，因此正式 summary/paired/图将在七次训练全部成功后产生。

## 已执行的轻量检查

检查通过：7 runs、2/2/2/1 GPU 分配、固定 batch/LR/975 steps、48 个显式选中层、head identity、A-only scale=1、显式 damping、零 B 初始输出等价，以及三种方法的两样本 BK/autograd 对照和一次 DP 更新。总参数 5,600,074；可训练参数 75,658，其中 LoRA 73,728、head 1,930。模拟汇总检查得到 7 行 summary、3 行 paired 和 4 张图，仅保存于 `.cache/analysis_check/`。未启动完整训练。
