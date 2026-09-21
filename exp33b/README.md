# Exp33b: Synthetic Wiener-A scale × Linear LR

仅实现 Synthetic Wiener-A 的 4 LR × 2 scale 二因素消融。正式训练尚未运行。

唯一完整实验启动命令（仓库根目录）：

```bash
conda run --no-capture-output -n curve bash exp33b/run_all.sh
```

GPU 0/1/2/3 分别负责 Linear LR 1e-4/3e-4/1e-3/3e-3；各 worker 顺序启动 none、rms_match 两个独立进程。严格八个 run；已有正式 run 目录即报错，不 resume、不跳过、不自动调参。训练直接读取 Exp30 已准备好的 `exp30/data/cifar-10-batches-py`，不重新下载或复制 CIFAR-10。

复用 Exp33 的 Exp22 `initialize`、explicit ViT conversion、Clipper、synthetic_stream 和 AUC 定义；训练循环由 Exp33 最小修改。模型为 `vit_tiny_patch16_224.augreg_in21k_ft_in1k`，`pretrained=True`，全参数训练，74 Linear（含 bias）采用 Wiener；LayerNorm、CLS、position embedding 保持 identity。CIFAR-10 bicubic resize 224、ToTensor、mean/std=0.5，无增强。seed=42，5 epochs，195 logical steps/epoch，975 total，batch=256、physical=128、accumulation=2。RDP epsilon=3.0、delta=1e-5、C=1.0，与 Exp33 使用相同固定 batch accountant 约定。AdamW 两组，Linear LR 来自 grid，identity LR 恒为 1e-4；betas=(0.9,0.999)、eps=1e-8、weight_decay=0.01。

每个 epoch 按 Exp33 构造 10×256 pink-noise synthetic samples（alpha=1，3×224×224），physical=128。synthetic x seed=42+10000+epoch、labels seed=42+20000+epoch；shuffle seed=42、DP noise seed=42+40000。每个 run 独立初始化模型、optimizer、accountant 和所有 RNG。global clipping 完全复用 Exp33（包括原实现范数分母的数值稳定项）。clean clipped batch mean Z 包含 Linear bias 列，A=mean(ZᵀZ/m)，tau²=(sigma*C/256)²，h=lambda/(lambda+tau²)。不实现 Full Wiener。

`none` 固定 s=1。`rms_match` 使用同一 synthetic covariance 的充分统计量：Eraw=m*sum(lambda)，Efilt=m*sum(lambda*h²)，s=sqrt(Eraw/Efilt)。这与逐 synthetic batch 计算 Frobenius energy 的公式等价；不用存储所有 batch matrices，不改变 eigenvectors 或相对 spectral shape。每层每 epoch 重建，private epoch 内固定。scale 无 cap、无额外超参数，零能量导致未定义 scale 时直接 finite assertion 失败。effective gain=s*h 允许大于 1。private 路径严格 clip → aggregate → DP noise → Wiener-A → AdamW。clean shadow 只供 cosine/NSR diagnostics，不参与 operator、scale、optimizer 或自适应。

保留 Exp33 每 epoch diagnostics，增加 scale/effective gain 分位数和 layer-group scale 均值。五个 update groups 每个 logical step 在真实 `optimizer.step()` 前后测量：组内所有参数差值的 L2 norm（含 bias），除以该组 step 前参数 L2 norm+1e-30 为 relative norm，然后跨 step 取 epoch mean。包含 AdamW moment、epsilon、weight decay 的实际影响。

所有写入限定 `exp33b/`；缓存、TMP、Torch、HF、Triton、CUDA、matplotlib 均定位 `.cache/`，禁用 Python bytecode。共享成熟实现仅只读引用；不导入会创建其他实验缓存的 Exp33 包。

每 run 输出 `results/runs/<run_name>/config.json` 和 `metrics.csv`。全部完成后 `analyze.py` 校验八个 run 均有五个 epoch 及完整 step counters，仅取 epoch 5 生成八行 `summary.csv`、四行 `paired.csv` 和要求的十张图；LR 横轴均为 log scale。scale/effective gain 图展示 epoch 5 的 median、p10–p90、p99（scale 按层，gain 按谱元素），不伪造原始分布。固定参考准确率 **0.9276 来源于 Exp33 DP-AdamW，使用相同 seed/privacy/training protocol**；accuracy 图显示水平参考线，paired CSV 包含参考及差值。不重新训练 DP-AdamW。

文件：`config.py` 固定协议与 grid；`wiener.py` synthetic estimator、scale、post-noise step 和 optimizer diagnostics；`run.py` 单 run；`worker.py` GPU 调度；`run_all.sh` 正式入口；`analyze.py` 完整结果分析；`checks.py` 仅 CPU 轻量检查；`__init__.py` 缓存隔离。`logs/`、`.cache/`、`results/runs/` 为运行输出目录。

轻量检查命令：`conda run --no-capture-output -n curve python -B exp33b/checks.py`。检查不下载 pretrained weights、不训练 CIFAR-10；pretrained 初始化通过验证复用同一函数及调用配置确认，模型结构/转换使用随机初始化的 ViT 做 CPU 检查。toy synthetic oracle、真实 toy AdamW steps 和临时分析 fixtures 仅用于检查，后者放于 `.cache/`，不写正式结果。

已在 `curve` 环境通过全部 lightweight checks（记录于 `logs/checks.log`）：八个 grid 点及 GPU 分配、synthetic clipping/covariance 独立 oracle、RMS 能量匹配与 uncapped scale、epoch 重建、两种模式四个 LR 的真实 AdamW 两步对照（五个组均有非零更新）、74 Linear conversion、RNG/cache 隔离、分析输出及不完整结果拒绝。RDP 校准 sigma=0.72021484375，975 steps 时 epsilon=2.9959030525484205。`bash -n exp33b/run_all.sh` 通过。未启动正式训练，`results/runs/` 未生成正式 run。
