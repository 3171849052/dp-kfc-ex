# Exp34：random-init ViT-Tiny 全参数 DP 训练

目标：与 Exp30 pretrained full fine-tuning 对照，在相同架构上测试从头训练时 DP-KFC / DP-KFC-A 相对 DP-AdamW 的表现。实现不下载或加载 pretrained weights。

## 启动

从仓库根目录执行唯一正式启动命令：

```bash
mkdir -p exp34/logs_clip2 && nohup conda run --no-capture-output -n curve bash exp34/run_all.sh > exp34/logs_clip2/launcher.log 2>&1 & echo $!
```

启动器先检查七个 run 目录均不存在，然后只读并验证 `exp30/data/` 中已有的 CIFAR-10，再启动两个 worker。Exp34 不下载或写入数据集。本次 clip_norm=2 的结果写入 `exp34/results_clip2/`，日志写入 `exp34/logs_clip2/`；GPU 0 顺序执行 damping 0.001/0.01 的 Full KFC、A-only，GPU 1 顺序执行 damping 0.1 的 Full KFC、A-only 后再执行 DP-AdamW。各 GPU 内顺序执行，每个 run 是独立进程。失败返回非零，不自动 resume、跳过或降低 batch size。已有任意正式 run 目录直接报错。原 clip_norm=1 结果保留在 `exp34/results/`。

## 协议及复用

- `torch.manual_seed(42)` 在 `timm.create_model(..., pretrained=False)` 前调用，然后使用 Exp22 `convert_vit`，替换为 192→10 head，全部参数可训练。
- ViT-Tiny/16：192 维、12 blocks、3 heads、CLS pooling；5,526,346 个参数；74 个 Linear。
- CIFAR-10：bicubic Resize(224,224)、ToTensor、mean/std=0.5；无 augmentation。
- 20 epochs，每 epoch 195 logical steps，总计 3900；logical/physical batch=256/128，drop_last=True。
- clip_norm（`MAX_GRAD_NORM`）=2；同时用于逐样本全局裁剪和 Gaussian noise 的 sigma×C。
- AdamW：lr=1e-4、weight_decay=0.01、betas=(0.9,0.999)、eps=1e-8。
- RDP 根据 epsilon=3、delta=1e-5、q=256/50000、3900 steps 重新计算 sigma。
- 直接复用 Exp22 geometry builder 和 BK clipping。Full KFC 为 `(G+λI)^(-1/2) g (A+λI)^(-1/2)`；A-only 沿用 Exp30 的去 scale 处理，为 `g(A+λI)^(-0.4)`，scale=1。
- KFC 覆盖 patch projection、每个 block 的 q/k/v/out 和 fc1/fc2、head。25 个 LayerNorm、cls_token、pos_embed 为 identity。DP-AdamW 全部 identity。
- 每 epoch 重新使用 10×256 个 alpha=1 pink synthetic samples，synthetic physical batch 同 Exp30 为 256。Full KFC 用随机 CIFAR-10 labels 构建 G；A-only 不构建 G。
- RNG：初始化和 private shuffle=seed；synthetic x=seed+10000+epoch；labels=seed+20000+epoch；DP noise=seed+40000。

## 文件与输出

`model.py` 实现随机初始化，`geometry.py` 封装成熟 builder 并读取频谱；`run.py` 保留 Exp30 训练指标，`worker.py`/`run_all.sh` 实现固定调度；`analyze.py` 严格验证七条完整 20-epoch trajectory，再输出 7 行 summary、3 行 paired 和 11 张图。`checks.py` 为 CPU 轻量验证入口。

每个 run 的 `config.json` 在训练前保存；`metrics.csv` 和 `geometry_metrics.csv` 每 epoch 更新。logical/accountant/optimizer/noise steps 均为累计值，`epoch_logical_steps` 为当 epoch 值。accuracy AUC 沿用 Exp22/Exp30：epoch 1 到当前 epoch 的梯形积分。

频谱来自未加 damping 的原始协方差（包含已有 builder 的 bias 增广），使用对称化后的 float64 eigvalsh；诊断不修改 factors 或 operator。relative damping 为 λ/(trace/dim)，零 trace 记录 inf。完整 A/G 分位数逐层保存，A-only 的 G 列为空；每组中位数进入 metrics.csv。DP-AdamW 的 geometry CSV 仅保留表头。builder_seconds 包含频谱诊断开销。

所有 cache、TMP、Torch/HuggingFace/Triton/matplotlib 路径都位于 `exp34/.cache/`；本次日志在 `exp34/logs_clip2/`，正式结果在 `exp34/results_clip2/`。Exp22 config 的 ROOT 仅在当前 Python 进程内重定向以避免其 model 导入写入旧缓存路径，不修改原文件。

轻量检查不下载数据、不运行正式训练：验证随机初始化可复现、conversion 输出、结构和覆盖、3900-step RDP、微型网络 builder/BK 和频谱不改变算子，以及隔离临时 fixture 的完整分析绘图。结果写入 `logs/checks.json`。完整 GPU 吞吐、显存和实验准确率需正式运行后获得。

`temporary_per_sample_grad_bytes` 保留 Exp22 的原始计量语义，包含 Gram tile 和 aggregate 等临时 workspace，非零不代表创建 `[B,Dout,Din]` 梯度张量。
