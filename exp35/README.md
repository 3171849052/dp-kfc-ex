# Exp35：A-only 谱归一化

从仓库根目录运行全部正式实验的唯一入口：

```bash
conda run --no-capture-output -n curve bash exp35/run_all.sh
```

固定使用物理 **GPU 2**（进程内 `cuda:0`），8 个 run 按 MNIST、ViT 顺序串行运行，每个 run 使用独立进程。不会自动 resume，也不会调整 batch。已有 run 目录会使启动失败，避免覆盖结果。实现和轻量检查不启动正式训练。

## 固定协议

| 项目 | MNIST | CIFAR-10 |
|---|---|---|
| 模型 | SimpleCNN | pretrained vit_tiny_patch16_224.augreg_in21k_ft_in1k，全参数微调 |
| 基线 | dp_adam | dp_adamw |
| 优化器 | Adam，lr=0.002，无 weight decay | AdamW，lr=1e-4，weight decay=0.01 |
| betas / optimizer eps | (0.9,0.999) / 1e-8 | (0.9,0.999) / 1e-8 |
| epochs / seed / C / delta | 5 / 42 / 1 / 1e-5 | 5 / 42 / 1 / 1e-5 |
| epsilon | 2 | 3 |
| logical / physical batch | 256 / 256 | 256 / 128 |
| 每 epoch pink geometry | 1 × 256 | 10 × 256 |
| 预处理 | ToTensor，Normalize(0.1307,0.3081) | bicubic Resize(224,224)，ToTensor，Normalize(0.5,0.5,0.5)，无增强 |

两个 grid 各自严格包含基线和 `dp_kfc_a_raw`、`dp_kfc_a_norm`、`dp_kfc_a_mix05`。只用 BK，不运行 Full KFC 或 Explicit engine。

每层对增广协方差 A 做特征分解，`g=(λ+0.001)^(-0.4)`；raw 使用 g，norm 使用 `g/max(g)`，mix05 使用 `0.5+0.5*g/max(g)`。保留同一组特征向量，不做 scale matching，不做 SVD。norm 每层最大 gain 为 1；mix05 每层 gain 在 [0.5,1] 内，运行时断言容差 2e-6。汇总 gain 分位数按所有被预条件化层的特征增益等权统计；基线 gain 为 1、operator state bytes 为 0。ViT 的 identity 参数继续使用 Exp22 原处理方式。

MNIST 原 `compute_a_covariance` 隐含 `1e-5 I`，本实验局部替换为未加正则的同一 bias-augmented 协方差，从而只有指定的 `1e-3` damping。

## 复用边界

- `scripts/paper/exp_cnn_mnist_a.py`：完整 `run_one`、pink geometry 采集、SimpleCNN、BK 微分、范数和聚合、噪声及 RDP accounting。数据加载按 Exp27 设置，强制根目录 `data/` 与 `download=False`。
- `exp30/run.py`：直接加载并执行原训练循环，保留 shuffle、synthetic、DP noise RNG 和全部层/组诊断。使用进程内 `exp30.config` 别名，避免原包初始化器将 cache 指向 Exp30。
- `exp22/geometry.py`：原 A-only covariance builder / synthetic stream；仅注入新的算子构造器。
- `exp22/model.py`、`exp22/methods.py`：原 pretrained 模型转换、全参数 BK clipping 和 AdamW 更新。
- `adapters.py`：函数局部依赖注入及逐 epoch CSV 输出适配，没有复制训练循环、修改共享模块函数或已有文件。

ViT 初始化、shuffle、synthetic x 和噪声种子分别沿用 42、42、`42+10000+epoch`、40042；A-only 不消耗 synthetic label RNG。MNIST 保留原始 noise seed 20042。两者均沿用固定大小、shuffle、drop-last 的 RDP accounting 约定。

数据只读取根目录 `data/`，不下载。ViT 正式启动时将仓库已有 `exp30/.cache/huggingface/hub/models--timm--vit_tiny_patch16_224.augreg_in21k_ft_in1k` checkpoint 复制到 Exp35 cache，并以 HF offline 模式加载。所有新 cache、临时文件、日志与结果都位于 `exp35/`。

## 输出与检查

每个 `results/{mnist,vit}/runs/<method>/` 在训练前写 `config.json`，每 epoch 写 `metrics.csv`。保存 loss、accuracy/best accuracy、epsilon、noise multiplier、clipping、transformed norm、gain 分位数与逐层边界、operator bytes、参数 finite/updated 检查；ViT 保留全部 `group_norm_*`、`layer_norm_*`。主 step 字段均为累计值，`epoch_*` 字段是当 epoch 值；noise_steps 与 noise_events 等价。MNIST clip fraction 沿用原代码 `(factor<1)`。

`analyze.py` 先验证 8 个 run 都存在 config 和连续 1–5 epoch，再生成两个子实验及总 summary，以及各自 accuracy、clip fraction、mean clip factor、norm p90/p99 曲线；ViT 散点图用颜色标出四种方法、数字标出 epoch。没有提前生成正式结果。

轻量检查：

```bash
conda run --no-capture-output -n curve python -B exp35/checks.py
```

检查 Python import/compile、shell syntax、算子公式/特征向量/谱边界、实际 Exp22 builder 和 BK 范数、4+4 grid、GPU、真实数据加载路径及禁止下载、逐 epoch 输出计数和分析完整性门禁。仅 CPU toy 运算和读取本地数据，不初始化 pretrained ViT、不执行正式训练。检查日志为 `exp35/logs/checks.log`。
