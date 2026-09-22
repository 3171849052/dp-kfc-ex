# Exp36c: bounded Oracle A vs Pink A

验证 bounded shrinkage 后，更准确的 private Oracle A 是否提高 ViT utility。
只运行 `oracle_dp_kfc_a_alpha025` 和 `oracle_dp_kfc_a_alpha05`，seed=42。
Oracle 使用 private geometry：`privacy_status=oracle_non_private_geometry`，
`nominal_dp_epsilon=3` 仅描述 Gaussian training accounting，不是端到端 epsilon=3 DP。

从仓库根目录启动全部正式实验的唯一命令：

```bash
conda run --no-capture-output -n curve bash exp36c/run_all.sh
```

脚本在 GPU 0 顺序执行两个 run，完成后分析；不自动 resume。
依赖已有根目录 `data/` CIFAR-10、Exp30 的 pretrained checkpoint cache，
以及 Exp36b 的 `results/oracle_indices.csv`。数据使用 `download=False`，
预训练 checkpoint 复用 Exp35 的本地复制逻辑；新增 cache、日志和结果均在本目录。
正式 run 目录在运行时创建；轻量检查不会生成 config/metrics 或伪正式结果。

## 固定协议

- `vit_tiny_patch16_224.augreg_in21k_ft_in1k`，pretrained，全参数训练。
- AdamW：lr=1e-4，weight_decay=.01，betas=(.9,.999)，eps=1e-8。
- 5 epochs，seed=42，logical batch=256，physical batch=128，C=1，delta=1e-5。
- preprocessing 与 Exp35/36b 相同：224×224 bicubic resize、mean/std=.5，无 augmentation。
- 直接读取 Exp36b 的同一 CSV，每 epoch 固定 2560 个 train samples，10×256 calibration。
- Exp22 `flatten_linear_input`：token 展平 `(B*T,d)`，有 bias 时添加常数 1；
  按展平后的样本/token 总数计算 `A=X.T@X/N`。calibration 无反向传播或 optimizer 更新。
- `P_raw=(A+1e-3 I)^(-.4)`，`P_norm=P_raw/||P_raw||_2`，
  `P_alpha=(1-alpha)I+alpha P_norm`。仅 alpha=.25/.5；每层断言
  gain 分别在 [.75,1]/[.5,1]，浮点容差 2e-6；outer scale=1。
- 每 epoch 重建 geometry → private per-example gradients → A-only transform →
  global clipping → aggregate → isotropic Gaussian noise → AdamW。
  BK、RDP accounting、shuffle RNG=42、noise RNG=40042 全部复用原训练循环。

## 复用与输出

`exp35b.runtime.configure` 隔离运行目录；进程内将 `exp36b.config` 绑定到本实验配置，
避免导入旧实验时初始化其输出目录，不修改原文件。
`exp36b.oracle_geometry.calibration/build` 复用固定索引数据流与 A builder；
`exp35.geometry.SpectralAOperator/spectral_operator` 复用谱分解、归一化、诊断和变换，
只注入 alpha mixing。通过 `exp35.vit.reference` 运行原 `exp30.run`，
复用 Exp22 模型、BK clipping 和 Exp35 `MetricsSink`。

每个正式 run 保存 `results/runs/<method>/config.json` 与逐 epoch `metrics.csv`，
包括全部原始训练指标、gain 分位数及每层界限、operator bytes、参数检查，
所有 `layer_norm_*` 和 attention_qkv/attention_out/mlp/patch_head/identity group norms。

`analyze.py` 只读取六组已有结果：Exp36b DP-AdamW、Exp35b Pink .25、Exp36c Oracle .25、
Exp35 Pink .5、Exp36c Oracle .5、Exp36b raw Oracle A-only。要求每组均有完整五个 epoch；
不重新训练或补造缺失结果。输出：

- `results/summary.csv`：六组 final/best accuracy 和 final test loss。
- `results/oracle_vs_pink.csv`：每个 alpha、每个 epoch 的 Oracle−Pink accuracy 与累计 best accuracy 差值。
  主要对照为 epoch=5 的 final accuracy difference；accuracy 以比例计，乘 100 得百分点。
- `results/layer_family_norm_ratios.csv`：attention_qkv、attention_out、mlp_fc1、mlp_fc2、patch_head
  相对同 epoch DP-AdamW 的 transformed norm ratio；复用 Exp36b family 分类，按各层 RMS 平方和开根号聚合。
- `results/figures/`：accuracy、test loss、clip fraction、mean clip factor、norm p90/p99，
  以及全部五类 ViT group norm 对比图。

不实现可选 offline checkpoint diagnostic，以保持正式实验改动简洁。

## 轻量检查

```bash
conda run --no-capture-output -n curve python -B exp36c/checks.py
conda run --no-capture-output -n curve bash -n exp36c/run_all.sh
```

CPU 检查真实 CIFAR 校准数据顺序和预处理、2560 indices、固定两组协议、
独立 A 累积公式验证、token/bias、谱归一化和混合公式、两组 gain 边界、
calibration 参数及梯度不变、历史结果完整性及 layer/group 指标。不会启动完整训练。
