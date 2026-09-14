# exp15：合成 K-LBFGS 逆海森幂

研究 `P_p = H^p` 对 MNIST + SimpleCNN DP 训练的影响。代码、配置、测试、日志和结果均在本目录，已有实验不作修改。正式实验尚未启动。

## 运行

在项目根目录：

```bash
conda activate curve
./exp15/run_all.sh
```

脚本顺序执行恰好 10 个正式 runs，然后生成汇总和图。单独重跑：

```bash
python exp15/run_exp15.py --p 0.5 --seed 1
python exp15/run_exp15.py --summarize
```

`--p` 可选 `0.0 0.25 0.5 0.75 1.0`，`--seed` 可选 `0 1`。同一 p/seed 重跑覆盖其结果。默认使用可见的第一张 CUDA GPU，可用 `CUDA_VISIBLE_DEVICES` 指定。

## 固定配置与复用

`configs/mnist.yaml` 继承项目 `configs/standalone/mnist_dp_kfc.yaml`，在每个 run 的 `config.json` 写入展开配置及 noise multiplier、sample rate、数据规模和步数。

| 参数 | 正式实验值 |
|---|---|
| p × seed | `{0, .25, .5, .75, 1}` × `{0, 1}` |
| 模型/数据 | 原 `SimpleCNN`、MNIST、原归一化 |
| SGD | lr=0.5，momentum=0，weight decay=0 |
| DP | epsilon=1，delta=1e-5，C=1，RDP |
| batch / epochs | 256 / 5 |
| 合成图像 | 原 `generate_pink_noise`，alpha=1，标签独立均匀采样 0–9 |
| 合成样本 / batch | 256 / 256 |
| 曲率刷新 | 每 epoch 开始，memory 跨 epoch 保留 |
| L-BFGS memory / 初始尺度 | 100 / 1（sibling 默认；5 epochs 最多累积 5 对/因子） |
| damping | sqrt(0.001)，沿 sibling 将既有 damping 分配到两因子的约定 |
| A EMA / secant EMA | 0.9 / 0.9（sibling 默认） |
| 合成虚拟 SGD 步长 | 0.5（沿用基础学习率；与 p 无关） |

直接 import 现有 `load_data`、`SimpleCNN`、`pink_batches`、`KFACRecorder`、`build_optimizer`、`evaluate`、norm/clipping/noise utilities。训练入口只补充曲率刷新和所需 metrics，未复制已有训练框架。现有 standalone 使用 Opacus `GradSampleModule(loss_reduction='sum')`，无需 Ghost Clipping：直接变换其显式 `grad_sample`，再交给原隐私工具。

采用原 standalone 的 shuffled minibatch、`q=256/60000`、`5*ceil(60000/256)` 个 RDP accountant steps，包括最后不足 batch 的原处理方式。`epsilon_spent` 是沿用项目该约定的 accountant 数值；它不是对 shuffled sampling 另行建立的隐私证明。未引入新的 sampling/accountant 实验变量。

## 曲率和 H 的正指数

`preconditioner.py` 直接加载项目同级 `../kbfgs_neurips2020_public/kbfgs_utils.py` 中的：

- `LBFGS_Hv`：原 compact inverse-Hessian product；
- `Kron_LBFGS_append_s_y`：原 acceptance、memory 和 compact 更新；
- `get_BFGS_PowellHDamping`、`get_BFGS_ModifiedDamping`：原先后顺序和公式，Powell 阈值 0.2。

每次刷新建立普通 CNN 副本，只载入当前模型参数，不复制私有梯度或 Opacus hooks。合成输入/标签在独立 RNG fork 中生成，曲率刷新不消耗私有训练 RNG。合成数据完全不进入私有训练 loss。

输入侧 `H_a` 使用 sibling 的 Hessian-action pair：输入含 bias 常数列，维护 `A=E[aaᵀ]` 的 EMA，`s=H_a mean(a)`，`y=(A+dI)s`。卷积输入按原项目的 unfold/空间位置约定处理。

输出侧 `H_g` 使用 sibling 的 preactivation/backprop secant：在同一合成 batch 上，对副本做固定步长的虚拟 SGD，取前后 preactivation 均值之差为 s、对应 sum-loss backprop 均值之差为 y；卷积按样本和空间位置求均值。按原 0.9 EMA 平滑，再依次调用两种 damping 和原 pair acceptance（使用合成 backprop 均值作为 g_k）。适配普通 CNN 的虚拟步固定为 SGD；不调用 sibling 依赖其自定义模型结构的整个 optimizer，也不以当前 p 改变合成虚拟步。副本随刷新结束丢弃，私有模型参数未被这个虚拟步更新。

每层将 weight/bias 合并为矩阵，应用

```
H_l ≈ H_g,l ⊗ H_a,l
G_tilde_i,l = H_g,l^p @ G_i,l @ H_a,l^p
```

随后严格执行：

```
所有层的 H^p g_i → 全模型 L2 norm → clipping → 求和 → Gaussian noise → 除 batch size → SGD
```

`p=0` 分支直接返回原梯度；即使做了合成诊断，也不改变私有更新或 RNG。不是加噪后乘 H，且没有使用 `H^{-p}`。

### 小矩阵谱分解

原 compact 表达为 `H=gamma I + L R`。令 Q 为 L 的 reduced QR 基，计算至多 `min(n,2m)` 阶的 `B=QᵀHQ`，对 B 的数值对称部分做特征分解 `B=U diag(lambda) Uᵀ`。于是

```
H^p v = gamma^p v + (QU) diag(lambda^p-gamma^p) (QU)^T v
```

实现只保存薄基，不显式构造完整 CNN Hessian，也不构造层的稠密 H。谱分解和 curvature pairs 用 float64，私有梯度乘法沿用模型 float32。检查谱为正且有限；不 clamp 特征值，不额外修复/回退。空 memory 遵循 sibling 的 identity 行为。

## 输出

正式结果位于 `results/formal/p{p}_seed{seed}/`；smoke 独立位于 `results/smoke/`。

- `config.json`：展开配置和实际噪声/采样参数。
- `metrics.csv`：每 logical step 的 `train_loss`、`preclip_norm_mean/p50/p90/p99`、`clip_fraction`、实际参数差的 `update_norm`、step/epoch，以及累计 `accepted/rejected/powell_damping/modified_damping`。
- `training.csv`：每 epoch 的 train loss、test loss、test accuracy、epsilon spent。
- `train.log`：训练进度；smoke 另含核心数值验证。
- `summary.json`：最终准确率、epsilon、步数、每层每因子累计 pair/damping 数量。
- 汇总目录 `summary.csv`：每个 p 的 seed 数、准确率 mean 和 sample std（ddof=1）；正式完成时每组 count=2。
- `test_accuracy_vs_h_power.png`：mean ± std。
- `clip_fraction_vs_step.png`：每个 p/seed 的 clipping fraction。

这些训练 loss、未加噪 norm 分位数和 clipping fraction 是私有数据上的研究诊断，本身不由训练 accountant 保护；`epsilon_spent` 只描述训练机制，不包含公开这些原始诊断的隐私成本。

## 已完成验证

使用 `conda activate curve` 后，先执行：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest exp15/tests -q -o cache_dir=exp15/.pytest_cache
python exp15/run_exp15.py --smoke
```

2026-09-14：3 个单元测试通过（日志 `results/unit_tests.log`）：

1. p=0 identity；p=1 对照原 `LBFGS_Hv`；.25/.5/.75 对照小型稠密 SPD 谱分解。
2. 相同 batch 和 Gaussian RNG 下，p=0 私有更新与原 standalone DP-SGD 一致。
3. Kronecker 两侧变换、全模型 preconditioned norm、clipping、指定 Gaussian noise、SGD 更新与独立计算一致。

真实 MNIST+CNN tiny smoke 全部完成：一套 smoke 覆盖 p=0、0.5、1，seed=0；每个分支使用 8 个 MNIST 训练样本、32 个测试样本、batch=4、2 epochs，合成 batch=8。它仅缩小运行规模，保留完整曲率/训练路径。每个分支完成 4 个 private logical steps、2 次曲率刷新，接受 16 对、拒绝 0 对，Powell damping 1 次、modified damping 5 次。三个分支每次刷新均验证 p=0、p=1 和 fractional finite；最大的 p=1 相对误差约 `1.95e-15`。

每个分支最终 accountant epsilon 为 `0.9963923627`，test accuracy 为 `0.09375`，所有 metrics、配置、summary、两张图均已写出。第一步 norm mean 从 p=0 的 4.57 变为 p=1 的 23.76，确认非 identity 变换实际生效。极小 batch 下 DP 噪声造成很大的更新，smoke 的准确率仅用于验证写出流程，不表示性能或正向 signal；正式 10-run sweep 才回答 hypothesis。

原 PyTorch backward hook 和 RDP order 提示保留在 smoke 日志中。未运行正式 runs，未新增其他 smoke 变体或 hardening 基础设施。
