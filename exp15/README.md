# exp15：合成 K-LBFGS 逆海森幂

研究 `P_p = H^p` 对 MNIST + SimpleCNN DP 训练的影响。代码、配置、测试、日志和结果均在本目录，已有实验不作修改。本次修复未启动正式实验；目录中既有旧协议结果保留，需按新协议重跑后再汇总。

## 运行

在项目根目录：

```bash
conda activate curve
./exp15/run_all.sh
```

脚本顺序执行恰好 10 个正式 runs，然后生成汇总和图。单独重跑：

```bash
python exp15/run_exp15.py --p 0.5 --seed 42
python exp15/run_exp15.py --summarize
```

`--p` 可选 `0.0 0.25 0.5 0.75 1.0`，`--seed` 可选 `42 7`，默认 `42`。同一 p/seed 重跑覆盖其结果。正式汇总会拒绝旧协议结果与本次修复结果混用；完整脚本重跑全部 10 个组合后，每个 p 的 count=2。默认使用可见的第一张 CUDA GPU，可用 `CUDA_VISIBLE_DEVICES` 指定。

## 固定配置与复用

`configs/mnist.yaml` 继承项目 `configs/standalone/mnist_dp_kfc.yaml` 的模型、SGD 和 DP 参数，并覆盖 synthetic 数量及 drop_last 以对齐 exp14，在每个 run 的 `config.json` 写入展开配置及 noise multiplier、sample rate、数据规模和步数。

| 参数 | 正式实验值 |
|---|---|
| p × seed | `{0, .25, .5, .75, 1}` × `{42, 7}` |
| 模型/数据 | 原 `SimpleCNN`、MNIST、原归一化 |
| SGD | lr=0.5，momentum=0，weight decay=0 |
| DP | epsilon=1，delta=1e-5，C=1，RDP |
| train/eval batch / epochs | 256 / 256 / 5 |
| private drop_last / steps | True / 1170（每 epoch 234） |
| 合成图像 | 原 `generate_pink_noise`，alpha=1，标签独立均匀采样 0–9 |
| 合成样本 / batch | 2560 / 256，即每 epoch 10 个 batches |
| 曲率刷新 | 每 epoch 开始，memory 跨 synthetic batches 和 epochs 保留 |
| L-BFGS memory / 初始尺度 | 100 / 1（sibling 默认；5 epochs 共尝试 50 对/因子） |
| damping | sqrt(0.001)，沿 sibling 将既有 damping 分配到两因子的约定 |
| A EMA / secant EMA | 0.9 / 0.9（sibling 默认） |
| 合成虚拟 SGD 步长 | 0.5（沿用基础学习率；与 p 无关） |

直接 import 现有 `load_data`、`SimpleCNN`、`pink_batches`、`KFACRecorder`、`build_optimizer`、`evaluate`、norm/clipping/noise utilities。训练入口只补充曲率刷新和所需 metrics，未复制已有训练框架。现有 standalone 使用 Opacus `GradSampleModule(loss_reduction='sum')`，无需 Ghost Clipping：直接变换其显式 `grad_sample`，再交给原隐私工具。

采用与 exp14 对齐的 fixed-batch shuffled accounting convention：私有 DataLoader `shuffle=True, drop_last=True`，每 epoch 丢弃不足 256 的尾 batch；noise multiplier 统一按 `q=256/60000`、`steps=5*floor(60000/256)=1170`、epsilon=1、delta=1e-5 和 RDP 计算。这不是 Poisson sampling。config snapshot 与 run summary 均记录该 convention、sample rate 和 total steps。`epsilon_spent` 是沿用项目该约定的 accountant 数值；它不是对 shuffled sampling 另行建立的隐私证明。未引入新的 sampling/accountant 实验变量。

exp15 保留显式 `GradSampleModule`；accuracy / clipping geometry 可以与 exp14 做方法层面对照，但 runtime / memory 不应直接与 exp14 的 Structured Ghost 实现做公平性能比较。

## 曲率和 H 的正指数

这里参考/复用 K-BFGS 的 factorized L-BFGS compact update、acceptance 和 damping，并非完整复现原 K-BFGS optimizer：`H_g` secant 来自 plain synthetic SGD lookahead，不是完整 K-BFGS search direction。

`preconditioner.py` 直接加载项目同级 `../kbfgs_neurips2020_public/kbfgs_utils.py` 中的：

- `LBFGS_Hv`：原 compact inverse-Hessian product；
- `Kron_LBFGS_append_s_y`：原 acceptance、memory 和 compact 更新；
- `get_BFGS_PowellHDamping`、`get_BFGS_ModifiedDamping`：原先后顺序和公式，Powell 阈值 0.2。

每次 refresh 开始克隆当前 private model 的参数状态 `theta_t`，建立 ordinary SimpleCNN。每个 synthetic batch 的 before-forward 前都重新载入完全相同的 `theta_t`；只在副本上做一次 virtual SGD lookahead，然后构造该 batch 的 pair。下一 batch 再次恢复 `theta_t`，不会继承前一 batch 的 lookahead 参数。只有 L-BFGS memory、activation covariance EMA、secant-pair EMA 和 diagnostics counters 跨 batches/epochs 持久化。不复制私有梯度或 Opacus hooks。合成输入/标签在独立 RNG fork 中生成，曲率刷新不消耗私有训练 RNG。合成数据完全不进入私有训练 loss。

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

正式汇总仅纳入当前配置的 seeds 42、7，保留但不混入已有 seed 0、1 结果。后续 `--smoke` 使用 `--seed` 指定的 seed（默认 42）；下述 seed=0 验证记录为历史结果。

正式结果位于 `results/formal/p{p}_seed{seed}/`；smoke 独立位于 `results/smoke/`。

- `config.json`：展开配置和实际噪声/采样参数。
- `metrics.csv`：每 logical step 的 `train_loss`、`preclip_norm_mean/p50/p90/p99`、`clip_fraction`、实际参数差的 `update_norm`、step/epoch，以及累计 `accepted/rejected/powell_damping/modified_damping`。
- `training.csv`：每 epoch 的 train loss、test loss、test accuracy、epsilon spent。
- `train.log`：训练进度；smoke 另含核心数值验证。
- `summary.json`：最终准确率、epsilon、步数、accounting convention、drop_last、sample rate、noise multiplier、synthetic batches 数和每层每因子累计 pair/damping 数量。
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

修复后的验证结果见下（日志 `results/unit_tests.log`、`results/smoke.log`）。测试覆盖：

1. p=0 identity；p=1 对照 sibling `LBFGS_Hv`；.25/.5/.75 对照小型 dense SPD reference，输出 finite。
2. 相同 batch 和 Gaussian RNG 下，p=0 私有更新与原 standalone DP-SGD 一致。
3. Kronecker 两侧变换、global norm、clipping、指定 Gaussian noise 和 SGD 更新与独立计算一致。
4. 正式配置固定 1170 steps、10×256 synthetic；创建 smoke 配置不会污染正式配置。
5. 多 synthetic batches 实际 before-forward 参数逐元素等于 refresh 开始的 `theta_t`；确认 lookahead 确实修改副本，随后 batch 已恢复；前一批接受的 pair 内容及计数保留。刷新后再独立计算整个 CNN 的 precondition→global clip→noise→update，对照 private step。

此前已完成的 tiny smoke 覆盖 p=0、0.5、1，seed=0；每分支使用 8 个 MNIST 训练样本、32 个测试样本、private batch=4、2 epochs，synthetic 为每 epoch **3×8**。共享 `verification.py` 在 smoke 中实际断言 before-forward reset、private model 不变、pair 保留及代表性 factor memory >1；每次刷新还检查 H 的幂。正式路径仍使用原 SyntheticKLBFGS，不加载这些测试断言。

2026-09-14 修复后在 `curve` 环境按上述顺序执行：**pytest 5 passed**；真实 MNIST+CNN smoke 的三个 p 分支全部通过。每个分支完成 4 个 private logical steps、2 次 refresh；所有因子在第一次 refresh 后有 3 对、第二次后有 6 对。每分支累计 accepted=48、rejected=0、Powell damping=3、modified damping=16；before-forward reset 和 pair retention 断言全部通过。p=1 相对 sibling Hv 最大误差 `4.84e-15`，fractional 输出 finite。

每分支 smoke 最终 epsilon=`0.9963923627`、test accuracy=`0.09375`；config、metrics、training、summary、日志和两张图已重新生成。极小 batch 下噪声较大，smoke 准确率只用于验证流程，不表示 hypothesis 的性能信号。

另外只计算了正式协议的 noise multiplier（未训练）：`q=256/60000`、1170 steps 给出 sigma=`1.068115234375`；配置确认 synthetic 为 10 batches/epoch。`run_all.sh` 保持原 5 powers × 2 seeds，最后 summarize，shell 语法检查通过。本次未启动该脚本，也未改写已有旧协议 formal 结果。PyTorch backward-hook 和 RDP order 提示保留于日志。
