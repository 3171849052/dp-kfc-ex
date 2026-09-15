# Exp19: Exact/full KFC → Ghost → forward-only A → weak power

本实验所有代码、缓存、日志和结果只写入 `exp19/`。只读复用现有 MNIST、模型、KFAC 数学原语与 Structured Ghost；不修改旧实验。

## 四方法及 attribution

| 方法 | 左侧 C | 右侧 A | global scale | clipping |
|---|---|---|---|---|
| M0_original | full `(C+1e-3 I)^(-1/2)` | full `(A+1e-3 I)^(-1/2)` | 1 | Exact / Opacus GradSampleModule |
| M1_full_ghost | 同 M0 | 同 M0 | 1 | Structured Ghost |
| M2_a_ghost_p05 | identity，无 damping | `(A+1e-3 I)^(-1/2)` | 1 | Structured Ghost |
| M3_a_ghost_p025 | identity，无 damping | `(A+1e-3 I)^(-1/4)` | `sqrt(m_0.5/m_0.25)` | Structured Ghost |

M1−M0 隔离 Ghost backend；M2−M1 移除 C（包括其带来的尺度变化）；M3−M2 为 weak power 加上指定 RMS matching 的联合贡献；M3−M0 是 headline。

M0/M1 直接使用 `exp13.operator.Operator`，保留已有原版实验的 augmented bias、double 精度对称 eigendecomposition、damping=1e-3 数学约定。builder 使用 `exp12.curvature` 的 forward、activation_sum、covariance，与 `estimate(..., 'KFAC-U')` 做 regression。这里沿用原版实验的 KFAC-U uniform pseudo-label 约定，而非另行改用其他 estimator 或额外 covariance ridge。

M2/M3 没有 C tensor，不计算输出概率/伪标签，不调用 curvature backward/VJP。A-only moment 为 `m_beta = sum_layers output_dimension * sum_j lambda_j*(lambda_j+1e-3)^(-2 beta)`，对应 Exp18 identity-side moment=dimension。只用本 operator 的 A 特征值，验证 `s_A²*m_0.25=m_0.5`；无 private calibration 或 full Fisher oracle。A 的 PSD 数值舍入沿用 Exp18 clamp_min(0)。

## 固定协议和随机数

MNIST / `dp_kfac.models.SimpleCNN`，5 epochs，batch=256，drop_last，234 steps/epoch，1170 steps。SGD lr=.5、momentum=0、weight_decay=0，epsilon=1、delta=1e-5、clip=1、damping=1e-3。每 epoch 重建，10×256 pink-noise；五 paired seeds：42、7、123、2024、3407。

每个 method/seed 是独立 fresh Python subprocess，顺序运行。相同 seed 初始化相同；private DataLoader generator=seed，DP generator=seed+40000，synthetic=seed+10000+epoch（fork_rng 保护全局状态），full pseudo-label generator=seed+20000+epoch。epoch 从 1 开始。noise 对固定参数顺序生成 flattened Gaussian，依次加到 clipped aggregate、除 batch size、SGD。所有方法使用同一 draw shape/schedule。

sigma 按相同固定参数通过 Opacus RDP calibration 计算；每个私有 step 更新 accountant。沿用仓库 shuffled fixed-batch RDP accounting convention；这不等于对该 shuffle sampler 另行证明的精确会计。smoke sigma=0、accountant_steps=0。

## Profiling 与诊断

CUDA 主计时和内部阶段边界均 synchronize。build/train 各自 reset peak 并记录 allocated/reserved 起点和峰值；incremental=peak_allocated−allocated_at_start。total peak 是 build/train 两者最大值，evaluation 排除。reserved 包含当前 subprocess 内 allocator cache；不在 batch 内 empty_cache/gc.collect。

`exp12.runtime` 只在正式模型创建前做 disposable 2×2 CUDA/cuBLAS warmup；没有 optimizer.step、accountant、metrics 或正式 RNG draw。未做模型级 warmup，因此首 epoch 可能包含额外 kernel 初始化成本，所有方法采用同一规则。

Builder 拆分 synthetic generation、activation forward、curvature backward（full 包括 softmax/伪标签/VJP）、factor accumulation、matrix function。总 build 还包含必要的对象构造/释放开销。A-only backward time 严格为 0。

Private time 包括 hook 安装/移除、实际训练、相同口径 CPU diagnostics transfer；层 norm 不额外重跑网络。Exact backward 后、clipping 前计量真实 grad_sample payload；Ghost 不创建 grad_sample。内部 phase 只作解释，不用来直接跨 backend 排名。`algorithm_epoch_seconds=build+private_train`；evaluation 独立，wall 为实际 epoch wall time（包括汇总，不包括 CSV 写盘）。

在下次 build 前释放旧 operator 和 Ghost hooks；builder 临时 factors 不返回、不跨 epoch 保存。operator_state_bytes/stored_scalar_count 只数 action tensor payload 和 A-only float64 scale，不计 Python metadata 和仅供报告的 moment 标量。

clipping 使用 transformed noise-before per-example global norm，`c=min(1,1/(norm+1e-6))`；clip_fraction 为 `c<1` 的比例，severity=mean(1−c)。layer ratio 为每层 augmented weight+bias norm² / global norm²；零梯度样本 ratio 定义为 0。汇总 mean、median、p90、p99。

**所有 private clipping/norm、layer_norm_diagnostics.csv 及未加噪 train_loss 均为 research-only，不是 DP release。不要将包含这些诊断的结果包当作隐私保护发布。**

## 运行

测试：

```bash
PYTHONDONTWRITEBYTECODE=1 \
XDG_CACHE_HOME="$PWD/exp19/.cache" \
MPLCONFIGDIR="$PWD/exp19/.cache/matplotlib" \
CUDA_CACHE_PATH="$PWD/exp19/.cache/cuda" \
conda run -n curve python -m pytest exp19/test_exp19.py -q -o cache_dir=exp19/.pytest_cache
```

Smoke：`conda run -n curve bash exp19/run_all.sh --smoke`。seed=42，四个 fresh subprocess，每个 1 epoch、1 private batch（256 MNIST）、1 synthetic batch（256）、sigma=0。输出 `results/smoke/`，日志 `smoke.log`；不覆盖正式结果。

正式唯一启动命令：

```bash
conda run -n curve bash exp19/run_all.sh
```

正式完整实验不在实现/验证过程中自动启动。缺失数据、CUDA 或依赖时直接失败。

## 结果与分析

每个 subprocess 增量写 `results[/smoke]/runs/<method>_<seed>/metrics.csv`、`layer_norm_diagnostics.csv` 和 `config.json`。全组成功后 analyze 合并为 metrics、summary（每 seed）、method_summary（mean/sample_std）、paired_summary（四条 attribution 对比）、timing_breakdown、memory_summary、builder_budget、layer_norm_diagnostics CSV，以及 config.json、report.md 和要求的十张 PNG。

正式结果仅在实际正式运行后生成，不用 smoke 数据填充正式结果。smoke 全套分析位于 `results/smoke/`，不计算只有一个 seed 的 CI。

final_accuracy 为 epoch 5，best_accuracy 为五次测试最大值，accuracy_auc 为 observed epochs 1..5 的 trapezoid integral（smoke 单点为 0）。paired delta 按 seed 对齐，20,000 paired bootstrap resamples、percentile 95% CI，sample std ddof=1。只有五 seeds，报告数据和区间，不声称强显著性，也不把未显著下降当成已证明保持精度。图中的 epoch curves 为跨 seed 均值，scatter 为单 seed；CSV 保留原值和 sample std。
