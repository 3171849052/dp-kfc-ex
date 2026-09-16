# Exp23: MNIST public proxy mismatch

固定私有任务 `dp_kfac.models.SimpleCNN` + MNIST。公共定义来自
`docs/static/dp_kfc.pdf` Fig. 2、Appendix G：FashionMNIST matched，CIFAR-10
mismatched。CIFAR-10 的 Resize(28,28) → Grayscale(1) → ToTensor →
Normalize((0.1307,), (0.3081,)) 仅适配输入接口，不改变 domain-mismatched 定义；
转换复用 `scripts/paper/ablation_fim_spectrum.py` 的顺序。

七个方法见 `config.py`。前四项是核心比较；pink 与 DP-SGD 是参照。
A-only 复用 Exp20 AOperator 的谱运算和全层 output-dimension-weighted RMS matching：
`m_p = Σ_l d_out Σ_i λ_i(λ_i+0.001)^(-2p)`，
`s_A = sqrt(m_0.5 / m_0.4)`。固定 p=0.4，左侧为完全无 damping 的 identity。
A-only builder 只 forward，不读取 label、不建立 G、不调用 backward/VJP。
Bias 使用 augmented activation。完整 KFC 复用 Exp12 的带标签 estimator 和
Exp13 的 `(G+0.001I)^(-1/2) g (A+0.001I)^(-1/2)`；内部 C 字段就是 G。
真实公共标签用于 full public，pink 使用独立 RNG 的均匀随机标签。
G 是 sum-cross-entropy 的逐样本 activation gradient covariance；卷积按空间位置平均。

所有方法使用 Exp19 Structured Ghost clipping、aggregate transform、Gaussian noise
和 SGD 更新。DP-SGD 使用 identity operator。沿用 Exp19/20：5 epochs、256 logical
batch、drop_last、SGD lr=0.5、无 momentum/weight decay、C=1、epsilon=1、delta=1e-5。
RDP 使用仓库 shuffled fixed-batch accounting convention；没有改变或增强其保证。
每 epoch 重建，所有六个 preconditioned 方法都恰好用 10×256 calibration samples；
DP-SGD 没有 preconditioner、无需 calibration。

Seeds 42/7/123 配对。CPU 初始化 seed，shuffle 独立 generator seed，DP noise 独立
CUDA generator seed+40000；calibration 使用隔离 RNG。公共样本每 epoch 用
seed+10000+epoch 无放回抽取，A/full 对应 source 完全一致。每个 run 是 fresh subprocess。
CUDA phase timing 沿用 Exp20 profiling，包含冷启动成本，没有 warmup；oracle/evaluation
不计入 builder/private train time。记录 builder forward/VJP/reverse-vector 计数；
backward_calls 指 `.backward()` 调用，因此 full 的 reverse 工作记为 VJP。
Peak memory 是 builder 与 private training 阶段 allocated 峰值的最大值，含常驻 oracle cache。

## 数据与环境

使用 conda `curve`，必须有 CUDA 与现有训练依赖。数据全部 `download=False`：

- MNIST：`exp1/data/MNIST`（只读）
- FashionMNIST：`exp23/data/FashionMNIST`（本次从本机已存在的数据复制）
- CIFAR-10：`exp22/data/cifar-10-batches-py`（只读）

路径在 `config.py` 明确指定。没有数据搜索、自动下载、fallback、batch 调整或失败跳过。
所有新增文件、缓存、日志、测试和结果仅位于 exp23。首次阅读 PDF 使用的 pypdf
仅安装于 `exp23/.cache/pdf`，实验不依赖它。

## Research-only diagnostics

每 epoch 在更新前，用固定 seed=23000 的 10×256 MNIST oracle 样本计算 raw factors。
所有方法固定同一组 oracle indices，模型状态随训练变化；A-only oracle 同样仅 forward。
记录每层 cosA / A relative Frobenius error；full 额外 cosG / G error / cosA×cosG。
Oracle 不进入训练、不改变权重、不用于选择 hyperparameters，也不是 DP release。
所有未加噪 oracle、clipping、norm 指标及关联图表均为 research-only，不能当作 DP 输出发布。

## 测试与 smoke

```
conda run --no-capture-output -n curve python -B -m pytest exp23/test_exp23.py -q -o cache_dir=exp23/.pytest_cache
conda run --no-capture-output -n curve bash exp23/run_all.sh --smoke
```

测试不会自动启动 smoke 或正式实验。Smoke 显式运行七个 fresh subprocess、seed=42、
1 epoch、512 私有样本（2×256）、512 测试样本；calibration 和 oracle 保留完整 2560 样本。
Smoke 使用正式 sigma 与 sample rate 来验证非零噪声路径，不能将 smoke epsilon 或准确率
解读为正式实验结果。输出独立放在 `exp23/results/smoke/`。

正式实验唯一启动入口：

```
conda run --no-capture-output -n curve bash exp23/run_all.sh
```

正式输出 `exp23/results/formal/`。脚本在全部 run 成功后调用 analyze；任何失败即退出。
分析必须读到完整 method×seed×epoch grid，不跳过缺失文件；检查 budget、计数、RMS 恒等式
以及初始化/noise RNG fingerprints。Smoke 还检查真实 private batch fingerprints。
每 run 有 metrics.csv、summary.json、config.json，preconditioned runs 还有 geometry.csv。
汇总提供 method/source mean 和 sample std（ddof=1）；所有 paired contrasts 按 seed 对齐。
核心：Penalty_A=Fashion_A−CIFAR_A，Penalty_Full=Fashion_Full−CIFAR_Full，
Interaction=Penalty_Full−Penalty_A；另列同 source A−Full、每个 public−对应 pink。
final/best accuracy 与 AUC 都做 paired 分析；AUC 为 observed epochs 上的梯形积分，
单 epoch smoke AUC=0。准确率采用 0–1 单位。仅三 seeds，描述性统计，不宣称强显著性。
图包括 accuracy curves、matched/mismatched final accuracy、mismatch penalty、
逐层 A/G alignment（含 relative error）、clipping/norm mean/p90/p99。
