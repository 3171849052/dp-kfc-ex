# Exp15b: K-LBFGS damping sweep

Exp15b 只研究 damping。p 固定 0.25，synthetic lookahead lr 固定 0.5。
lambda 是原 K-BFGS 意义下的 damping，factor_damping = sqrt(lambda)。
仅 sweep lambda ∈ {1e-4, 1e-3, 1e-2, 1e-1, 3e-1}，seeds 为 42、7，共 10 runs。
结果不能用于重新选择 p。选择最佳 lambda 的 primary criterion 是
2-seed mean final test accuracy；其他 clipping/damping diagnostics 用于解释。

固定 MNIST + SimpleCNN，epsilon=1、delta=1e-5、max_grad_norm=1，
SGD lr=0.5、momentum=0、weight_decay=0；训练和评估 batch size 均为 256，
5 epochs，shuffle=True，drop_last=True。每 epoch 刷新 10 × 256 pink-noise
synthetic batches，memory=100，activation EMA=pair EMA=0.9。
沿用 exp15/exp14bc 的 fixed-batch shuffled RDP accounting：
q=256/60000，steps=5×floor(60000/256)=1170，不是 Poisson sampling。

直接复用 exp15.preconditioner.SyntheticKLBFGS 及其 InverseLBFGS，
通过配置将 sqrt(lambda) 同时传入 activation pair 的
(A + factor_damping I)s 和 sibling Modified Damping 的
Kron_BFGS_H_epsilon。Powell threshold 保持 0.2。
沿用每个 synthetic batch 从当前 private model 重置 probe 的语义；
跨 batch 保留 memories/EMAs。Smoke 复用 exp15 的 CheckedSyntheticKLBFGS 验证 reset。
private_step 直接导入 exp15，保留 GradSampleModule 和顺序：
per-sample grad → H^p → global clip → aggregate → Gaussian noise → SGD。

所有新增文件、数据副本、日志与结果放在 exp15b/。不修改已有实验。
正式运行：
```bash
conda activate curve
python exp15b/run_exp15b.py
```

单测与 smoke（不启动正式实验）：
```bash
conda activate curve
TMPDIR="$PWD/exp15b/results/tmp" PYTHONDONTWRITEBYTECODE=1 python -m pytest exp15b/tests -q -o cache_dir=exp15b/.pytest_cache --basetemp=exp15b/results/pytest_tmp
python exp15b/run_exp15b.py --smoke
python exp15b/analyze.py --smoke
```

Smoke 跑最小和最大 lambda、seed=42；每 run 2 epochs、每 epoch 2 private
steps（batch=4），3 synthetic batches（batch=8），32 test examples。
这些结果仅验证执行，不用于性能结论。数据读取自 exp15b/data（本次复制已有 MNIST）。

每 run 输出 config.json、逐 step metrics.csv、逐 epoch training.csv、
factors_epochN.json、train.log、summary.json。
clip_fraction 与 preclip mean/p90/p99 的 final 值为最后 epoch 的 batch
统计均值；p90/p99 不是整 epoch 样本合并后的分位数。metrics.csv 保留原始每步值。
factor accepted/rejected 和 damping counts 均为累计计数。
Powell/Modified rate 分别为触发数 / g-side pair 尝试数；
activation-side 不调用 damping，eligible=0、rate=0。
damping_trigger_rate 是两类检查合计触发数 / (2 × g-side pair 尝试数)，
即每项检查的触发比例，非至少一种 damping 触发的 pair 比例。

results/{formal,smoke}/summary.csv 每行一个 lambda × seed；
lambda_summary.csv 按 lambda 输出 mean/sample std 和 seed_count，
包括逐 factor 计数。四张图横轴均为 log10(lambda)，诊断使用 final epoch
clipping 与累计 damping rates；smoke 单 seed 的 std 在表中为空，绘图误差棒为 0。
正式完成 10 runs 后自动分析，也可用 python exp15b/analyze.py 重建分析。
