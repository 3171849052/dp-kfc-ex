# Exp4：Wiener-filtered Residual DP-KFC

研究原型：只新增 Wiener-Residual-AG 和 Wiener-Residual-AOnly，复用 Exp3 的 shape-only multiplicative controller、Exp1 的 pink factors / oracle / evaluate 和已有 DP 训练算子。不重跑 Exp3 baselines，不修改 src、exp1、exp2、exp3。

从仓库根目录启动完整实验：

```bash
conda run --no-capture-output -n curve python exp4/run_exp4.py
```

固定配置沿用 Exp3：MNIST、SimpleCNN、conv1/conv2/fc1/fc2、seeds=(42,7)、epsilon=1、delta=1e-5、5 epochs、batch=256、drop_last=True、C=1、SGD(lr=0.1,momentum=0.9)。每 epoch 重算 pink-noise base 并重建 residual controller。234 batches 分为 47/47/47/47/46，在 47/94/141/188 后更新 correction，下一 batch 生效。沿用 Exp3 的 shuffle minibatch RDP 记账、数据归一化、诊断集与 RNG 隔离方式。

每层将 weight 和 bias 拼成 m×n matrix。Wiener statistics 只累计当前 interval 的历史 raw released gradients：S_A+=ZᵀZ/m、S_G+=ZZᵀ/n，不使用 EMA。当前 batch 先以历史统计过滤，再交给 controller observe，随后累计当前 raw matrix，最后 optimizer.step；边界更新 controller 后清空 Wiener statistics。每段第一批使用 identity filter。

对历史均值减 tau²I，tau²=(sigma/batch_size)²；对称化、eigh、负特征值截断为零。alpha=max((mean(eig_A)+mean(eig_G))/2,1e-12)。在两个特征向量基底上使用 signal=eig_G[:,None]*eig_A[None,:]/alpha、gain=signal/(signal+tau²)，即要求的逐 Kronecker eigen-direction gain，而非独立左右滤波。

filtered matrix 通过独立的轻量 gradient view 交给 Exp3 原 observe；模型 p.grad 从未被替换或原地修改。controller 的 centered EMA、首次更新固定的 reference scales、谱范数截断和矩阵指数完全复用 Exp3。AG 更新 A/G；AOnly 跳过 G 更新，整个 epoch U_G 为 synthetic base、C_G=I。

p.summed_grad 在仓库中是 clipping 后、加噪前的 batch average，仅计算 clean shadow diagnostics，不进入 Wiener statistics 或 controller。两个方法 privacy_valid=True；含 clean shadow、oracle、loss 和 clipping 的结果文件整体不是 DP publication。

输出在 results/（smoke 在 results/smoke/）：

- metrics.csv、controller_metrics.csv、summary.csv：沿用 Exp3 的 whitening error、gain=(base-corrected)/base、accuracy、clipping、epsilon 和更新时序；feedback_type=wiener_filtered_grad。
- wiener_metrics.csv：每 method/seed/epoch/interval/layer 一行，含 tau2、alpha、alpha_over_tau2、gain mean/median/p10/p90/min/max、cos_raw/filtered、NSR_raw/filtered 和 interval_steps/fitted_batches。gain 分位数先按 batch 的所有 eigen-directions 计算，再按 interval 平均；identity warmup gain=1。alpha 和 alpha_over_tau2 仅对有历史的 batches 平均，无 fitted batch 时记 0，并以 fitted_batches=0 标明。cos 和 NSR 包含第一批，均按 batch 平均。
- whitening_gain_by_layer.png、base_vs_corrected.png、accuracy.png、wiener_feedback_quality.png、wiener_gain_distribution.png：seed 细线、seed mean 粗线；Wiener 图每 epoch 展示 5 个 intervals。

hypothesis 的判断是 cos_filtered>cos_raw 且 NSR_filtered<NSR_raw，并结合 whitening gains 和 accuracy。tiny smoke 仅验证 pipeline，不能据此下科研结论。

唯一 tiny smoke：seed=42、train=36、test=8、diagnostic=8、batch=4、2 epochs。两个方法均走正式路径，9 batches 分为 2/2/2/2/1，重新校准 tiny schedule 的 sigma。

```bash
conda run --no-capture-output -n curve python exp4/run_exp4.py --smoke
conda run --no-capture-output -n curve python exp4/check_smoke.py
```

运行时核心检查覆盖历史 count、identity warmup、filtered controller EMA、原始 optimizer gradient 和 AOnly G 不变；checker 检查完成情况、interval/更新时序、correction 侧别、Wiener 字段有限性及 gain 范围、privacy 标记和 Exp3 whitening gain 公式。没有额外 smoke 变体、golden regression 或 hardening 基础设施。

已在 conda curve / cuda:0 完成 smoke 和 checker：PASS。两个方法各完成 2 epochs，生成 16 行 metrics、64 行 controller metrics、80 行 Wiener metrics，以及 summary 和全部图像。未启动完整实验。
