# Exp1: MNIST 层内 Kronecker shape whitening

从仓库根目录启动完整实验：

```bash
conda run --no-capture-output -n curve python exp1/run_exp1.py
```

仅新增 `exp1/` 内容，不改变现有实验。使用 `src/dp_kfac/models.py` 的 SimpleCNN，固定 conv1/conv2/fc1/fc2；所有方法共用现有 per-sample Kronecker 变换与 global clipping/Gaussian noise。DP-SGD 使用单位矩阵。

完整配置：MNIST，epsilon=1，delta=1e-5，5 epochs，batch=256，C=1，SGD(lr=0.1,momentum=0.9)，seeds=[42,7]。输入归一化为仓库既有 MNIST mean=0.1307/std=0.3081。相同 seed 使用相同初始化、shuffle 顺序和 DP noise 随机流。Pink 的随机数独立于 DP noise 流；每 epoch 使用一个 256 样本的 synthetic batch、随机标签和现有 mean cross-entropy，复用 KFACRecorder、compute_covariances（默认 eps=1e-5）、compute_inverse_sqrt（默认 damping=1e-3）。

隐私记账沿用仓库 RDPAccountant/get_noise_multiplier，q=256/60000，steps=5*floor(60000/256)。训练统一 drop_last=True，使实际步数与该校准一致；每 epoch 丢弃 shuffle 后最后 96 个样本。这延续仓库 shuffle minibatch + subsampled RDP 的实验记账惯例，未改为 Poisson sampling。epsilon_spent 是这一惯例下的报告值。

CLW 的 U 初始为 identity，epoch 内固定；只在 clip_and_noise_gradients 后、optimizer.step 前读取 p.grad。EMA beta=0.99，首轮末固定 reference scale；rho=0.1，以谱范数整体缩放 R 后作 matrix exponential，并按左 G/右 A 顺序更新。EMA 和矩阵指数运算使用 float64，U 转回模型 dtype；没有 U 标量归一化或 DP noise variance subtraction。每轮更新后清零 EMA。

主 whitening metric 基于真实 MNIST diagnostic subset 上的 private-oracle KFAC A/G factors，固定取 MNIST train 前 2048 个样本，diagnostic batch size=16。每 epoch 训练结束后、CLW controller.update() 前，使用当前模型通过现有 KFACRecorder + compute_covariances 计算各层 A/G。使用 sum cross-entropy 得到不受 batch size 缩放的 backprops；激活/backprops 转 float64 后计算协方差，按 batch 样本数加权平均。oracle 显式传入 eps=0，不加入 covariance ridge 或 damping；Synthetic 的既有 eps/damping 和 synthetic batch 数量保持不变。

以该 epoch 实际训练使用的左右矩阵计算 `A' = U_A.T @ A @ U_A`、`G' = U_G @ G @ U_G.T`，然后 `Abar=A'/(trace(A')/n)`、`Gbar=G'/(trace(G')/m)`。主字段为 `W_kron=sqrt(||Abar||_F² * ||Gbar||_F²/(m*n)-1)`、`W_A=||Abar-I||_F/sqrt(n)`、`W_G=||Gbar-I||_F/sqrt(m)`，以及 `log_kappa_kron=log(kappa(Abar))+log(kappa(Gbar))`。trace normalization 使这些指标只评价层内 shape，不评价 scalar scale。删除旧 gradient-marginal 指标；CSV、summary 和现有图使用以上主字段。

Oracle 仅用于科研评价，不进入训练或 controller；DP-SGD 使用 identity，Synthetic 使用该轮实际的 synthetic 预条件器，CLW 使用更新前该轮固定的 U。W_kron 不构造 Kronecker product；平方根内部仅对浮点舍入产生的极小负值 clamp(min=0)。条件数保留数值秩判断：最小特征值 <= dimension * float64_eps * 最大特征值时记录 +inf。2048 避免 fc1 的 1569 维 A 因样本数不足而必然秩亏；真实网络死特征等造成的秩亏仍如实保留。

Oracle、train loss、clipping 统计均为科研诊断，未做隐私发布保护；它们不进入训练/controller，epsilon_spent 仅对应训练机制。train/test loss 按样本加权；clip_fraction 是变换后全局 norm > C 的样本比例，mean_clip_factor 直接使用现有 clipping 公式（含 1e-6）。

结果：

- `results/metrics.csv`：method/seed/epoch/layer 长表，效用与 clipping 字段在四层重复。
- `results/summary.csv`：最终 epoch 按 method/layer 的两 seed 均值；各 seed 最终值保留在长表。
- `results/whitening_by_layer.png`、`condition_by_layer.png`：四层曲线。
- `results/accuracy.png`：test accuracy，范围 0–1。

图中 seed 为细线，均值为粗线；不作显著性检验。每完成一个 method/seed 写出当前结果，完整执行结束才包含全部六次训练。MNIST 下载缓存位于 `exp1/data/`。自动使用可用 CUDA，否则 CPU。

唯一 tiny smoke 开关为 `--smoke`：单 seed，8 个训练/测试/诊断样本，batch=4，2 epochs，三种方法走同一训练、controller、oracle 和输出路径，结果位于 `results/smoke/`，不代表完整实验结论。

本次 oracle-KFAC 修改验证：curve 环境语法/import 通过；原有 tiny smoke 完成，共 24 行指标及三张图。CLW 第一轮与同 seed DP-SGD 的训练/测试 loss、accuracy、clipping、epsilon 和新 oracle 指标一致。未运行完整 6-run 实验。`results/smoke/` 已更新为新指标；已有 `results/` 完整实验 CSV/图没有重算，不能将旧结果解读为新 oracle-KFAC 指标，需运行完整命令重新生成。
