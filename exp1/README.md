# Exp1: MNIST 层内 Kronecker shape whitening

从仓库根目录启动完整实验：

```bash
conda run --no-capture-output -n curve python exp1/run_exp1.py
```

仅新增 `exp1/` 内容，不改变现有实验。使用 `src/dp_kfac/models.py` 的 SimpleCNN，固定 conv1/conv2/fc1/fc2；所有方法共用现有 per-sample Kronecker 变换与 global clipping/Gaussian noise。DP-SGD 使用单位矩阵。

完整配置：MNIST，epsilon=1，delta=1e-5，5 epochs，batch=256，C=1，SGD(lr=0.1,momentum=0.9)，seeds=[42,7]。输入归一化为仓库既有 MNIST mean=0.1307/std=0.3081。相同 seed 使用相同初始化、shuffle 顺序和 DP noise 随机流。Pink 的随机数独立于 DP noise 流；每 epoch 使用一个 256 样本的 synthetic batch、随机标签和现有 mean cross-entropy，复用 KFACRecorder、compute_covariances（默认 eps=1e-5）、compute_inverse_sqrt（默认 damping=1e-3）。

隐私记账沿用仓库 RDPAccountant/get_noise_multiplier，q=256/60000，steps=5*floor(60000/256)。训练统一 drop_last=True，使实际步数与该校准一致；每 epoch 丢弃 shuffle 后最后 96 个样本。这延续仓库 shuffle minibatch + subsampled RDP 的实验记账惯例，未改为 Poisson sampling。epsilon_spent 是这一惯例下的报告值。

CLW 的 U 初始为 identity，epoch 内固定；只在 clip_and_noise_gradients 后、optimizer.step 前读取 p.grad。EMA beta=0.99，首轮末固定 reference scale；rho=0.1，以谱范数整体缩放 R 后作 matrix exponential，并按左 G/右 A 顺序更新。EMA 和矩阵指数运算使用 float64，U 转回模型 dtype；没有 U 标量归一化或 DP noise variance subtraction。每轮更新后清零 EMA。

Oracle 固定 MNIST train 前 1024 个样本，分 16 样本小批计算 clean per-sample gradients。在每 epoch 训练结束后、CLW 更新前，以当前模型和该轮实际使用的 U 测量不中心化二阶矩；不 clipping、不加 noise。矩阵累积/eigensolver 用 float64；trace normalization 中样本数量相消。W_kron 不构造 Kronecker product。W 的平方根只截去浮点舍入引起的负值。条件数不加 damping；最小特征值 <= dimension * float64_eps * 最大特征值时记录 +inf（数值秩亏），图中注明，CSV 保留。尤其 ReLU 的死特征可能使 fc1/fc2 条件数为无穷。

Oracle、train loss、clipping 统计均为科研诊断，未做隐私发布保护；它们不进入训练/controller，epsilon_spent 仅对应训练机制。train/test loss 按样本加权；clip_fraction 是变换后全局 norm > C 的样本比例，mean_clip_factor 直接使用现有 clipping 公式（含 1e-6）。

结果：

- `results/metrics.csv`：method/seed/epoch/layer 长表，效用与 clipping 字段在四层重复。
- `results/summary.csv`：最终 epoch 按 method/layer 的两 seed 均值；各 seed 最终值保留在长表。
- `results/whitening_by_layer.png`、`condition_by_layer.png`：四层曲线。
- `results/accuracy.png`：test accuracy，范围 0–1。

图中 seed 为细线，均值为粗线；不作显著性检验。每完成一个 method/seed 写出当前结果，完整执行结束才包含全部六次训练。MNIST 下载缓存位于 `exp1/data/`。自动使用可用 CUDA，否则 CPU。

唯一 tiny smoke 开关为 `--smoke`：单 seed，8 个训练/测试/诊断样本，batch=4，2 epochs，三种方法走同一训练、controller、oracle 和输出路径，结果位于 `results/smoke/`，不代表完整实验结论。

本次验证：curve 环境语法/import 通过；上述 tiny smoke 完成，共 24 行指标和三张图。确认 CLW 第一轮与 DP-SGD 的效用、clipping 和 whitening 数值一致；核心矩阵检查通过（feedback traceless、更新行列式约为 1、epoch EMA 清零）。未运行完整实验。本机下载端点失败后，手动复制已有 MNIST 原始缓存到 `exp1/data/MNIST/raw/`，没有增加下载兼容逻辑。
