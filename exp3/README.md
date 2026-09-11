# Exp3：Residual DP-KFC 研究原型

验证历史 gradient feedback 学到的乘法 residual 能否降低 synthetic pink-noise KFAC 剩余的 shape whitening error。只训练 Residual-Noisy 与 Residual-Clean，不重跑 baseline；只新增 exp3/，复用 exp1 的 pink_factors、oracle、evaluate 和 src 的训练算子。

完整实验唯一启动命令（从仓库根目录）：

```bash
conda run --no-capture-output -n curve python exp3/run_exp3.py
```

配置：MNIST、SimpleCNN 的 conv1/conv2/fc1/fc2，epsilon=1、delta=1e-5、5 epochs、batch=256、drop_last=True、C=1、SGD(lr=0.1, momentum=0.9)、seeds=[42,7]。归一化 mean=0.1307/std=0.3081，复用 exp1/data 缓存。沿用 Exp1/2 的 RDPAccountant/get_noise_multiplier（q=256/60000，steps=5×234）及 shuffle minibatch 记账惯例。相同 seed 的初始化、shuffle 和 Gaussian noise 流相同；每 epoch pink batch 使用隔离 RNG，seed+10000+epoch，调用 Exp1 原函数，保持 mean reduction、eps 和 damping。自动使用可用 CUDA，否则 CPU。

每 epoch 对当前模型重算 synthetic base，再新建 controller：C_A/C_G=I，EMA=0，reference scales 尚未设置。因此第一段 U 就是本轮 base，不跨 epoch 继承 residual、EMA 或 scales。按实现中的矩阵梯度约定 Z'=U_G Z U_A，采用 U_G=C_G G_s^(-1/2)、U_A=A_s^(-1/2) C_A；A 侧乘法顺序遵循需求中的明确更新式。

controller 读取 weight 与 bias 拼接的 gradient matrix。Residual-Noisy 只读取 p.grad，privacy_valid=True；Residual-Clean 只读取 p.summed_grad，privacy_valid=False，作为 oracle ablation。两者均调用 store_summed_grad=True；仓库 summed_grad 实际是 clipping 后、加噪前的 batch average。两者用于 SGD 的仍是加噪后的 p.grad。

Q_G=ZZᵀ/n，Q_A=ZᵀZ/m；减去 trace(Q)/dimension 倍单位阵。EMA=0.99 EMA+0.01 Q_centered，每 epoch 首次更新固定 s=||E||_F/sqrt(d)+1e-12。R=E/s，R_hat=R min(1,2/||R||_2)，K=exp(-0.1 R_hat/2)。U_G←K_G U_G，U_A←U_A K_A，同时按相同方向维护 C 以记录 ||C-I||_F/sqrt(d)。EMA、矩阵指数和 C 累积采用 float64，U 转回模型 dtype。每次更新清零 EMA，不显式减 noise variance，不学习 scalar scale。

234 batches 分为 47/47/47/47/46，只在 47/94/141/188 后更新。每 batch：当前 U 预条件 per-sample gradient → global clipping → Gaussian noise → observe → optimizer.step → 边界时 update。新 U 从下一 batch 起使用，最后一段不再更新。全局更新序号跨 epoch 累加，每个 run 从 1 开始。

每 epoch 末固定当前模型参数，使用同一顺序的 train 前 2048 样本、diagnostic batch=16，分别调用 Exp1 原 oracle 评价本轮 base 和最后一段实际使用的 corrected U。SimpleCNN 无随机层，两次调用无参数更新，因而得到同一 oracle covariance；额外一次调用隔离 RNG，保持后续训练随机流。Oracle 使用 sum reduction、float64 covariance、eps=0，不参与 controller 或训练。W_A/G/kron 及 log condition number 完全继承 Exp1，真实秩亏保留 +inf。gain=(base_W-corrected_W)/base_W，正数表示改善。

privacy_valid 指训练机制在继承记账惯例下的标记；Clean 的 epsilon_spent 只是 nominal schedule 的数值。Private oracle、训练 loss 和 clipping 统计均为科研诊断，结果文件整体不是 DP 发布。

输出到 results/：

- metrics.csv：每 method/seed/epoch/layer 一行，包含训练、测试、隐私及 clipping 指标、base/corrected 的四项诊断、gain_A/G/kron；base_reset_error 记录 epoch 初 U 与新 base 的最大差值（应为 0）。
- controller_metrics.csv：每次 update/layer 的 interval、global_update_index、interval_steps、实际 feedback 属性名、EMA norm、C-I norm 和 K 的最小/最大特征值；update_after_step 与 applies_from_step 标明生效时序。
- summary.csv：最后 epoch 按 method/layer/privacy_valid 的 seed 均值。
- whitening_gain_by_layer.png：各层 gain_kron；base_vs_corrected.png：各层 W_kron（base 虚线、corrected 实线）；accuracy.png。seed 细线、seed 均值粗线。

一个 tiny end-to-end smoke（不启动完整实验）：

```bash
conda run --no-capture-output -n curve python exp3/run_exp3.py --smoke
conda run --no-capture-output -n curve python exp3/check_smoke.py
```

seed=42、train=36、test=8、diagnostic=8、batch=4、2 epochs，两个方法共用正式代码路径。9 batches 分成 2/2/2/2/1，只在 2/4/6/8 后更新，输出 results/smoke/。重新校准 tiny schedule 的 noise multiplier。运行中的 identity reset 和更新边界断言，以及 check_smoke 的结果检查仅覆盖核心 invariant：两方法完成、epoch reset、反馈属性、历史更新时序、privacy_valid、whitening 字段有限性和 gain 公式。没有额外 smoke 变体或 hardening suite。tiny 结果仅说明 pipeline 可运行，不能据此判断 hypothesis。

已在 curve 环境、cuda:0 完成上述 smoke 和 check_smoke：PASS，16 行 metrics、64 行 controller_metrics，summary 与三张图已生成。两个方法各完成 2 epochs、每 epoch 4 次更新。完整实验尚未启动。
