# Exp2：CLW feedback noise / controller frequency 消融

从仓库根目录运行唯一完整实验命令：

```bash
conda run --no-capture-output -n curve python exp2/run_exp2.py
```

仅新增 exp2/，不修改 exp1/ 或 src/。复用 Exp1 的 CLWKron.update、oracle、evaluate，以及 src 的 SimpleCNN、per-sample preconditioning 和 clipping/noise 函数。仅支持 conv1/conv2/fc1/fc2。MNIST 缓存写入 exp2/data/；结果写入 exp2/results/。

固定配置与 Exp1 相同：MNIST（mean=0.1307/std=0.3081）、SimpleCNN、epsilon=1、delta=1e-5、epochs=5、batch_size=256、drop_last=True、C=1、SGD lr=0.1/momentum=0.9、seeds=[42,7]。RDPAccountant/get_noise_multiplier 使用 q=256/60000、steps=5*234；沿用 Exp1 的 shuffle minibatch + subsampled RDP 记账惯例。相同 seed 使用相同初始化、shuffle 顺序和 Gaussian noise 随机流。没有超参数搜索或额外方法。自动使用可用 CUDA，否则 CPU。

| 方法 | controller sensor | 每 epoch 更新次数 | privacy_valid |
| --- | --- | --- | --- |
| CLW-Noisy-Slow | p.grad | 1 | True |
| CLW-Noisy-Fast | p.grad | 4 | True |
| CLW-Clean-Slow | p.summed_grad | 1 | False |
| CLW-Clean-Fast | p.summed_grad | 4 | False |

四种方法均用正常 clip + Gaussian noise 更新模型参数。调用 clip_and_noise_gradients(..., store_summed_grad=True)，其中 summed_grad 是变换、global clipping 后且加噪前的 batch-average gradient，grad 是 released noisy gradient。每层 weight 展成 m×(n−1)，bias 作为最后一列。controller 不读取 unclipped per-sample gradient。

Clean 是 oracle mechanism ablation：controller 使用非私有 feedback，整体算法不满足所报告 epsilon 的 DP 声明。Clean 的 epsilon_spent 仅为相同 nominal DP training schedule 的 accountant 数值。Noisy 的 privacy_valid=True 指训练机制在继承的记账惯例下有效；输出的 clean shadow diagnostics、private-oracle whitening、train loss 和 clipping 统计未做隐私发布保护，整份结果文件不是 DP 发布。

Slow 每轮 234 batches 后更新，共 5 次；Fast 的四段为 [1..58]、[59..117]、[118..175]、[176..234]，长度 58/59/58/59，共 20 次。每段 U 固定：读取当前 batch 的 clean/noisy feedback 后，先 optimizer.step，再在边界更新 controller、清零两套 EMA。新 U 只供下一 batch 使用，不能重处理当前 batch。Slow 第一轮全部使用 identity；Fast 第一段使用 identity。

controller 完全沿用 Exp1：Q_G=XXᵀ/n、Q_A=XᵀX/m，去掉 trace/dimension 倍单位阵；EMA beta=0.99。每个方法/seed 第一次更新固定各侧 s=||E||_F/sqrt(d)+1e-12，之后不改变。R=E/s，以 min(1,2/||R||_2) 整体缩放；K=exp(−0.1 R_hat/2)，U_G←K_G U_G，U_A←U_A K_A。EMA 和矩阵指数采用 float64，U 转回模型 dtype。没有额外归一化、damping 或 noise subtraction。

两套 shadow EMA 使用相同 beta、interval 和 reset 时机。在每个 interval 结束、更新清零前，按层/侧记录 NSR=||E_noisy−E_clean||_F/(||E_clean||_F+1e-12) 和 cos=<E_noisy,E_clean>_F/(||E_noisy||_F ||E_clean||_F+1e-12)，以及两套 EMA 的 Frobenius norm。这些是整个 interval 的 EMA 指标，不是单 batch 指标。只有选定 sensor 的 EMA 进入 controller。

Oracle 完全调用 Exp1 当前 private-oracle KFAC diagnostic：MNIST train 前 2048 样本，batch_size=16；sum cross-entropy backprops，float64 covariance，按样本数加权，eps=0，无 ridge/damping。epoch 最后一个 batch 前保存当时使用的 U 引用快照；边界 controller update 后，以结束训练时模型和这个旧 U 计算 oracle。因此评价的是最后一个 interval 实际使用的预条件器，不是下一 interval 的新 U，不增加 controller update。oracle 不进入训练/controller。

A'=U_Aᵀ A U_A，G'=U_G G U_Gᵀ；分别除以 trace/dimension 得到 Abar/Gbar。W_A=||Abar−I||_F/sqrt(n)，W_G=||Gbar−I||_F/sqrt(m)，W_kron=sqrt(||Abar||_F² ||Gbar||_F²/(mn)−1)。log_kappa_kron 为两侧 log condition number 之和；沿用 Exp1 数值秩判断，奇异时保留 +inf。这些 whitening 指标越小越好。

输出：

- metrics.csv：每 method/seed/epoch/layer 的隐私标记、效用、accountant、clipping 和四项 oracle 指标。
- controller_metrics.csv：每 interval/layer 的 method、seed、epoch、interval（每轮从 1 起）、global_update_index（每 run 从 1 起）、interval_steps、sensor_type、update_frequency（每轮次数），以及 NSR_A/G、cos_A/G、clean_A/G_norm、noisy_A/G_norm。
- summary.csv：最终 epoch 按 method/layer/privacy_valid 分组的 seed 均值。
- whitening_by_layer.png、factor_whitening_fc.png、accuracy.png：epoch 指标，fc 图分别显示 fc1/fc2 的 W_A/W_G。
- feedback_nsr.png、feedback_cosine.png：每层 A/G 的 interval-level 指标；横轴是实际 interval 结束的 epoch 位置。Slow 实线、Fast 虚线，四种方法各自颜色；所有图细线为 seed、粗线为均值。

每完成一个 method/seed 保存累计结果；完整运行才包含全部 8 runs。

判断逻辑（“优于”指 whitening 更好，并结合各层、两 seed 和效用判断）：

A. 若 Clean-Slow 明显优于 Noisy-Slow，说明 DP noise 是主要限制之一。

B. 若 Clean-Fast 明显优于 Clean-Slow，说明 Exp1 controller 更新过慢。

C. 若 Clean-Fast 优于 Clean-Slow，但 Noisy-Fast 不优于 Noisy-Slow，说明存在明显 control-speed / feedback-variance tradeoff：更快更新本来有利，但短窗口中的 DP noise 太大。结合 interval NSR 和 cosine 检查方向污染及短窗口方差。

D. 若 clean/noisy 和 fast/slow 差别都很小，主要瓶颈可能是 aggregate-gradient second moment 与 per-example KFAC/Fisher target 的结构失配，而非 noise 或更新频率。

验证使用 --smoke：单 seed=42，24 个训练样本、8 个测试/diagnostic 样本、batch=4、2 epochs，仍跑全部四种方法及原模型。Slow 每轮 6 steps；Fast 用同一整数分段公式，边界 1/3/4/6，段长 1/2/1/2，以 tiny 数据验证不等长四段。smoke 重新校准 tiny schedule 的 noise multiplier，仅验证 pipeline 和文件，不代表完整配置。输出独立放 results/smoke/。不会自动启动完整 8-run 实验，不根据 smoke 作实验结论。

本次验证：curve 环境 syntax/import 检查通过；tiny smoke 四种方法全部完成，产生 32 行 metrics、80 行 controller_metrics 和全部五张图。check_smoke.py 验证完整 234-step 边界、smoke 每轮 Slow=1/Fast=4 次更新、global update 序号、两套 feedback quality 有限性、privacy_valid 及文件输出；两种 Slow 第一轮所有数值指标完全一致，检查 identity 使用时序与随机流一致性。完整 8-run 实验未运行。首次下载遇到远端 SSL 错误，验证前将既有 exp1/data/MNIST 缓存复制到 exp2/data/MNIST；未修改原缓存，代码没有增加下载 fallback。
