# exp16：clipping/noise 前的两阶段预条件

复用 exp15 的 MNIST + SimpleCNN、synthetic K-LBFGS、训练及隐私会计协议。
所有新代码、数据缓存和输出位于 exp16；exp15 作为只读依赖。

## 数学定义

支持 q ∈ {0, 0.25, 0.5}：

- hessian：u = H^(-q) g。
- fisher：u = F_g^(-q) g。
- nested：z = H^(-q) g，F_z = E[zzᵀ]，u = F_z^(-q) z。

Fisher 使用逐层 KFAC 近似，而非完整梯度外积矩阵。
nested 不是 (FH)^(-q)g；F_z 是在变换后的 synthetic gradient 空间中重新估计的。

exp15 的 InverseLBFGS 存储逆 Hessian，其 power(v, q) 对应 H^(-q)，
因此 hessian 路径直接调用 exp15 的 apply，不改变符号或数值实现。

对含 bias 的每层梯度矩阵 g，将常数 1 拼入 activation。
对于卷积，activation 使用 unfold patch，backprop 展平 batch 和空间位置，
协方差按所有 patch/位置取平均，与 DP-KFC 一致；这是卷积 KFAC 近似。
Stage B 收集：

    a_tilde = H_a^(-q) a
    delta_tilde = H_g^(-q) delta
    A_z = E[a_tilde a_tildeᵀ]
    G_z = E[delta_tilde delta_tildeᵀ]

逐样本应用：

    u = (G_z + λI)^(-q) [H_g^(-q) g H_a^(-q)] (A_z + λI)^(-q)

fisher 模式直接从原始 a、delta 估计 A、G。
Fisher 的 λ=0.001，与 src/dp_kfac/covariance.py 一致：
对称化后使用 torch.linalg.eigh，特征值 floor=1e-6，再计算 -q 次幂。
Hessian 仍使用 exp15 的紧凑 LBFGS 谱分解及 factor_damping=sqrt(0.001)。
代码只形成逐层 Kronecker factors，不构造完整 Fisher/Hessian。

q=0 时跳过 refresh 和全部梯度变换，保持原 grad_sample 对象及数值严格不变。

## Refresh 顺序

每个 epoch 在读取 private minibatch 前：

1. hessian/nested 的 Stage A：直接调用 exp15 SyntheticKLBFGS.refresh。
   每个 synthetic lookahead 均重置到当前模型参数；LBFGS memory/EMA 跨 epoch 保留。
2. Stage A 完成后冻结 H；Stage B 只能读取 H、计算其 power，不调用 append。
3. fisher/nested 的 Stage B：新建普通 CNN 并复制当前模型参数，
   用全新的 synthetic batches 收集 factors；不进行 lookahead，不更新模型或 H。
   Fisher factors 每个 epoch 重新估计，不跨 epoch 做 EMA。

每个启用的阶段均为 2560 samples、batch size 256（10 batches）。
Stage A seed = 42+10000+epoch；Stage B seed = 42+20000+epoch。
两个阶段都使用 fork_rng，不改变 private training 的 RNG 流。
输入沿用 exp15 的 pink noise 和随机 synthetic labels，不访问 private loader。

Private step 顺序固定为：

    backward → Hessian → Fisher → per-sample norm → clipping
             → Gaussian noise → SGD step

只启用所选模式的阶段。记录中间范数仅用于诊断，不参与裁剪。
这些 private training diagnostics 沿用实验日志用途，未额外进行 DP 发布处理。

## 固定正式训练协议

seed=42；MNIST 60000/10000；batch/eval batch=256；
shuffle=True，drop_last=True；5 epochs、1170 steps。
SGD lr=0.5、momentum=0、weight_decay=0；
epsilon 目标 1、delta=1e-5、clipping bound=1、RDP accountant。
沿用 exp15 的 fixed-batch shuffled 会计约定：
sample_rate=256/60000，按 1170 steps 校准 Gaussian multiplier。
运行设备与 exp15 一致，为 cuda:0。

配置读取 exp16/configs/mnist.yaml 和原 standalone base；
MNIST 缓存单独位于 exp16/data，避免向旧实验目录写入数据。

## 命令

仓库根目录执行：

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n curve python -m pytest exp16/tests/test_exp16.py -q -o cache_dir=exp16/results/pytest_cache > exp16/results/unit_tests.log 2>&1
PYTHONDONTWRITEBYTECODE=1 conda run -n curve python exp16/run_exp16.py --smoke > exp16/results/smoke.log 2>&1
```

Tiny smoke 跑同样七种组合，每组 2 epochs、4 DP steps；
private train/test 子集为 8/32；batch=4，每阶段 synthetic 24 samples、batch=8。
其 accuracy 仅验证流程完成，不用于性能结论。

完整 seed=42 实验（不会由测试或 smoke 自动启动）：

```bash
conda run -n curve bash exp16/run_all.sh
```

脚本仅运行 identity 一次（以 hessian_q0 命名），
随后 q=0.25/0.5 各跑 hessian、fisher、nested，最后生成汇总，无 seed=7。
单组示例：

```bash
conda run -n curve python exp16/run_exp16.py --mode nested --q 0.5 --seed 42
```

## 结果文件

- results/unit_tests.log：单元测试结果。
- results/smoke.log：完整 smoke 控制台日志。
- results/{smoke,formal}/summary.csv：七种组合最终 accuracy、epsilon、steps。
- results/{smoke,formal}/{mode}_q{q}_seed42/config.json：解析后的完整配置。
- 同目录 metrics.csv：每 step 的 loss、preclip norm mean/p50/p90/p99、
  clip fraction、实际参数 update norm、Hessian 阶段后及最终预条件后平均梯度范数，
  和各层 A/G 的 damping 后条件数（nested 中即 A_z/G_z）。
  fisher/identity 的 hessian_stage_norm_mean 是未变换的原始范数。
- 同目录 training.csv：每 epoch 的 train/test loss、test accuracy、epsilon。
- 同目录 train.log、summary.json：训练进度及最终结果。

单元测试覆盖三种模式的严格 identity、exp15 Hessian 数值一致、
与独立稠密公式及相同 Gaussian noise 的更新对照、
Stage B 冻结/独立输入/RNG 隔离、变换后的 covariance、对称性和正定性。

### 本次验证结果

- curve 单元测试：14 passed（15.73 秒）。
- 七组 CUDA tiny smoke：全部完成；每组最终 test accuracy=0.125、
  epsilon=0.9963923627、4 steps。这些数值不是正式实验结论。
- 结果完整性/有限数值检查：
  `PYTHONDONTWRITEBYTECODE=1 conda run -n curve python exp16/tests/check_smoke.py`，
  日志为 results/artifact_checks.log。
- Shell 语法检查：`conda run -n curve bash -n exp16/run_all.sh`。
- 未启动正式实验，results/formal 尚未生成。
