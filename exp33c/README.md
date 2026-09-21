# Exp33c：Synthetic Wiener-A 收缩强度

仅改变 `gamma=(0, 0.1, 0.25, 0.5, 0.75, 1)`。复用 Exp33/Exp33b 的训练协议、synthetic estimator 和 Exp33b 的真实 AdamW step diagnostics。

## 固定协议

- `vit_tiny_patch16_224.augreg_in21k_ft_in1k`，正式训练 `pretrained=True`，复用 `exp22.model.initialize` 与 explicit ViT conversion，全参数微调，74 个 Linear（含 bias）应用 Wiener-A。
- CIFAR-10，bicubic Resize 224×224、ToTensor、mean/std 均为 `(0.5,0.5,0.5)`，无 augmentation。只读复用 `exp30/data` 已有数据，缺失时直接报错。
- seed 42，5 epochs；logical batch 256、physical batch 128、accumulation 2，drop-last，每 epoch 195 steps，共 975 optimizer/accountant/noise steps。
- AdamW 所有参数恒定 LR `1e-4`，betas `(0.9,0.999)`、eps `1e-8`、weight decay `0.01`，无 scheduler。
- epsilon 3、delta `1e-5`、global clipping C=1；与 Exp33 相同的 RDP fixed logical-batch convention。沿用原 Clipper 的浮点实现 `min(1,C/(norm+1e-6))`。
- 每个 run 在独立新进程初始化 model、optimizer、accountant 和 RNG。shuffle seed=42；DP noise seed=`seed+40000`；synthetic x/labels seed 分别为 `seed+10000+epoch` / `seed+20000+epoch`。

## 算子与顺序

每 epoch 生成 10 个 256 样本的 pink-noise logical batches（alpha=1，3×224×224），synthetic physical batch=128。synthetic gradients 按全模型 global norm clipping，形成每层含 bias 的 clean logical-batch mean matrix Z。

`A = mean_b(Z_b.T @ Z_b / output_dimension)`，`A = U diag(lambda) U.T`；`tau2=(sigma*C/256)^2`；原始 gain `h=lambda/(lambda+tau2)`；有效 gain `h_gamma=(1-gamma)+gamma*h`。只消除协方差特征值的负浮点舍入误差，完全沿用原 estimator。

Private path：`clip → aggregate → DP noise → Z U diag(h_gamma) U.T → AdamW`。`gamma=0` 在 apply/transform 直接跳过矩阵运算和 gradient 写回，精确 identity，但仍正常构建 synthetic operator 供 diagnostics。`gamma=1` 复现 Exp33 Wiener-A。运行时检查有效 gain 在 `[1-gamma,1]`（tol=`1e-7`）。LayerNorm、CLS、positional embedding 为 identity。不使用 Full Wiener、RMS scale、独立 Linear LR、private gradient history 或其他 gain heuristic。

## Diagnostics

每 epoch 的 `metrics.csv` 保留训练/测试 loss、accuracy、best accuracy、accuracy AUC（沿用梯形积分定义）、privacy/step counters、clipping、private norm 分位数、raw/effective gain 的 mean/median/p10/p90/p99/min/max、norm ratio mean/p10/p50/p90、clean-shadow cosine/NSR、builder/private 时间、operator bytes、参数 finite/updated 状态。

`group_raw_wiener_gain_<group>`、`group_effective_gain_<group>` 保存组内谱元素均值；`group_wiener_norm_ratio_<group>` 保存每 step 组内 Frobenius norm ratio 的 epoch 均值，group 为 patch_head、attention_qkv、attention_out、mlp。全局 ratio 包含 identity 参数。

每个 logical step 在 `optimizer.step()` 前后复制参数并直接测量实际更新，包含 Adam moments、epsilon 和 weight decay。记录上述四组及 identity 的 `update_norm_<group>` 与 `relative_update_norm_<group>`，后者为 `||delta theta||/(||theta_before||+1e-30)`，CSV 为各 step 值的 epoch 算术平均。

Clean shadow 是 clipping 后、加噪前的 aggregate mean，仅用于 cosine/NSR 诊断，不参与 fitting、gamma、operator、optimizer 或任何适应。**这些 clean-shadow、clipping、private norm 等科研 diagnostics 不是 DP-publishable diagnostics；CSV 整体不应视为 DP 可发布结果。** Accountant 记录的预算对应原训练机制，不覆盖这些额外诊断发布。

## 运行与输出

唯一完整实验启动命令（仓库根目录）：

```bash
conda run --no-capture-output -n curve bash exp33c/run_all.sh
```

固定四 worker 并行，同卡顺序：GPU 0 为 gamma 0、0.75；GPU 1 为 0.1、1；GPU 2 为 0.25；GPU 3 为 0.5。run names 为 `wiener_a_gamma_<gamma>`（端点为 `0`、`1`）。已有任何正式 run 目录直接报错，不 resume 或跳过。

所有新增代码、日志、结果在本目录；TMP、Torch、HF、Triton、CUDA、matplotlib、Numba、Python cache 均限定在 `.cache/`，禁写外部 bytecode。数据和公共算法仅只读复用。每 run 保存 `results/runs/<run_name>/config.json`、`metrics.csv`；worker 日志在 `logs/gpu<id>.log`。

六个 run 全部完成后自动生成 epoch 5 的六行 `results/summary.csv` 和六行 `paired.csv`（含 `delta_accuracy_vs_gamma0`），以及线性 gamma 横轴的十张图：accuracy、accuracy_auc、test_loss、clip_fraction、wiener_norm_ratio、effective_gain、cosine、nsr、update_norm、relative_update_norm，文件名为 `gamma_vs_<指标>.png`。Accuracy 图标记两端点、Exp33 DP-AdamW 0.9276 参考和 interior optimum。Exp33 Wiener-A 0.7180 也存入 summary，仅作参考；主要比较使用本实验两端点。

## 轻量验证

`checks.py` 在 CPU toy model 上验证逐样本 global clipping、covariance/gain、gamma 公式/边界、与原 Exp33 builder/transform 的直接逐位对照、精确 identity、独立 noise/filter/AdamW 两步路径、clean shadow、真实更新范数、operator 不变与 RNG 重复性。ViT 使用 `pretrained=False` 检查 74 Linear 的结构和 conversion 前后输出一致性，不下载权重；正式 initialization 的 `pretrained=True` 通过共享函数/源码核验。

还验证 975-step RDP、固定 grid/GPU/LR、缓存约束、临时模拟数据上的六行分析与十图生成和不完整 run 拒绝。模拟结果仅在 `.cache/` 临时目录生成并清理；不创建正式 run、不启动完整训练。执行记录见 `logs/checks.log`。
