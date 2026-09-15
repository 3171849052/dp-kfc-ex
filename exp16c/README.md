# exp16c: Hessian A-side ablation

研究问题：

$$
\boxed{\text{在 }\lambda_H=0.1\text{ 下，关闭 }H_G^{-q}\text{ 是否优于完整 Hessian A+G nested？}}
$$

```yaml
nested_ag: Hessian A + Hessian G + Fisher A + Fisher G
nested_a : Hessian A             + Fisher A + Fisher G
fisher   :                         Fisher A + Fisher G
```

基于 exp16b 复制 runner/Fisher 数值路径，复用 exp15 的 SyntheticKLBFGS 与现有项目 API。
所有实验输出与 MNIST 数据副本位于 exp16c/，不修改已有实验。

## 方法

Stage A：两个 nested 模式都调用 `SyntheticKLBFGS.refresh()`，以相同 synthetic
lookahead 维护 `(ha, hg)`。Hessian damping 固定 0.1。

Stage B：`nested_ag` 同时变换 activation/backprop；`nested_a` 只将 activation
变换为 $H_A^{-q}a$，backprop 不变。因此其 Fisher A 为 $A_z$，G 为原始 $G_F$。
Stage B 使用独立 pink-noise stream，不更新 Hessian memory，不访问 private batches。

private gradient（包含拼接 bias 的 Conv/Linear 矩阵）上：

- fisher：$u=G_F^{-q}gA_F^{-q}$。
- nested_ag：$u=G_z^{-q}(H_G^{-q}gH_A^{-q})A_z^{-q}$，与 exp16b nested 一致。
- nested_a：$u=G_F^{-q}(gH_A^{-q})A_z^{-q}$。

`nested_a` 手动调用 `ha.power`，不调用 `SyntheticKLBFGS.apply` 或 `hg.power`。
保留项目的末维乘法/转置约定，Fisher apply 与 exp16b 相同。
所有变换先于 per-sample norm → clipping → Gaussian noise → SGD。

## 固定协议

seed=42；q ∈ {0.25, 0.5}；MNIST + SimpleCNN；batch=256；5 epochs；1170 private steps；
SGD lr=0.5，momentum=0，weight_decay=0。DP epsilon=1、delta=1e-5、clip=1，
沿用 fixed-batch shuffled / drop_last RDP accountant。
Hessian damping=0.1，memory=100，initial_scale=1，activation_decay=pair_decay=0.9，
synthetic_lookahead_lr=0.5；Fisher damping=0.001。
Pink-noise synthetic samples=2560，batch=256，每 epoch refresh。

## 验证与输出

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n curve python -m pytest exp16c/tests/test_exp16c.py -q -o cache_dir=exp16c/results/pytest_cache
PYTHONDONTWRITEBYTECODE=1 conda run -n curve python exp16c/run_exp16c.py --smoke
PYTHONDONTWRITEBYTECODE=1 conda run -n curve python exp16c/tests/check_smoke.py
```

smoke 对六个组合各运行 2 epochs、4 private DP steps、2 evaluations，使用 CUDA，
private batch=4、train subset=8、test subset=32、synthetic samples=24/batch=8。
这些缩小配置仅限 smoke。结果在 `results/smoke/summary.csv` 和各组合子目录。

每个组合保存 config.json、metrics.csv、training.csv、train.log、summary.json。
metrics 包括 Hessian-stage/final norm、preclip mean/p50/p90/p99、clip_fraction、
update_norm、train_loss，以及每层 A/G damped condition；training 保存
train_loss、test_loss、test_accuracy、epsilon_spent。nested_a 的条件数对应 A_z / G_F。

测试覆盖：配置、与 exp16b 的逐元素一致性、G covariance 精确相等、独立 dense
A covariance 与最终梯度公式、禁止 hg.power、clipping/noise 顺序、Stage B memory 冻结、
synthetic refresh 隔离。smoke 检查所有数值日志 finite。

正式实验输出在 `results/formal/`，run_all.sh 只运行 seed=42 的六个组合。
测试和 smoke 不启动正式实验；smoke 不能用于判断研究问题的效果。

```bash
conda run -n curve bash exp16c/run_all.sh
```

### 本次验证结果

- pytest：18 passed（17.53s）；日志 `results/pytest.log`。
- 六组 CUDA smoke：全部 PASS，每组 4 DP steps / 2 evaluations，数值日志全部 finite。
- smoke 运行日志：`results/smoke.log`；检查日志：`results/smoke/check.log`。
- `nested_ag` 与 exp16b nested 的 covariance / private gradient 逐元素一致。
- `nested_a` G covariance 与 fisher 精确相等；dense covariance 和 CNN gradient
  在 float64 下使用 rtol=1e-8、atol=1e-9；实际 float32 clipping/noise/SGD 更新另有测试。
- 正式实验尚未启动。
