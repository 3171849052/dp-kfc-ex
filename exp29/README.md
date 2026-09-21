# Exp29: DistilBERT / SST-2 DP-KFC-A damping sweep

从仓库根目录使用已有 curve 环境启动完整实验（GPU 1）：

```bash
conda run -n curve bash exp29/run_all.sh
```

仅直接调用 `scripts/paper/exp_distilbert_sst2.py::run_one`，不复制或修改训练、模型、
BK clipping、privacy accountant。五个 damping 点为 0.001、0.003、0.01、0.03、0.1。
保留 reference 的 matrix_power 语义，记录 effective_damping = damping + 1e-5。

固定 geometry=a_only、source=synthetic、engine=bk、epsilon=3、delta=1e-5、seed=42、
epochs=3、learning rate=5e-4、C=2、logical/physical batch=1024/128、geometry
batch/physical batch=256/16、A_POWER=0.4、MAX_LENGTH=128、profile=False、
collect_diagnostics=True。不增加 RMS scale 或其他算法改动。

每次 run_one 重新初始化模型、Adam、数据 shuffle RNG、DP noise RNG 和 accountant；
synthetic geometry RNG 由相同 seed 和 epoch 生成。MAX_GRAD_NORM=2 同时控制逐样本
全局裁剪与 Gaussian noise 的 sigma*C。模块常量在调用期间临时设置，finally 恢复。

每个 run 的原始逐 epoch CSV 保留在 `results/runs/damping_{value}/`。reference 仅在
最后一个 epoch 计算 validation，前两行 accuracy/test_loss 为 NaN。
每完成一个 run 更新 summary.csv 和 ranking.csv；accuracy/test_loss 为最终 validation
指标，train_loss 和裁剪/范数诊断来自最后一个 epoch；step/accounting 为三 epoch 累计：
logical/optimizer/noise/accountant 各 198，physical 1581。noise_std=sigma*C，表示 batch
平均前的噪声标准差。排名按 accuracy 降序、test_loss 升序、damping 升序。

全部训练完成后生成 damping_vs_accuracy.png、damping_vs_test_loss.png、
damping_vs_clip_fraction.png、damping_vs_mean_clip_factor.png 和 damping_vs_norm_tail.png
（同时展示 norm_p90/norm_p99），所有图的 damping 横轴均为 log scale。

日志、缓存和临时文件均在 exp29/。脚本使用当前环境的 Python，不创建或安装环境；
不自动 resume，已有 summary/runs 时拒绝覆盖。已有缓存复制到 exp29/ 使用，原缓存不修改。
`_validation/smoke.py` 使用 mock 训练及合成数据检查协议和分析，其输出不是真实实验结果。
