# Exp28: DistilBERT / SST-2 DP-KFC-A

从仓库根目录启动完整实验（使用已有 curve 环境，GPU 2）：

```bash
conda run -n curve bash exp28/run_all.sh
```

仅调用 `scripts/paper/exp_distilbert_sst2.py::run_one`，不修改 reference。
16 个完整训练 run：C = [0.5, 1, 2, 4]，learning rate = [1e-5, 1e-4, 5e-4, 1e-3]。
固定 a_only / synthetic / bk，epsilon=3，delta=1e-5，seed=42，epochs=3，
logical/physical batch=1024/128，geometry batch/physical batch=256/16，
A_POWER=0.4，DAMPING=1e-3，MAX_LENGTH=128；profile=False，collect_diagnostics=True。
不添加 RMS scale 或其他算法改动。模型、Adam、shuffle RNG、DP noise RNG、accountant
由 reference 在每次调用时重新初始化。C 同时控制全局逐样本裁剪与 sigma*C 噪声。
模块常量只在调用期间设置，finally 恢复。

每点原始逐 epoch CSV 位于 `results/runs/C{C}_LR{learning_rate}/`；保留 reference
全部原始字段。reference 仅在最后一个 epoch 计算 validation，前两行 accuracy/test_loss
为 NaN。summary.csv 和 ranking.csv 每完成一个 run 更新；accuracy/test_loss 是最终
validation 指标，train_loss 和范数/裁剪诊断来自最终 epoch，step/accounting 计数为
三 epoch 累计（logical/optimizer/noise/accountant 各 198，physical 1581）。
noise_std 为 Gaussian noise 在 batch 平均前的 sigma*C。
排名依次按 accuracy 降序、test_loss 升序、C 升序、learning_rate 升序。

完成全部 run 后生成 accuracy、test_loss、clip_fraction、mean_clip_factor 四张
heatmap，以及 norm_p90_p99_heatmap.png。不自动 resume；已有 summary/runs 时拒绝覆盖。
日志、临时文件与库缓存均定向至 exp28/。脚本使用当前环境的 Python，不创建或安装环境。

`_validation/smoke.py` 用 mock reference 检查调用协议、常量恢复与独立目录，使用合成数据
检查排名与五张分析图；其输出仅在 `_validation/`，不代表真实实验结果。
