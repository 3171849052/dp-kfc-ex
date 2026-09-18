# Exp27：MNIST 二维超参数响应面

固定 MNIST / SimpleCNN / Adam，直接调用 `scripts/paper/exp_cnn_mnist_a.py::run_one`，不修改参考文件或重写训练、KFAC、BK clipping、privacy accountant。

- DP-KFC-Pink：`geometry="full"`；DP-KFC-A-Pink：`geometry="a_only"`。
- 两者均为 `source="pink"`、`engine="bk"`、`profile=False`。
- epochs=5，batch size=256，epsilon=2.0，delta=1e-5，seed=42。
- damping=1e-3，A-power=0.4，pink alpha=1.0（检查参考函数默认值），每 epoch 重建一次 geometry。
- C：`[0.1, 0.5, 1, 2, 5]`；learning rate：`[0.0001, 0.0005, 0.001, 0.002, 0.005]`。
- 共 25 个点 × 2 种方法 = 50 个完整 run（250 个训练 epoch）。按 C、lr 的上述顺序，每点先 DP-KFC 再 DP-KFC-A。

每次调用临时设置参考模块的 `LR` 和 `MAX_GRAD_NORM`，通过 `finally` 恢复。参考 `run_one` 每次重置模型随机种子、private shuffle generator，以及 seed+20000 的 DP noise generator；geometry 使用独立 RNG。各点使用相同 seed，保持配对可比性。

要求 `curve` 环境、CUDA 和仓库 `data/MNIST` 中已有的完整数据。数据只读加载（download=False），归一化、batch、shuffle、worker 和 drop-last 设置与参考实现一致。无需加载 FashionMNIST 或 PathMNIST。失败直接退出，不回退 CPU、不降低 batch、不跳过失败点。没有 resume：再次启动将从头执行并覆盖 Exp27 同名输出。

在仓库根目录执行：

```bash
conda run --no-capture-output -n curve bash exp27/run_all.sh
```

所有新代码、日志、缓存和输出均位于 `exp27/`。脚本保留终端训练输出并记录到 `logs/run.log`；分析输出记录到 `logs/analyze.log`。每个 run 的逐 epoch 原始 CSV 保存在独立的 `results/runs/dp_kfc_C1_LR0.001/` 等目录；每完成一个 run 更新 `results/summary.csv`。

汇总保留参考返回字段并加入 C、learning_rate、noise_std 和固定协议。accuracy 为最后一轮测试准确率（0–1），train_loss / test_loss 也是最后一轮；四种 steps 为整个 5 epoch 的累计值，正常均为 1170。noise_std = noise_multiplier × C，指除以 batch size 前的 Gaussian noise 标准差。profiling 字段的 NaN 是 profile=False 的预期输出。隐私核算沿用参考实现的固定大小 shuffled、drop-last batch 和 RDP 采样率约定。

分析仅接受完整且唯一的 50 个网格条目，生成 `dp_kfc_ranking.csv`、`dp_kfc_a_ranking.csv`、`paired_grid.csv`，并输出各方法最佳实际网格点。delta_accuracy = DP-KFC-A − DP-KFC，单位为准确率比例。并列时按 C、lr 升序选第一个。

`accuracy_surface.png` / `.pdf` 为双 panel 图，共用完全相同的准确率色标（百分数）；坐标按 log10 参数绘制、刻度显示原始值，黑点为真实网格点，白星及标注为最优真实点。等高背景只在已测网格的 log 空间内插值，不表示额外训练或测量。

轻量验证：`_validation/check.py` 使用模拟 `run_one` 返回值验证 50 次配对调用、独立目录、逐 run 汇总、全局量成功/异常恢复、失败传播、完整性检查、排名与 PNG/PDF 输出。验证日志和明确标为 synthetic 的数据位于 `_validation/`，不是正式实验结果。已在 `curve` 中通过模块导入、AST、shell 语法和已有 MNIST 读取检查；未执行训练。验证时 CUDA 不可用，正式启动需要在 CUDA 可用的运行环境中执行。
