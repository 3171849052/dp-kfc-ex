# Exp26b: Full DP-KFC damping sweep

只研究 damping 能否压低 Exp26 中异常的 preconditioned gradient norm tail 并改善 utility。
直接调用当前 `scripts/paper/exp_distilbert_sst2.py` 的 BK `run_one`，沿用 Exp26 的实验包装方式，不修改 reference。

- Damping：`[1e-3, 3e-3, 1e-2, 3e-2, 1e-1]`，共 5 个独立完整 run。
- 固定：full geometry、synthetic source、BK、完整 SST-2 train、3 epochs、seed 42。
- Adam，LR `5e-4`，默认 weight_decay `0`，无 scheduler。
- C `2.0` 同时控制 per-example global clipping threshold 和平均前 Gaussian noise std `sigma * C`。
- epsilon `3.0`，delta `1e-5`；logical batch `1024`，physical batch `128`。
- geometry batch `256`，geometry physical batch `16`，max length `128`。
- 保留原公式 `covariance + (DAMPING + 1e-5) * I`；不改 matrix power、KFAC layer selection、synthetic prompts、每 epoch geometry 更新或 accountant。

每次调用 reference 都 reset seed，并新建 pretrained model、Adam、DataLoader RNG、DP-noise RNG、accountant。
Damping 只修改 reference 的 `DAMPING`，不参与 sigma、sample rate 或 accounting 计算。
运行器检查所有 run 的 privacy calibration、batch sizes 和 step counts 相同。
不运行 DP-Adam、A-only 或 Explicit，不做第二阶段 multi-seed，不做 checkpoint/resume 或 fallback。

调用固定为 `profile=False, collect_diagnostics=True`，关闭 timing、CUDA/BK memory profiling 和 profiling counters。
保留每 epoch 的 clip_fraction、mean_clip_factor、norm_p50、norm_p90、norm_p99、norm_max。
Reference 在运行中仍写含空 profiling 列的中间 CSV；每个 run 完成后按固定字段表覆盖，最终 CSV 不含这些列，也不分析它们。

每个 run 在 `results/full_synthetic_bk_damping0.001_eps3_seed42.csv` 等独立文件中保留 3 个 epoch 行。
Reference 只在最后 epoch 评估 validation，因此前两行 accuracy/test_loss 为空。
`results/summary.csv` 每个 damping 一行，utility 和 diagnostics 取 final epoch；遵循 Exp26/reference，step counts 是三轮累计，epsilon_spent 是累计隐私消耗。Per-run CSV 的 step counts 则为各 epoch 的计数。

分析按 damping 升序读取 summary，按 validation accuracy 降序输出 `ranking.csv`（并列时按 validation loss、damping 升序），打印最佳 damping。
生成 `damping_vs_accuracy.png`、`damping_vs_loss.png`（train/validation）、`damping_vs_norm_tail.png`（p90/p99）、`damping_vs_clip_fraction.png`，所有 x 轴均为 log scale。

轻量检查：

```bash
conda run -n curve python -m py_compile exp26b/config.py exp26b/run.py exp26b/analyze.py
conda run -n curve pytest -q tests/test_distilbert_sst2.py
```

完整实验（需要 CUDA；不会由轻量检查启动）：

```bash
conda run -n curve bash exp26b/run_all.sh
```
