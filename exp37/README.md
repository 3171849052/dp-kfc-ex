# Exp37：Trace-A / Trace-AG layer-scale

在 pretrained ViT-Tiny + CIFAR-10 上运行 11 个固定实验：`dp_adamw`，
`trace_a_p01` 至 `trace_a_p05`，`trace_ag_p01` 至 `trace_ag_p05`。
p = 0.1、0.2、0.3、0.4、0.5；identity 只运行 DP-AdamW。

## 算法

每个 trainable Linear（包括 patch embedding、拆分后的 Q/K/V 和 head）累计
synthetic activation 的平方和，得到 `a = Tr(A)/d_A`。
与 exp22 一致，带 bias 的 activation 增广常数 1，且 d_A 包含这一维。
计数分母为展平后的 sample × token 数量。Trace-AG 另累计 backward-error
平方和，得到 `g = Tr(G)/d_G`；backward 使用 sum cross entropy，与 full KFAC 一致。

- Trace-A：`r = (a + 1e-3)^(-p)`。
- Trace-AG：`r = ((a + 1e-3)(g + 1e-3))^(-p)`。
- 每次构建统一归一化：`c = sqrt(sum(D) / sum(D*r*r))`，`s = c*r`。
  D 是每个 Linear weight+bias 的参数数目，只对这些 Linear 做一次全模型 RMS normalization。

`LayerScaleOperator` 只保存 `{layer_name: scalar}`，不保存 trace 或矩阵。
在原 BK 的增广 activation 上乘 s，等价于构造逐样本梯度时 weight+bias 同乘 s。
LayerNorm、position embedding、CLS token 仍由原 identity 路径处理。
流程为 **scalar transform → per-example global clipping → aggregate → Gaussian noise → AdamW**。
无 covariance matrix、eigendecomposition、gain cap、EMA、adaptive exponent 或 spectral normalization。

## 训练与复用

完整继承 exp36d 协议：

- `vit_tiny_patch16_224.augreg_in21k_ft_in1k`，pretrained full-model fine-tuning。
- CIFAR-10 只读取仓库根 `data/`，`download=False`；224×224 bicubic，
  mean/std 均为 (0.5,0.5,0.5)，无 augmentation。
- AdamW lr=1e-4，weight_decay=.01，betas=(.9,.999)，eps=1e-8。
- epochs=5，seed=42，logical batch=256，physical batch=128，C=2。
- epsilon=3，delta=1e-5，975 logical/optimizer/noise/accountant steps。
  保留原 fixed shuffled/drop_last RDP accounting 和 noise multiplier 计算。
- 每 epoch 重建 geometry，pink probes 10×256；synthetic physical batch=256。
  直接复用 exp22 synthetic_stream，连续 RNG seed=42+10000+epoch。
- Trace-A 只 forward，不创建 label RNG、不生成 labels、不 backward。
  Trace-AG 复用 full KFAC 的 label 函数与连续 RNG seed=42+20000+epoch，
  每个 logical probe batch 生成 labels 后按原 physical chunk 顺序 backward。
- DP noise seed=40042，聚合后加 sigma*C Gaussian noise，再除以 logical batch size。

`config.py` 使用 exp35b.runtime.configure 隔离 exp35 的输出根目录；
`worker.py` 沿用 exp36d 的依赖注入结构，复用 exp35.vit 数据加载/checkpoint、
exp30 的训练循环、exp35.adapters 的 prepare/MetricsSink，以及未改动的
exp22.methods.Clipper。没有复制或重写训练框架。
已有 exp30 pretrained cache 只读复制到本实验每卡 cache，offline 加载。

## 启动

唯一完整实验启动命令（仓库根目录）：

```bash
conda run --no-capture-output -n curve bash exp37/run_all.sh
```

| Physical GPU | 串行 runs |
|---|---|
| 1 | dp_adamw, trace_ag_p01, trace_ag_p04, trace_a_p01 |
| 2 | trace_ag_p02, trace_ag_p05, trace_a_p02, trace_a_p04 |
| 3 | trace_ag_p03, trace_a_p03, trace_a_p05 |

三卡固定 CUDA_VISIBLE_DEVICES 并行；任何 run 失败，该卡停止后续任务，
脚本等待三个 worker 结束并非零退出。全部成功才执行 analyze.py。
启动前拒绝任何已存在的 run 目录；单独 worker 也通过 exist_ok=False 拒绝覆盖。
没有自动 resume、fallback、动态调度或自动降 batch。

## 输出与分析

所有写入位于 exp37/：`logs/`、`.cache/`、`workers/gpuN/.cache/`，以及
`results/vit/runs/<method>/config.json`、逐 epoch `metrics.csv`。

保留 exp36d 的全部训练/测试、privacy、计数、时间/内存与 norm 指标，并增加：

- `trace_a_per_dim_<layer>`，AG 另有 `trace_g_per_dim_<layer>`。
- `raw_scale_<layer>`、`scale_<layer>`，及 `scale_min/p10/p50/p90/max`。
- `family_norm_attention_qkv/attention_out/mlp_fc1/mlp_fc2/patch_head`：
  transformed per-example norm 的 RMS；fc1/fc2 使用各层 RMS 平方和开根号。
- 原 clip fraction、mean clip factor、transformed norm p50/p90/p99/max。

Baseline 不构建 geometry，trace 不适用且不记录；各层 raw_scale/scale 均为 1。
Trace-A 不记录 G。trace 和 raw scale 只进入 metrics，不存入 operator。

analyze.py 要求 11 个 runs 均包含完整 5 epochs 和一致步数，否则直接失败。
生成 `results/summary.csv`（保留 final_accuracy、test_accuracy、best_accuracy），
accuracy vs p 双曲线和 DP-AdamW 水平基线、clip fraction/mean clip factor vs p、
layer family norm ratio vs p、scalar gain min/p10/p50/p90/max vs p，
以及 test loss、norm diagnostics 和 family norm/ratio CSV。
所有比较使用最后一个 epoch；family ratio 分母为本实验 baseline 同一 family norm。

## 轻量检查

```bash
conda run --no-capture-output -n curve python -B exp37/checks.py
```

只做 CPU、小模型数值与协议检查：grid/GPU、数据与 transform、runtime 写入隔离、
与原 full KFAC 的 A/G trace 和 label RNG 对照、两个公式及所有 p、参数加权 RMS、
BK per-example norm 和 clipped aggregate、聚合后的 Gaussian noise、metrics 边界和缺失结果拒绝。
不初始化正式 ViT，不启动训练，不产生正式结果。检查日志为 `logs/checks.log`。
