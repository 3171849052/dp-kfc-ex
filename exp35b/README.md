# Exp35b

从仓库根目录启动（不自动 resume；只跑两个新 alpha）：

```bash
conda run --no-capture-output -n curve bash exp35b/run_all.sh
```

固定 GPU 1 顺序运行 alpha=.25 的 MNIST、ViT；GPU 2 顺序运行 alpha=.75 的 MNIST、ViT。两个 GPU 并行，各 run 为独立进程。seed=42，其他模型、优化器、数据、accounting、RNG 与 Exp35 相同。谱算子复用 Exp35 的 norm 算子，然后计算 `(1-alpha) I + alpha P_norm`，每层断言 gain 范围。

`runtime.py` 在进程内将原 Exp35 包的 ROOT 重定向至本目录，直接加载原模块；不会执行原位置的缓存初始化或修改旧文件。`worker.py` 注入两个方法、谱公式和输出配置。训练复用 `exp35.mnist` / `exp35.vit`，最终调用原 MNIST/Exp30 训练循环。继承全部 loss、accuracy、clip、norm、gain 及 ViT layer/group diagnostics。

数据仅读取根目录 `data/`，download=False。预训练权重复制自 Exp30 已有 cache。并行 worker 的缓存分别位于 `workers/alpha025/.cache/` 和 `workers/alpha075/.cache/`；其余缓存也留在本目录。日志在 `logs/`，正式 metrics/config 在 `results/{mnist,vit}/runs/dp_kfc_a_alpha{025,075}/`。分析生成两个数据集及总 summary、各 epoch 的 accuracy/clip/mean-clip-factor/p90/p99 对 alpha 图。

可在正式运行完成后只读合并 Exp35 的 seed42 baseline（alpha=0）、mix05（.5）、norm（1），生成独立的 `*_five_point.csv/png`，不复制旧 results：

```bash
conda run --no-capture-output -n curve python -B exp35b/analyze.py --include-exp35
```

轻量检查（CPU，无正式训练或伪正式结果）：

```bash
conda run --no-capture-output -n curve python -B exp35b/checks.py --variant alpha025
conda run --no-capture-output -n curve python -B exp35b/checks.py --variant alpha075
```
