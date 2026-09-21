# Exp32c: 随机初始化 CNN 全参数 DP 训练

复刻 Exp32 全部训练协议，仅将 SqueezeNet 1.1 改为随机初始化，检验 pretrained initialization 是否影响 synthetic DP-KFC。

## 固定协议

- `torchvision.models.squeezenet1_1(weights=None)`；seed=42，classifier dropout=0，seeded `Conv2d(512,10,1)`，全部 727,626 个参数可训练。
- 全部 26 个 groups=1 Conv2d（含 classifier）进入 KFC geometry；bias 以齐次坐标并入 A。非参数层不进入 geometry。ReLU 使用非原地计算，确保卷积输出梯度 hook 正确。
- CIFAR-10：bicubic Resize(224,224)、ToTensor、ImageNet mean=(0.485,0.456,0.406)、std=(0.229,0.224,0.225)，无 augmentation。
- 5 epochs；50,000 训练样本；shuffle=True、drop_last=True；logical/physical batch=256，accumulation=1。每 epoch 195 steps，总计 975。
- 普通 `torch.optim.Adam`：lr=1e-4，betas=(0.9,0.999)，eps=1e-8，weight_decay=0。
- epsilon=3、delta=1e-5、clip bound=1。沿用 Exp30/31 的 `get_noise_multiplier(accountant='rdp', sample_rate=256/50000, steps=975)` 与 RDPAccountant，每个 logical batch 计步一次。这是既有固定批次 accounting convention，不是 Poisson loader。

## Geometry 与裁剪

DP-Adam 所有参数 identity；DP-KFC-A 使用 `g(A+λI)^(-0.4)`，scale 恒为 1，从不进行 RMS scale matching；DP-KFC 使用 `(G+λI)^(-1/2) g (A+λI)^(-1/2)`。

复用 `exp21.geometry.activation_matrix` 的 unfold/im2col 定义、`exp21.handlers.Record` 的 Conv2d 梯度重建和 `exp21.bk.BookKeeping` 的 BK/Fast/Ghost 范数路径。`clipping.py` 仅为 detached BK backprop 加入左侧 G 变换，不改变反向传播到上游的梯度。每个样本按整个模型的 transformed norm 联合裁剪，再对聚合梯度加各向同性高斯噪声、除以 batch size，并执行 Adam。

每 epoch 重建 geometry；复用 Exp22 synthetic pink noise（3×224×224、alpha=1、10 batches×256、physical=256）。A 按展开的空间位置求二阶矩，G 使用 synthetic random labels 的 summed cross-entropy 输出梯度，按相同空间位置计数归一化。damping 显式传入 builder 和矩阵函数，不修改任何旧模块常量。

每个 run 是独立进程，重新初始化 model、optimizer、shuffle generator、DP noise generator、accountant。model/init/private shuffle seed=42；synthetic X=seed+10000+epoch；synthetic labels=seed+20000+epoch；DP noise=seed+40000。

## 正式 grid 与 GPU

严格七个 run，四个 worker 并行，每张 GPU 内按下表顺序执行：

| GPU | 第一个 run | 第二个 run |
| --- | --- | --- |
| 0 | DP-KFC, damping=1e-3 | DP-KFC-A, damping=1e-3 |
| 1 | DP-KFC, damping=1e-2 | DP-KFC-A, damping=1e-2 |
| 2 | DP-KFC, damping=1e-1 | DP-KFC-A, damping=1e-1 |
| 3 | DP-Adam, 无 damping | — |

唯一完整实验启动命令（仓库根目录）：

```bash
conda run --no-capture-output -n curve bash exp32c/run_all.sh
```

启动器先检查目标 run 目录不存在，再验证 exp32/data 中已有 CIFAR-10（禁止下载）；已有任何正式 run 目录即报错。无自动 resume、跳过、fallback 或 batch size 下调。worker 失败时启动器返回失败，不生成不完整汇总。

## 文件与输出

`config.py` 固定协议和 grid；`model.py` 模型与 Adam；`geometry.py` synthetic factors；`clipping.py` BK 适配；`run.py` 单次训练；`worker.py` GPU 顺序队列；`run_all.sh` 启动；`analyze.py` 汇总绘图；`checks.py` 轻量检查；`__init__.py` 缓存隔离。

所有新增文件只在 exp32c 内。HF/Torch/CUDA/Triton/Inductor/matplotlib/XDG/TMP 缓存位于 `.cache/`，禁用 Python 字节码写入；数据只读复用 `../exp32/data/`；日志在 `logs/`。

每个 `results/runs/<run_name>/` 包含训练前写入的 `config.json` 和逐 epoch 更新的 `metrics.csv`。记录损失、accuracy/best/AUC、noise/epsilon、累计 accountant/logical/optimizer/noise steps、clip fraction、clip factor、norm 分位数、builder/private timing、operator/BK/临时梯度内存、逐层与 Conv group norm、builder forward/backward 数、CUDA peak memory、参数有限性与更新检查、geometry 层覆盖。`all_parameters_updated` 表示每个参数张量相较初始化均发生变化，不表示张量内每个元素均变化；`accuracy_auc` 沿用 epoch 点间梯形积分。

`analyze.py` 要求七个 run 都具备完整 epochs 1–5，再生成严格 7 行的 `results/summary.csv`（epoch 5）和每个 damping 一行的 `results/paired.csv`。生成 `damping_vs_accuracy.png`、`damping_vs_test_loss.png`、`damping_vs_clip_fraction.png`、`damping_vs_mean_clip_factor.png`、`damping_vs_norm_p99.png`；横轴 log damping，DP-Adam 为水平参考线。

## 已完成的轻量检查

在 curve 环境通过语法/import、正式 grid=7、GPU 分配=2/2/2/1、随机初始化可复现与禁止预训练下载、对比 Exp32 模型源码结构一致、224×224 单样本 forward、10 类输出、dropout=0、26 层 Conv2d 覆盖、727,626 个 trainable 参数、Adam 类型与超参数检查。微型两层 CNN 的三种方法均通过逐样本 autograd oracle 对 transformed norm 与裁剪聚合梯度的核对，并完成一次带噪 Adam smoke step。日志见 `logs/checks.log`。

轻量检查不执行完整训练，也不验证 batch=256 的完整 GPU 显存占用；不会自动降低 batch size。正式训练由启动器执行，结果完成后自动汇总。
