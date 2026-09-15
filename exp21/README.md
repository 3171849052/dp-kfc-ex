# Exp21: Hybrid Book-Keeping + Ghost Differentiation

所有实现、测试、日志、输出均在 `exp21/`；只读复用 Exp19/20 的 operator、噪声/SGD、profiling 与 MNIST protocol。正式 25-job 实验未启动。

## 运行

在仓库根目录执行：

```bash
conda run -n curve python -B exp21/test_exp21.py
conda run -n curve bash exp21/run_all.sh --smoke
conda run -n curve python -B exp21/analyze.py --smoke
# 正式实验：5 methods × 5 seeds，每个 job 独立 subprocess
conda run -n curve bash exp21/run_all.sh
```

正式配置：MNIST / 当前 SimpleCNN；batch=256，5 epochs；seeds=(42,7,123,2024,3407)，power=.25，damping=.001，C=1，epsilon=1，delta=1e-5；SGD lr=.5，momentum=0，weight_decay=0。每 epoch 10×256 pink-noise synthetic samples，只做 forward 建 A。按 seed 轮换 method 顺序。每个 subprocess 用相同 batch size 的独立 disposable warmup，保存/恢复 RNG。

Smoke：seed=42，前 768 个训练样本 shuffle 后 3 batches，1 epoch，1×256 synthetic samples，noise=0，测试集前 256 个样本。完整数据继续读取已有 `exp1/data`，不下载、不修改。

初始化用 CPU seed；shuffle 用独立 CPU generator(seed)；噪声 generator(seed+40000)；synthetic 使用仅当前 CUDA device 的 fork_rng(seed+10000+epoch)。沿用 Exp20 shuffled fixed-batch RDP accounting 约定；此实验不提供新的 privacy proof。正式总计 1170 private steps。

## 文件

- `config.py`：固定协议。
- `handlers.py`：Linear/Conv2d、稀疏 Embedding、LayerNorm/RMSNorm、Conv1D layout，以及 Ghost/Fast/BK 公式。
- `bk.py`：activation/backprop capture、GD anchor、identity-based shared parameter 合并、缓存生命周期、两遍对照路径。
- `methods.py`：Opacus explicit baseline 与统一 Clipper。
- `profiling.py`：复用 Exp20 deferred CUDA events。
- `run_one.py` / `run_sweep.py` / `run_all.sh`：独立进程实验入口。
- `test_exp21.py`：独立 naive per-example backward 金标准和 CUDA/MNIST 集成测试。
- `analyze.py`：汇总、配对 speedup、smoke 更新误差检查。
- `results/correctness.json`：真实 256 样本 MNIST 金标准误差。
- `results/smoke/runs/*`：smoke metrics、config、最终模型/norm/factor state。
- `results/summary.csv`、`method_summary.csv`、`report.md`：最后一次分析输出；运行正式分析时会更新这三个 Exp21 汇总文件。

## 五条路径

| method | norm | aggregate | backward 数 |
|---|---|---|---:|
| exact | Exp19 风格 Opacus explicit per-example gradient，逐样本 A transform | clip 后求和 | 1 |
| fast2 | 各层临时形成 per-example preconditioned gradient | weighted backward，再 aggregate A transform | 2 |
| ghost2 | tiled spatial/sequence Gram identity | weighted backward，再 aggregate A transform | 2 |
| bk | 各层 auto Ghost/Fast | 缓存直接 BK reconstruction | 1 |
| bk_gd | 同 bk，参数在第一遍 requires_grad=False | 同 bk | 1 |

`exact` 是当前带 bias 的 Linear/Conv2d A-only 显式基线，服务正式 SimpleCNN 对照。Transformer primitives 直接使用 `BookKeeping`，以独立逐样本 backward 为 oracle，并测试两遍路径；非 Linear/Conv 的 norm/Embedding 参数在这些 primitive tests 中使用 identity geometry。Exp20 synthetic A builder 的正式适用范围仍为 SimpleCNN；此任务未引入完整 Transformer curvature builder 或训练实验。

Linear 将 `[B,...,d]` 展平到 `[B,T,d]`；Conv2d 使用 unfold。带 bias 时拼接 1，然后 `Z=A_aug @ P`（P 包含 Exp20 scale matching）。Fast norm 临时构造 `B.transpose(1,2) @ Z`，norm 后释放。Ghost 通过 64×64 的 position tiles 累加 `(B_j B_k^T) * (Z_j Z_k^T)`，不形成完整 `[B,T,T]` 或 per-example parameter gradient。

Auto 判据为 `2*T*T <= d_in_aug*d_out`；SimpleCNN：conv1/conv2 选 Fast，fc1/fc2 选 Ghost。最终梯度直接 `sum_i c_i B_i^T Z_i`；BK 不再执行 `G @ P`。clipping 因子沿用 `min(1,1/(norm+1e-6))`。噪声在 aggregate 后加入，随后除以 batch size，再 SGD。

Embedding 将每个样本的 token id 与 backprop 构造 sparse COO，并 coalesce（内部按 token 累加），因此重复 token 的 cross term 正确。BK 最终使用 index_add_ 写入一个普通 `[V,d]` aggregate；不构造 one-hot 或 `[B,V,d]`。支持 padding_idx；明确拒绝 max_norm 和 scale_grad_by_freq。LayerNorm/RMSNorm 直接对 position 求和得到小型 per-example affine gradients，用 Fast norm 和 BK sum。

### GD 与 shared parameters

GD 在 forward 前关闭所有 trainable parameter 的 requires_grad。浮点输入使用新的 gradient anchor；integer token id 不求导，Embedding 的输出设置为 activation anchor。输出 tensor 的 backward hook 保存 backprop；用缓存重构参数梯度后恢复参数 requires_grad。测试通过参数级 hooks 验证第一遍没有普通参数 gradient 计算，而不是事后清除 `.grad`。

共享参数按 Parameter identity 检测，包括同一模块重复调用。涉及共享的 record 按样本合并所有贡献再计算 norm，包含 cross term；Embedding 贡献保持 sparse，必要时与该样本的 dense projection gradient 合并，不创建 batch×vocab dense 缓存。

第一版 shared/tied 多模块参数要求 `operator=None`（identity geometry）。带不同 augmented A operator 的共享参数没有唯一的参数空间变换，显式拒绝，不静默选择某个 module 的 operator。非共享 Linear/Conv 测试覆盖非 identity A，且正式 MNIST 使用真实 scale-matched A。此范围限制不会触发或影响 SimpleCNN。

### 显式 adapters 与 unsupported

标准 nn.Linear、nn.Conv2d（groups=1、zero padding）、nn.Embedding、nn.LayerNorm、nn.RMSNorm 直接支持。其他 RMSNorm-like 和 HF Conv1D 使用显式注册，无名称猜测或静默 fallback：

```python
from transformers.pytorch_utils import Conv1D
from exp21.handlers import register_conv1d, register_rmsnorm
register_conv1d(Conv1D)                 # weight: [input, output]
register_rmsnorm(MyRMSNorm, 'eps')     # x * rsqrt(mean(x²)+eps) * weight
```

测试包含实际安装的 HuggingFace Conv1D。RMS adapter 约定 FP32 公式如上，不涵盖额外 offset、gating 或特殊 mixed precision 变体。两遍对照针对确定性、样本独立模型；没有实现 dropout RNG replay、BatchNorm 或自定义跨样本算子。

没有注册真正的 FGC fallback module：所有支持的层均可 BK reconstruction；不支持的 trainable module/parameter 明确报错。结果 `fallback_layers=0`、`fallback_layer_count=0`、`fallback_layer_names=[]`；BK 的 `requires_second_backward=false`。Fast norm 的层不是 fallback。fast2/ghost2 的 `requires_second_backward=true` 表示其设计本身需要第二遍。

## 验证和计时

先执行测试，再 smoke。Primitive FP32 norm/clip/gradient 使用 rtol=1e-4；真实 CNN 因 batch convolution / Gram 求和顺序不同使用 rtol=5e-4、gradient atol=3e-5、norm atol=1e-4。真实 MNIST reference 为 256 次独立 sample backward，不依赖 hooks 的梯度公式。结果中 relative L2 error 约 4e-7～8e-7，详见 `correctness.json`。

测试还覆盖强制 Ghost/Fast、重复 token、共享/tied/reused module、无噪声 SGD、GD 参数 hook 不触发、弱引用缓存释放、错误后恢复、Ghost 禁止调用 per-example materializer、6 个连续 full-size CNN steps 的 allocated 显存稳定、RNG 隔离和计时区间不主动 synchronize。

CUDA events 在 phase 结束后同步并解析；timed context 内无 synchronize。`first_pass_seconds` 不含独立 `norm_seconds`；BK 的 `second_pass_seconds` 精确为 0。timing 分解是每 epoch 的累计值；`seconds_per_batch` 来自完整 private train wall time，包含数据加载/传输和 Python 开销，因此不等于 event 分解之和除以 batches。

`peak_cuda_*_bytes` 是 private training phase 的 allocator peaks；build peak 另列。reserved 包含同进程 warmup 的 allocator cache。`bk_cache_bytes` 是缓存 tensor payload 总和（view 可能共享 storage），`temporary_per_sample_grad_bytes` 是 Fast/explicit/shared 路径单阶段 per-sample gradient payload 高水位，非 allocator peak；不包含 Gram workspace。Fast norm 的 aggregate reconstruction workspace 与 retained ordinary gradients 已体现在 CUDA allocated peak 中。

输出每层策略、缓存释放状态及每个 batch 结束 allocated bytes。正式保存 norm/clip/loss diagnostics 每 batch 约增长 2.5 KiB，属于有界的 epoch diagnostics；六步不保存 diagnostics 的独立测试验证 BK 缓存不增长。accuracy 仅作 sanity check；3-batch smoke 的时间和 speedup 不作正式性能或 utility 结论。
