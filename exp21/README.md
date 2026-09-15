# Exp21 — Transformer Hybrid BK / Ghost Differentiation / FGC fallback

在原 Exp21 CNN BK 数学核心上增量修复。所有修改和输出均在 `exp21/`；Exp19/20 只读复用，不修改历史结果。此次只运行测试与 smoke，没有启动正式 25-job 实验；已有 `results/runs/` 数据不作改动。

## 运行与固定协议

从仓库根目录运行：

```bash
conda run -n curve python -B exp21/test_exp21.py
conda run -n curve bash exp21/run_all.sh --smoke
conda run -n curve python -B exp21/analyze.py --smoke
# 正式实验命令（本次不执行）
conda run -n curve bash exp21/run_all.sh
```

正式 MNIST / 当前 SimpleCNN 协议不变：batch=256，5 epochs，seeds=(42,7,123,2024,3407)，power=.25，damping=.001，C=1，epsilon=1，delta=1e-5；SGD lr=.5、momentum=0、weight_decay=0。每 epoch 使用 10×256 个 pink-noise synthetic samples，只 forward 构建 A。五方法为 exact、fast2、ghost2、bk、bk_gd，每个 method/seed fresh subprocess，并轮换 method 顺序。

初始化为 CPU seed；shuffle 用独立 generator(seed)；噪声 generator(seed+40000)；synthetic 在当前 CUDA device 的 fork_rng(seed+10000+epoch) 中生成。沿用 Exp20 shuffled fixed-batch RDP accounting 约定，总计 1170 private steps；不提出新的 privacy proof。测试确认有 bias 的 SimpleCNN operator 与 Exp20 逐元素完全相同。

Smoke：seed=42，前 768 个 MNIST 训练样本 shuffle 后 3 batches，1 epoch，1×256 synthetic samples，noise=0，前 256 个测试样本作 accuracy sanity check。读取已有 `exp1/data`，不下载。每方法独立的 disposable warmup 保持 RNG 隔离。短 smoke 仅用于诊断，不支持正式速度或 utility 结论。

## 路径与接口

```python
from exp21.methods import HybridBKClipper, Clipper, build_from_cache

operator, stats = build_from_cache(model, .25, calibration_batches, seed=42, epoch=1,
                                  layer_names=selected_affine_names)  # 可省略 layer_names
clipper = HybridBKClipper(model, operator, method='bk_gd', max_grad_norm=.37)
loss_sum, norms, factors, layer_squares, stats = clipper.aggregate(x, target,
                                                               loss_fn=per_example_loss)
clipper.step(optimizer, sigma, len(x), noise_generator)
clipper.remove()
```

输入是带 batch 维的 tensor；`loss_fn(model(x), target)` 必须返回 `[B]`，默认 cross entropy。HF 模型可以直接接受 token ids，并在 loss_fn 中读取 output.logits。当前执行范围为 FP32、样本独立的模型；不支持跨样本 loss/算子、gradient checkpointing、任意输入 pytree 或 autocast。

`max_grad_norm` 必须为正。所有路径使用 `min(1,C/(norm+1e-6))`；`clipper.step` 使用同一个 C 生成标准差 `sigma*C` 的噪声，然后除以 batch size 再 SGD，避免 clipping/noise 的 C 分离。

| 请求路径 | 无 fallback 时 norm | aggregate | reverse 调用 |
|---|---|---|---:|
| exact | Opacus explicit per-example gradient + A transform | clip 后求和 | 1 |
| fast2 | 临时 layer-wise per-example preconditioned gradient | weighted backward 后 A transform | 2 |
| ghost2 | tiled Gram；norm affine/Embedding 使用专用局部算法 | weighted backward 后 A transform | 2 |
| bk | hybrid Ghost/Fast | 直接 BK reconstruction | 1 |
| bk_gd | 同 BK，使用 output anchors | 直接 BK reconstruction | 1 |

exact 保留 Opacus baseline，服务正式 SimpleCNN；直接 BK 支持范围不等于 Opacus sampler 范围。遇到明确注册的 fallback，**所有请求方法（包括 exact/bk_gd）都转为 whole-step FGC**，metadata 反映真实执行。两遍路径 replay 首遍 CPU/当前 CUDA RNG，使 dropout 等随机 forward 与第一遍匹配，并恢复外部 RNG 进度。

## 修正后的 A-only geometry

`geometry.py` 复用 Exp20 AOperator 的 damped matrix function 和 scale matching，修正 activation covariance：

- 有 bias：使用 `[a,1]`，P 为 `(d+1)×(d+1)`。
- 无 bias：仅使用 a，P 为 `d×d`，不伪造 coordinate。
- Linear 的 `[B,...,d]` 展平所有 batch/position 行后构建 A。
- Conv2d 使用 unfold，维度为 `in_channels*kernel_h*kernel_w`，按有无 bias 决定 augmentation。
- 实际 HF `Conv1D` 按 `[input,output]` weight layout 构建 input covariance，output dimension 用 weight.shape[1]。

operator 可以只覆盖一部分 affine 层。只有名字出现在 operator.data 的层使用 A transform，其余层使用 identity，**全部 trainable parameters** 都纳入 global clipping norm。输出 `preconditioned_layers` 和 `identity_geometry_layers`。

默认 builder 选择非共享的 Linear/Conv2d/HF Conv1D。共享参数所在模块默认不构建 A，并在 `builder_identity_shared_layers` 中明确记录；可用 layer_names 指定更小的覆盖集。Embedding/LayerNorm/RMSNorm 使用 identity geometry。校准没有实际执行的指定层明确报错；packed MHA 的 out_proj 由 functional 运算调用，不能作为普通 affine forward hook 校准，TinyViT 测试显式只对 patch_embed/head 建 A。

对于共享参数，不定义互相冲突的 augmented module operators：共享组件要求 identity geometry，**其他层仍可非 identity A**。显式把共享组件加入 operator.data 会报错。预条件层内部部分冻结也明确报错；未覆盖的普通 BK 层允许 affine 参数冻结子集。

## 直接 BK 支持与缓存

直接支持 nn.Linear、nn.Conv2d（groups=1、zero padding）、nn.Embedding、nn.LayerNorm、nn.RMSNorm，以及实际 HF Conv1D/LlamaRMSNorm。其他兼容类型通过显式注册，不按名称猜测：

```python
from exp21.handlers import register_conv1d, register_rmsnorm
register_conv1d(MyConv1D)                # [input,output] 权重
register_rmsnorm(MyRMSNorm, 'eps')      # 标准 x*rsqrt(mean(x²)+eps)*weight
```

Linear/Conv：`Z=A_aug @ P`（P 已含 scale matching）。Fast norm 临时形成 `B^T Z` 并释放；Ghost norm 通过 64×64 position tiles 累加 Gram identity，不构造完整 `[B,T,T]` 或 per-example parameter gradient。Auto 判据为 `2*T² <= d_in_aug*d_out`。SimpleCNN：两层 Conv 选 Fast，两层 Linear 选 Ghost。

最终 BK aggregate 为 `sum_i c_i B_i^T Z_i`，已是预条件后的结果，不再次乘 P。形成 z 后 Record 只保留 z+b，x=None；Norm 仅保留归一化后的 z+b；Embedding 仅保留 ids+b。记录的 `bk_cache_bytes` 按实际 retained storage 去重，包含必要 view 的底层 storage，不重复计数 broadcast views。每步结束清空全部 records/pending；测试验证 raw activation 不再长期持有和连续六个完整 CNN batch 无缓存显存增长。

Embedding 每个样本按 token id coalesce sparse COO，正确累加重复 token 与 padding_idx；不生成 one-hot 或 `[B,V,d]`。最终只向普通 `[V,d]` aggregate 做 index_add。LayerNorm/RMSNorm 对 position 求和得到小型 affine per-example gradients，用 Fast local norm 和 BK aggregate。

### 高效 tied Embedding / LM head

支持同一 Parameter 对象在一次 Embedding 调用与一次 Linear output projection 调用之间共享；head bias 可以独立训练。head 始终使用 tiled Ghost norm，即使请求 forced Fast，也以 `ghost_tied` 明确记录，避免形成 batch×vocab×hidden。

norm 使用：

```
||G_embed + G_head||²
= ||G_embed||² + ||G_head||² + 2 <G_embed, G_head>
```

Embedding 用 sparse token rows；只计算 head 中这些 token 的 rows 与 sparse values 的内积，不形成单样本完整 `[V,d]` dense gradient。最终两路各自 BK aggregate 后加到同一 Parameter。对实际 GPT-2/LLaMA tied LM head 的完整模型测试已覆盖。

HF Conv1D 的 Parameter layout 是 `[input,output]`，与 Embedding 的 `[vocab,hidden]` 直接 identity tie 通常不兼容。实现支持实际形状兼容的同对象 HF tie（测试为 square layout），不将 transpose-view 伪装成 Parameter identity。常见语言模型的 lm_head 是 Linear。

保留原 generic shared/reused Linear tests 的逐样本合并 reference 路径，包含全部 cross terms，适合小型共享层。超过一次 embedding/head 调用或多 owner 的 tied Embedding pattern 明确拒绝，需将相应外层模块显式注册为 whole-step FGC；不会悄悄进入大 vocab 逐样本 dense BK 路径。

## 真正的 output-anchor Ghost Differentiation

GD 在 forward 前关闭真实 trainable parameters 的 requires_grad，并 detach 原始输入。forward hook 遇到没有 grad graph 的 supported output 时，将其设为 requires_grad=True；从该 output 得到 backprop，供该层 BK reconstruction 使用。

因此第一层 Conv/Linear 不反传无用 input gradient；token ids 永不求导。多个独立分支各自有 output anchor，例如 GPT-2 word/position embeddings。`gd_anchor_module` 为首个 anchor，`gd_anchor_modules` 列出全部，`gd_applied` 表示实际是否执行 GD。

HF BERT/GPT-2 的 position embeddings 原本以 `[1,T]`/`[T]` 输入后 broadcast。通过按实际 HF 类型识别的 adapter，在相加前扩展 embedding output 到 batch 维，并在这个 output 捕获逐样本 backprop；否则会错误地提前合并样本。没有名称猜测。测试验证参数 hooks 首遍完全不触发、first_pass_parameter_grad_count=0、原始 x.grad 不产生，而最终重构梯度与 oracle 一致。

## Whole-step Fast-Clipping fallback

`routing.py` 初始化扫描每个 trainable Parameter，并分为：

1. BK-supported：上述 adapters 和已有小型 generic shared 路径。
2. FGC-fallback：明确注册的 sample-independent 模块。
3. unsupported-dangerous / unsupported：明确报错。

内置 fallback：nn.MultiheadAttention 的全部 packed/functional 参数；仓库 TinyViT 的直接 Parameter（pos_embed）。TinyViT 的 patch_embed、norm、FFN 等仍会被扫描，但只要存在一个 fallback，整个 step 转为标准两遍路径：

1. 一次 batched forward + batched VJP (`autograd.grad(..., is_grads_batched=True)`) 得到所有参数的 per-example gradients；应用 partial A，得到 global norm/clip；释放 per-example gradients。
2. RNG replay 后 weighted forward/backward，再执行 aggregate A transform。

这个版本的 FGC 保留完整 per-example 参数梯度，内存 O(B*参数量)，是 packed/custom module 的正确性 fallback，不用于替代大 vocab 的直接 BK 路径。reverse 调用实际为一次 batched VJP + 一次 backward，`backward_calls=2`；reverse vectors 数为 B。fallback 不启用 GD（gd_applied=False）；首遍 autograd.grad 不填充普通 p.grad，但这不表示执行了 GD。

扩展需显式注册，并保证 sample independence 和无 forward state mutation：

```python
from exp21.routing import register_fallback
register_fallback(MyPackedModule)  # 整个 subtree 的参数
register_fallback(MyModel, direct_parameters_only=True)  # 仅外层直接参数，子模块继续扫描
```

BatchNorm、Embedding max_norm/scale_grad_by_freq 明确拒绝；未知 trainable module 或额外参数不静默忽略。fallback 使用 PyTorch math attention，不自动尝试其他 backend。未实际贡献梯度的 trainable 参数也明确报错。

metadata 包含 fallback_layer_names/count、fallback_parameter_names、requires_second_backward、backward_calls、第一遍参数 grad 数、layer norm 策略、partial geometry、anchors、cache/temporary bytes。Fast local norm 不是 fallback。SimpleCNN 的 BK/GD 仍然 fallback=0、second backward=false、second_pass_seconds=0。

## 验证结果与文件

测试入口包括原测试和 `test_transformer_cases.py`，当前 **117 项通过**。原容差不降低：primitive norm/clip/gradient rtol=1e-4；真实 256 样本 MNIST 的 convolution/Gram 求和路径 rtol=5e-4、gradient atol=3e-5。oracle 继续是独立逐样本 backward。

新增覆盖：bias=False Linear/Conv + 非 identity A；实际 HF Conv1D + A；partial A；五方法 C=.37 + 有噪声 SGD；output anchors/input grad 缺失；raw cache 释放；解析 tied cross-term 禁止 dense per-example head materialization；非 identity partial A 的 tiny BERT/GPT-2/LLaMA；identity/partial A 的 TinyViT FGC；危险模块拒绝。tiny HF 仅通过 config 随机初始化，offline 环境，无 pretrained 下载；dropout=0，短序列、小 batch、eager attention。

BERT/GPT-2/LLaMA 的 bk/bk_gd integration 均 **直接 BK、无 fallback、单 reverse**；TinyViT 明确 **whole-step FGC、两次 reverse**，覆盖 pos_embed 和全部 MHA 参数。结果 JSON 写入 `results/integration/`。

主要文件：

- `geometry.py`：修正 A builder；`handlers.py`：层公式与最小缓存。
- `routing.py` / `fallback.py`：完整参数扫描与真正两遍 fallback。
- `bk.py` / `methods.py`：hybrid norm、GD、tied cross-term、统一 clipping/noise C。
- `profiling.py` / `run_one.py` / `run_sweep.py` / `run_all.sh`：原协议与计时。
- `test_exp21.py` / `test_transformer_cases.py`、`pytest.log`：验证。
- `smoke.log`、`results/smoke/`：fresh-process smoke。
- `results/correctness.json`：真实 MNIST 对 naive oracle 的误差。
- `results/summary.csv`、`method_summary.csv`、`report.md`：最新分析。

CUDA events 在 phase 边界同步后解析，timed region 内不主动 synchronize。first pass、norm、second pass、BK reconstruction、aggregate transform、noise step 分开记录；CSV 中 phase seconds 为 epoch 累计，report 转为每 batch。private wall time 包含数据加载/传输和 Python 开销；不通过改变计时范围强制制造 GD 加速。report 给出 GD 相对 BK 的首遍/总时间变化、norm/重构差异与未归入 CUDA phases 的 wall residual；后者不是纯 CPU 时间。

allocated peak 是 private training phase 的峰值；build peak 另列；reserved 含 disposable warmup 的 allocator cache。temporary_per_sample_grad_bytes 表示梯度 payload 高水位，非 allocator peak，不含 Gram workspace。norm/clip/loss diagnostics 每 batch 保存约 2.5 KiB，epoch 结束释放，与 BK cache 泄漏区分。
