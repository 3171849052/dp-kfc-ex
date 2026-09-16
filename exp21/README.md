# Exp21 — Transformer Hybrid BK / Ghost Differentiation / FGC fallback

在原 Exp21 CNN BK 数学核心上增量修复。所有修改和输出均在 `exp21/`；Exp19/20 只读复用，不修改历史结果。测试与 smoke 完成后，按用户追加指令将完整 25-job 实验提交后台启动，不监视。新结果根目录为 `results/formal_transformer_bk_v2/`，已有 `results/runs/` 数据不作改动。日志为 `formal_transformer_bk_v2.log`，启动 PID 写入 `formal_transformer_bk_v2.pid`。

## 运行与固定协议

从仓库根目录运行：

```bash
conda run -n curve python -B exp21/test_exp21.py
conda run -n curve bash exp21/run_all.sh --smoke
conda run -n curve python -B exp21/analyze.py --smoke
# 正式实验：显式指定一个尚不存在的新目录，避免覆盖旧结果
conda run -n curve bash exp21/run_all.sh --output exp21/results/formal_transformer_bk_v2
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

exact 保留 Opacus baseline，服务正式 SimpleCNN；直接 BK 支持范围不等于 Opacus sampler 范围。遇到明确注册的 fallback，bk/bk_gd 保留支持层的 BK norm/reconstruction，仅 fallback 参数使用分块 VJP 与 weighted gradient。exact 遇到 fallback 时使用两遍 Fast engine。两遍路径 replay 首遍 CPU/当前 CUDA RNG，使 dropout 等随机 forward 与第一遍匹配，并恢复外部 RNG 进度。

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

Linear/Conv：`Z=A_aug @ P`（P 已含 scale matching）。Fast norm 临时形成 `B^T Z`，直接对 strided gradient view 做 vector_norm 后释放，避免 flatten 复制和完整 square 临时张量；Ghost norm 通过单轴 row tiles（默认 64 行、覆盖全部 T 列）累加 Gram identity，Python 循环数 O(T/tile)，临时 Gram O(B*tile*T)。tile≥T 时限制行数，避免完整 `[B,T,T]`。Auto 先按真实 augmented gradient 大小执行 256 MiB Fast hard cap，再比较 MACs、梯度归约读取与 launch 开销；详细估算和 routing_reason 写入 layer_routing。SimpleCNN：两层 Conv 选 Fast，两层 Linear 选 Ghost。

最终 BK aggregate 为 `sum_i c_i B_i^T Z_i`，已是预条件后的结果，不再次乘 P。形成 z 后 Record 只保留 z+b，x=None；Norm 仅保留归一化后的 z+b；Embedding 仅保留 ids+b。记录的 `bk_cache_bytes` 按实际 retained storage 去重，包含必要 view 的底层 storage，不重复计数 broadcast views。每步结束清空全部 records/pending；测试验证 raw activation 不再长期持有和连续六个完整 CNN batch 无缓存显存增长。

Embedding 将 `(sample,token)` 编成 composite key，以一次 unique/index_add 合并全 batch 的重复 rows，并过滤 padding_idx；没有 Python per-sample sparse loop，不生成 one-hot 或 `[B,V,d]`。最终只向普通 `[V,d]` aggregate 做 index_add。LayerNorm/RMSNorm 对 position 求和得到小型 affine per-example gradients，用 Fast local norm 和 BK aggregate。

### 高效 tied Embedding / LM head

支持同一 Parameter 对象在一次 Embedding 调用与一次 Linear output projection 调用之间共享；head bias 可以独立训练。head 在 `ghost_tied` 与 `chunked_fast_tied` 之间选择。后者按 output/vocab 维切块（默认 256，受 Fast workspace cap 限制），每块梯度立即归约释放，绝不形成完整 `[B,V,d]` 或逐样本 `[V,d]`。考虑分块开销，以 `1.25*B*T*V*d < B*T²*(V+d)` 选择 chunked Fast；无可容纳的 output row 时使用 Ghost。

norm 使用：

```
||G_embed + G_head||²
= ||G_embed||² + ||G_head||² + 2 <G_embed, G_head>
```

Embedding 用 sparse token rows；只计算 head 中这些 token 的 rows 与 sparse values 的内积，不形成单样本完整 `[V,d]` dense gradient。最终两路各自 BK aggregate 后加到同一 Parameter。对实际 GPT-2/LLaMA tied LM head 的完整模型测试已覆盖。

HF Conv1D 的 Parameter layout 是 `[input,output]`，与 Embedding 的 `[vocab,hidden]` 直接 identity tie 通常不兼容。实现支持实际形状兼容的同对象 HF tie（测试为 square layout），不将 transpose-view 伪装成 Parameter identity。常见语言模型的 lm_head 是 Linear。

保留原 generic shared/reused Linear tests 的逐样本合并 reference 路径，包含全部 cross terms，适合小型共享层。generic shared 合并后的单样本 dense 参数量超过 max_shared_sample_bytes（默认 64 MiB）时，将共享组件路由到 bounded Fast fallback。同一 module 重复调用在 forward 后、任何 sample gradient 分配前检测；必要时 RNG replay 重建带 fallback 参数的图，该开销包含在 first-pass timer 内。未被 guard 路由的多 owner Embedding pattern 仍需显式注册为 bounded Fast fallback；不会悄悄进入大 vocab 逐样本 dense BK 路径。

## 真正的 output-anchor Ghost Differentiation

GD 在 forward 前关闭真实 trainable parameters 的 requires_grad，并 detach 原始输入。forward hook 遇到没有 grad graph 的 supported output 时，将其设为 requires_grad=True；从该 output 得到 backprop，供该层 BK reconstruction 使用。

因此第一层 Conv/Linear 不反传无用 input gradient；token ids 永不求导。多个独立分支各自有 output anchor，例如 GPT-2 word/position embeddings。`gd_anchor_module` 为首个 anchor，`gd_anchor_modules` 列出全部，`gd_applied` 仅标记正式 bk_gd 方法。Fast2/Ghost2 首遍也禁用参数梯度，使用 output anchors，实际状态由 first_pass_param_grad_disabled 表示。非共享层在 backward hook 内归约并释放；共享/tied cross terms 在首遍结束前完成并清空 records。

HF BERT/GPT-2 的 position embeddings 原本以 `[1,T]`/`[T]` 输入后 broadcast。通过按实际 HF 类型识别的 adapter，在相加前扩展 embedding output 到 batch 维，并在这个 output 捕获逐样本 backprop；否则会错误地提前合并样本。没有名称猜测。测试验证参数 hooks 首遍完全不触发、first_pass_parameter_grad_count=0、原始 x.grad 不产生，而最终重构梯度与 oracle 一致。

## Hybrid BK + bounded local Fast fallback

`routing.py` 初始化扫描每个 trainable Parameter，并分为：

1. BK-supported：上述 adapters 和已有小型 generic shared 路径。
2. FGC-fallback：明确注册的 sample-independent 模块。
3. unsupported-dangerous / unsupported：明确报错。

内置 fallback：nn.MultiheadAttention 的 packed/functional 参数与 TinyViT 的直接 Parameter（pos_embed）。patch_embed、norm、FFN、head 继续使用 BK。

1. 一次 sum-loss reverse 捕获支持层 backprop，autograd.grad 不填充参数 `.grad`。
2. 每次最多 K（默认 2）个 example 的 batched VJP，targets **仅 fallback 参数**；立即归约并释放梯度。加上 BK norm² 后得到同一个全局 clip factor。
3. 支持层直接 BK reconstruction；RNG replay 后 `autograd.grad(weighted_loss, fallback_params)` 只取得 fallback aggregate，再对相关 A 层变换。

fallback 临时参数梯度为 O(K*M_fallback)，不含 forward graph/activation workspace。metadata 的 fallback_parameter_count 是参数元素数；fallback_temporary_grad_bytes 是实际最大 VJP gradient tensor bytes。reverse 调用数为 `2+ceil(B/K)`，逐次如实计数。TinyViT B=2、K=2 时为 3 次；不会为支持层生成 batch×parameter 梯度。

API 直接提供 `ghost_tile=64, max_fast_temp_bytes=256*2**20, fallback_vjp_chunk_size=2, max_shared_sample_bytes=64*2**20, tied_output_chunk_size=256`，没有配置对象。

扩展需显式注册，并保证 sample independence 和无 forward state mutation：

```python
from exp21.routing import register_fallback
register_fallback(MyPackedModule)  # 整个 subtree 的参数
register_fallback(MyModel, direct_parameters_only=True)  # 仅外层直接参数，子模块继续扫描
```

BatchNorm、Embedding max_norm/scale_grad_by_freq 明确拒绝；未知 trainable module 或额外参数不静默忽略。fallback 使用 PyTorch math attention，不自动尝试其他 backend。未实际贡献梯度的 trainable 参数也明确报错。

metadata 包含 fallback_layer_names/count、fallback_parameter_names、requires_second_backward、backward_calls、第一遍参数 grad 数、layer norm 策略、partial geometry、anchors、cache/temporary bytes。Fast local norm 不是 fallback。SimpleCNN 的 BK/GD 仍然 fallback=0、second backward=false、second_pass_seconds=0。

## 验证结果与文件

测试入口包括原测试和 `test_transformer_cases.py`，当前 **147 项通过**（含 `test_optimized_cases.py`）。原容差不降低：primitive norm/clip/gradient rtol=1e-4；真实 256 样本 MNIST 的 convolution/Gram 求和路径 rtol=5e-4、gradient atol=3e-5。oracle 继续是独立逐样本 backward。

新增覆盖：bias=False Linear/Conv + 非 identity A；实际 HF Conv1D + A；partial A；五方法 C=.37 + 有噪声 SGD；output anchors/input grad 缺失；raw cache 释放；解析 tied cross-term 禁止 dense per-example head materialization；非 identity partial A 的 tiny BERT/GPT-2/LLaMA；identity/partial A 的 TinyViT FGC；危险模块拒绝。tiny HF 仅通过 config 随机初始化，offline 环境，无 pretrained 下载；dropout=0，短序列、小 batch、eager attention。

BERT/GPT-2/LLaMA 的 bk/bk_gd integration 均 **直接 BK、无 fallback、单 reverse**；TinyViT 使用 **Hybrid BK + bounded local fallback**，覆盖 pos_embed 和全部 MHA 参数。结果 JSON 写入 `results/integration/`。

主要文件：

- `geometry.py`：修正 A builder；`handlers.py`：层公式与最小缓存。
- `routing.py` / `fallback.py`：完整参数扫描与真正两遍 fallback。
- `bk.py` / `methods.py`：hybrid norm、GD、tied cross-term、统一 clipping/noise C。
- `profiling.py` / `run_one.py` / `run_sweep.py` / `run_all.sh`：原协议与计时。
- `test_exp21.py` / `test_transformer_cases.py`、`pytest.log`：验证。
- `smoke.log`、`results/smoke/`：fresh-process smoke。
- `results/correctness.json`：真实 MNIST 对 naive oracle 的误差。
- `results/summary.csv`、`method_summary.csv`、`report.md`：最新分析。

CUDA events 在 phase 边界同步后解析，timed region 内不主动 synchronize。first pass、norm、second pass、BK reconstruction、aggregate transform、noise step 分开记录。Fast2/Ghost2 的 fused norm 在 backward hook 内执行，因此计入 first-pass；其 norm phase 只完成 clip factor 等收尾，不可把较小 norm_seconds 解释成单独 norm kernel 的加速。CSV 中 phase seconds 为 epoch 累计，report 转为每 batch。private wall time 包含数据加载/传输和 Python 开销；不通过改变计时范围强制制造 GD 加速。report 给出 GD 相对 BK 的首遍/总时间变化、norm/重构差异与未归入 CUDA phases 的 wall residual；后者不是纯 CPU 时间。

allocated peak 是 private training phase 的峰值；build peak 另列；reserved 含 disposable warmup 的 allocator cache。temporary_per_sample_grad_bytes 表示梯度 payload 高水位，非 allocator peak，不含 Gram workspace。norm/clip/loss diagnostics 每 batch 保存约 2.5 KiB，epoch 结束释放，与 BK cache 泄漏区分。

局部 router benchmark：`conda run -n curve python -B exp21/benchmark_router.py`，输出 `results/router_benchmark.csv`。默认 sweep 72 个形状，同时比较旧双 tile、新 row tile、Fast。当前 router 在原始 MACs 上加入简单等效成本：Fast 加 8 个等效 token 的 sample-matrix 归约成本；Ghost skinny GEMM 使用系数 8；每个 launch group 加 4.4e8 等效 MACs。常数按 RTX 3080 Ti 的局部 sweep 手工校准，换硬件应重新验证；metadata 同时保留原始 MACs。不会自动拟合路由表或启动训练。
