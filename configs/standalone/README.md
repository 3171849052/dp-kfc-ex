# MNIST standalone 训练

三种算法都通过 `scripts/train.py` 训练，`training.optimizer` 支持 `sgd` 和 `adamw`。
新增 AdamW 配置：

- `mnist_dp_sgd_adamw.yaml`：逐样本梯度裁剪、加噪后使用 AdamW。
- `mnist_dp_kfc_adamw.yaml`：KFC 预条件、逐样本裁剪、加噪后使用 AdamW。
- `mnist_dp_equil_adamw.yaml`：Equil 预条件、逐样本裁剪、加噪后使用 AdamW。

`algorithm: dp_sgd` 在这里表示不使用预条件的基线；实际更新器由
`training.optimizer: adamw` 指定。AdamW 的一、二阶矩只接收裁剪并加噪后的聚合梯度。

三份配置均以 `learning_rate: 0.001`、`betas: [0.9, 0.999]`、
`eps: 1.0e-8`、`weight_decay: 0.01` 为起点，尚未调参。
`momentum` 仅用于 SGD。其他参数复制自对应现有配置，包括当前 Equil 的
`tau: 0.7` 和缩放范围 `[1.0e-8, 1.0e+8]`。

在仓库根目录通过现有 tmux 启动器运行（需要 `curve` conda 环境）：

```bash
bash run.sh configs/standalone/mnist_dp_sgd_adamw.yaml
bash run.sh configs/standalone/mnist_dp_kfc_adamw.yaml
bash run.sh configs/standalone/mnist_dp_equil_adamw.yaml
```

以上每条命令都会启动独立训练；默认均使用 GPU 0，可按需修改 `runtime.gpu`。
也可以在安装好依赖的环境中直接前台运行，无需 tmux：

```bash
CONFIG=configs/standalone/mnist_dp_sgd_adamw.yaml
RUN_DIR=$(python scripts/train.py --config "$CONFIG" --prepare-run)
set -o pipefail
python -u scripts/train.py --config "$CONFIG" --run-dir "$RUN_DIR" 2>&1 | tee -a "$RUN_DIR/train.log"
```

输出目录包含 `adamw`、betas、eps 和 weight decay；完整参数保存在
`resolved_config.yaml`，`summary.json` 记录实际优化器。
