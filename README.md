# DP-KFC: Data-Free Preconditioning for Privacy-Preserving Deep Learning

<div align="center">

[**🌐 Project page**](https://molinamarcvdb.github.io/DP-KFC/) &nbsp;·&nbsp;
[**📄 Paper**](https://molinamarcvdb.github.io/DP-KFC/static/dp_kfc.pdf) &nbsp;·&nbsp;
[📚 arXiv](#) &nbsp;·&nbsp;
[❝ BibTeX](#citation)

*Accepted at the International Conference on Machine Learning (ICML), 2026.*

</div>

---

Differentially private SGD injects **isotropic** noise into networks whose loss landscape is wildly **anisotropic**. Second-order preconditioners like KFAC can fix this geometric mismatch, but estimating curvature has traditionally cost either privacy budget (estimating from the private data) or a public proxy (which may not exist for your domain).

**DP-KFC** sidesteps both. We show the KFAC Fisher block decomposes into *architectural sensitivity* (recoverable from synthetic noise) and *input correlations* (approximable from modality-specific frequency statistics), so the preconditioner can be built **with no real data and no privacy cost**. Empirically it matches public-data preconditioning on vision, improves over DP-SGD and adaptive baselines across modalities, and strictly dominates public proxies under domain shift.

> 👉 **The [project page](https://molinamarcvdb.github.io/DP-KFC/) has the walk-through, all figures, and the headline numbers.** This README focuses on running the code.

## Install

```bash
git clone https://github.com/molinamarcvdb/DP-KFC.git
cd DP-KFC

uv sync                  # core
uv sync --extra nlp      # + transformers, datasets
uv sync --extra medical  # + MedMNIST
```

Requires Python ≥ 3.13, PyTorch ≥ 2.9.1, [Opacus](https://github.com/pytorch/opacus) ≥ 1.5.4.

## Quick start

```bash
uv run scripts/paper/exp_cnn_mnist.py --fast              # 30-second smoke test
uv run scripts/paper/exp_cnn_mnist.py --seed 42 --epsilon 1.0
```

Every paper experiment script accepts `--fast`, `--seed`, `--epsilon`.

## Reproducing the paper

Scripts live in `scripts/paper/`. They write CSVs to `results/`; the `visualize/` helpers turn those into the paper figures and LaTeX tables.

```bash
# Main benchmarks
uv run scripts/paper/exp_cnn_mnist.py            # MNIST       / CNN
uv run scripts/paper/exp_crossvit_cifar100.py    # CIFAR-100   / CrossViT
uv run scripts/paper/exp_stackoverflow.py        # StackOverflow / BERT
uv run scripts/paper/exp_imdb_logreg.py          # IMDB        / Logistic regression
uv run scripts/paper/exp_sst2.py                 # SST-2       / DistilBERT

# Ablations
uv run scripts/paper/ablation_fim_spectrum.py        # eigenspectrum alignment (Fig. 2)
uv run scripts/paper/ablation_cov_tracking.py        # covariance tracking through training (Fig. 3)
uv run scripts/paper/ablation_adadps.py              # AdaDPS comparison
uv run scripts/paper/ablation_transfer_alignment.py  # negative-transfer setting (Table 2)

# Figures + LaTeX tables from saved results
uv run scripts/paper/visualize/visualize_vision.py
uv run scripts/paper/visualize/visualize_nlp.py
uv run scripts/paper/visualize/visualize_spectrum.py
uv run scripts/paper/visualize/visualize_cov_tracking.py
uv run scripts/paper/visualize/generate_latex_tables.py
```

The full per-(dataset, ε) accuracy tables are in the paper appendix.

## Repo layout

```
src/dp_kfac/
├── trainer.py        plain / DP-SGD / DP-KFC training loops
├── optimizer.py      DPKFACOptimizer (clip, noise, preconditioner update)
├── covariance.py     KFAC A / G factor estimation
├── precondition.py   per-sample gradient preconditioning
├── privacy.py        clipping + Gaussian mechanism
├── methods.py        method registry (see below)
├── models.py         MLP, CNN, CrossViT, ConvNeXt, BERT / RoBERTa / DistilBERT
├── data.py           dataset loaders (vision + NLP + TF-IDF)
└── analysis.py       eigenvalue spectra, covariance tracking

scripts/paper/        all paper experiments, ablations and figure generation
configs/              YAML experiment configurations
docs/                 the project page (served at the link above)
```

## Methods

| `--method`         | preconditioner source       | needs side data? |
|:-------------------|:----------------------------|:-----------------|
| `dp_sgd`           | —                            | no (baseline)    |
| `dp_kfac_public`   | public-data activations + gradients | **yes** (public proxy) |
| `dp_kfac_pink`     | structured synthetic noise (1/fᵅ)    | **no** ← ours       |
| `dp_kfac_noise`    | white-noise probes           | no               |
| `adadps`           | diagonal E[g²]               | yes (public)     |

## Citation

```bibtex
@inproceedings{molina2026dpkfc,
  title     = {{DP-KFC}: Data-Free Preconditioning for Privacy-Preserving Deep Learning},
  author    = {Molina Van den Bosch, Marc and Taiello, Riccardo and
               Sund Aillet, Albert and Protani, Andrea and
               Gonzalez Ballester, Miguel Angel and Serio, Luigi},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```

## Acknowledgements

Supported by the Innovative Health Initiative Joint Undertaking and its members (grant 101172825) and the CAFEIN® R&D fund, the CERN Quantum Technology Initiative (QTI), the ERC Synergy Grant *Zee-Zoom-Zap* (grant 101224844), and the María de Maeztu Units of Excellence Programme (CEX2021-001195-M, MICIU/AEI/10.13039/501100011033).
# dp-kfc-ex
# dp-kfc-ex

## Standalone training (Exp6)

在 `curve` conda 环境中通过 tmux 后台启动一个独立 YAML 配置：

```bash
./run.sh configs/standalone/mnist_dp_sgd.yaml
./run.sh --config configs/standalone/mnist_dp_kfc.yaml
./run.sh configs/standalone/mnist_dp_equil.yaml
```

首版支持 SimpleCNN + MNIST，默认 5 epochs、batch size 256、GPU 0。
算法复用 Exp6 的 shuffled minibatch protocol、Synthetic DP-KFC 和 Full-Fisher
Equil；预条件发生在全局逐样本裁剪之前，然后加入各向同性 Gaussian noise，
执行 SGD 和 RDP accountant step。保留 Exp6 的 RDP 校准口径（包含最后一个
不足整批的 step），不使用 Poisson sampler。

每次启动在 `outputs/` 创建秒级命名目录，保存原始 `config.yaml`、
`resolved_config.yaml`、每 epoch 一行且 fsync 的 `metrics.csv`、完成后的
`summary.json` 和 `train.log`。prepare 阶段设备和 noise multiplier 尚未解析，
对应 metadata 为 null，训练初始化后更新实际值。summary 包含从第一轮构建
预条件器前到最后一次评估完成的总耗时、各轮平均耗时、整个 run 的最大显存
和最大预条件器存储量；CPU 显存指标为空。

也可分两步调用 Python 入口：

```bash
conda run -n curve python scripts/train.py --config configs/standalone/mnist_dp_sgd.yaml --prepare-run
conda run -n curve python -u scripts/train.py --config configs/standalone/mnist_dp_sgd.yaml --run-dir <上一步输出目录>
```

每个 prepared directory 只用于一次训练。相对数据和输出路径基于当前工作目录；
`run.sh` 使用仓库根目录。启动脚本要求 tmux 存在，打印 attach、tail 和 kill 命令。

验证：

```bash
PYTHONPATH=src conda run -n curve python -m pytest -q tests
```

测试使用临时 CPU 小配置，不改动正式 YAML；覆盖三个算法、输出命名与碰撞、
逐轮日志、隐私操作顺序、RNG 隔离和与 Exp6 构建结果的逐张量精确回归。
