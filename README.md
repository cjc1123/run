# HybrIK

HybrIK: Hybrid Analytical-Neural Inverse Kinematics for Body Mesh Recovery

本仓库托管 HybrIK 项目的全部源代码（约 56 MB）。

## 仓库结构

- [`HybrIK-/`](./HybrIK-/) — 项目源码（`configs/`、`hybrik/`、`scripts/`、`setup.py` 等），完整说明见该目录下的 [README](./HybrIK-/README.md)

## 模型权重下载

由于 GitHub 单文件限制（100 MB），模型权重文件（SMPLX 模型 `.pkl`/`.npz`、预训练权重 `.pth`）未包含在本仓库中，已托管在 Hugging Face：

- **Hugging Face 官方链接**：https://huggingface.co/cjccjc/run
- **国内镜像（推荐）**：https://hf-mirror.com/cjccjc/run

下载后按项目 README 说明放置到 `model_files/` 和 `pretrained_models/` 目录即可使用。

## 论文

- HybrIK: https://arxiv.org/abs/2011.14672
- HybrIK-X: https://arxiv.org/abs/2304.05690
