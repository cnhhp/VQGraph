# VQGraph Teacher (GCN / SAGE + Codebook)

This repository trains a **teacher GNN with a VQ-VAE style graph tokenizer** and exports the learned **codebook**. Student MLP distillation has been removed; use `train_teacher.py` only.

Paper: [VQGraph: Rethinking Graph Representation Space for Bridging GNNs and MLPs](https://openreview.net/forum?id=h6Tz85BqRI) (ICLR 2024).

## Preparation

```bash
conda create -n vqgraph python=3.9
conda activate vqgraph
pip install -r requirements.txt
```

## Datasets

Place datasets under `data/` (same as the original VQGraph README):

- **CPF** (`cora`, `citeseer`, `pubmed`, `a-computer`, `a-photo`): `.npz` files from [Dropbox](https://www.dropbox.com/sh/fchrckrpf99gho2/AABZwMOeOnuiCxBjqYd46Qz3a?dl=0).
- **OGB** (`ogbn-arxiv`, `ogbn-products`): auto-downloaded via `dataloader.py`.
- **NonHom** (`pokec`, `penn94`) and **BGNN** (`house_class`, `vk_class`): see original dataset instructions.

## Train teacher and export codebook

**GCN (full-graph, recommended for small graphs):**

```bash
python train_teacher.py --exp_setting tran --teacher GCN --dataset cora --output_path outputs --seed 0 --max_epoch 100 --patience 50 --device 0
```

**SAGE (mini-batch, for large graphs):**

```bash
python train_teacher.py --exp_setting tran --teacher SAGE --dataset ogbn-products --output_path outputs --seed 0 --device 0
```

**Output directory:**

```text
outputs/transductive/{dataset}/{GCN|SAGE}/seed_{seed}/
  codebook_embeddings.npz   # main artifact, shape [codebook_size, feat_dim]
  log
  model.pth                 # only if --save_results
  loss_and_score.npz        # only if --save_results
```

Load codebook:

```python
import numpy as np
cb = np.load("outputs/transductive/cora/GCN/seed_0/codebook_embeddings.npz")["arr_0"]
```

**Inductive setting** (optional): `--exp_setting ind --split_rate 0.2`

## Project layout

| File | Role |
|------|------|
| `train_teacher.py` | Entry point |
| `train_and_eval.py` | Training / evaluation loop |
| `models.py` | GCN & SAGE with VQ |
| `vq.py` | Vector quantization module |
| `dataloader.py` | Dataset loading |
| `data_preprocess.py` | Graph preprocessing helpers |
| `utils.py` | Seeds, logging, config |
| `train.conf.yaml` | Per-dataset GCN/SAGE hyperparameters |

## Citation

```bibtex
@inproceedings{yang2024vqgraph,
  title={VQGraph: Rethinking Graph Representation Space for Bridging GNNs and MLPs},
  author={Ling Yang and Ye Tian and Minkai Xu and Zhongyi Liu and Shenda Hong and Wei Qu and Wentao Zhang and Bin CUI and Muhan Zhang and Jure Leskovec},
  booktitle={International Conference on Learning Representations},
  year={2024}
}
```
