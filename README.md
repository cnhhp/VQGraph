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
- **Text DGL Cora** (recommended for LLM pipeline): put files under `data/dataset/cora/`:
  - `cora_graph.pth` — DGL graph (`feat` [N,768], `label`, `train/val/test_mask`)
  - `cora_text.pkl` — list of `"Title: ... Abstract: ..."` per node
  - `cora_metadata.pth` — `categories` per node
- **OGB** (`ogbn-arxiv`, `ogbn-products`): auto-downloaded via `dataloader.py`.
- **NonHom** (`pokec`, `penn94`) and **BGNN** (`house_class`, `vk_class`): see original dataset instructions.

### Text Cora auto-detection

`load_graph_data('cora', root='./data')` uses **text DGL first** when `data/dataset/cora/cora_graph.pth` exists. Otherwise it falls back to CPF `data/cora.npz`.

- Force text: `--data_source text`
- Force CPF BoW: `--data_source cpf` (or remove/rename `data/dataset/cora/`)

**After switching to text Cora, retrain the structural codebook** — node features are 768-dim (not 1433 BoW) and semantics use real Title+Abstract via Sentence-BERT:

```bash
python train_codebook.py --dataset cora --data_root ./data --compute_tfidf \
  --tokenbook_dir ./codebook --warmup_epochs 20 --lambda_semantic 0.1 --device 0

python preprocess_data.py --dataset cora --data_root ./data \
  --codebook_dir ./outputs/codebook/cora/GCN/seed_0 --tokenbook_path ./codebook
```

## GCN classification baseline (no VQ)

Standard 2-layer GCN for node classification on the **same Cora masks** as the LLM pipeline (140 / 500 / 1000). Use this to compare against QLoRA and the VQ codebook teacher.

```bash
python train_gcn_baseline.py --dataset cora --data_root ./data --device 0 --console_log --save_model
```

Output: `outputs/baseline_gcn/cora/seed_0/baseline_metrics.json` (train / val / test accuracy and per-class breakdown).

Hyperparameters default from `train.conf.yaml` (`cora` → `GCN`: hidden 64, dropout 0.8, weight_decay 0.001; lr defaults to 0.01).

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

## Module 5: LLM fine-tuning

Generate JSONL first (module 2–4), then fine-tune with LoRA / QLoRA:

```bash
# Extra deps for module 5
pip install transformers>=4.40 peft>=0.10 datasets accelerate
# Optional for QLoRA 4-bit (CUDA only):
# pip install bitsandbytes
```

**CPU / smoke test** (small model, few samples):

```bash
python finetune_llm.py \
  --train_jsonl ./data/llm_finetune/train.jsonl \
  --val_jsonl ./data/llm_finetune/val.jsonl \
  --test_jsonl ./data/llm_finetune/test.jsonl \
  --base_model Qwen/Qwen2.5-0.5B-Instruct \
  --mode lora \
  --max_samples 20 \
  --lora_epochs 1 \
  --device -1 \
  --output_dir ./outputs/llm_smoke
```

**Full training** (CUDA + HuggingFace access for Llama-3-8B):

```bash
python finetune_llm.py \
  --train_jsonl ./data/llm_finetune/train.jsonl \
  --val_jsonl ./data/llm_finetune/val.jsonl \
  --test_jsonl ./data/llm_finetune/test.jsonl \
  --mode both \
  --device 0 \
  --output_dir ./outputs/llm
```

Notes:

- Default base model in `config.py` is `meta-llama/Meta-Llama-3-8B-Instruct`; override with `--base_model`.
- QLoRA needs CUDA + `bitsandbytes`. On CPU or Windows without 4-bit support, use `--mode lora` or `--skip_qlora`.
- Outputs: `outputs/llm/qlora/`, `outputs/llm/lora/`, and `finetune_metrics.json`.

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
| `preprocess_data.py` | JSONL generation (modules 2–4) |
| `finetune_llm.py` | LLM LoRA / QLoRA fine-tuning (module 5) |
| `models/llm_finetune.py` | Instruction dataset + LLMFinetuner |

## Citation

```bibtex
@inproceedings{yang2024vqgraph,
  title={VQGraph: Rethinking Graph Representation Space for Bridging GNNs and MLPs},
  author={Ling Yang and Ye Tian and Minkai Xu and Zhongyi Liu and Shenda Hong and Wei Qu and Wentao Zhang and Bin CUI and Muhan Zhang and Jure Leskovec},
  booktitle={International Conference on Learning Representations},
  year={2024}
}
```
