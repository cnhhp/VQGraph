# VQGraph 项目交接文档

> 更新日期：2026-07-18  
> 仓库：https://github.com/cnhhp/VQGraph  
> 规格说明：[`prompt.md`](prompt.md)  
> 新增：§17 层次粗/细码 + HeteroGen（**训练逻辑已实现**；LLM JSON 未做）

---

## 1. 项目目标

将引文网络图节点离散化为「结构 token + 文本 token」序列，经偏置欧拉游走序列化后，供大模型（LLM）做节点分类微调。

**Pipeline：** 模块1 结构码本(E1) → E5b predictor + TF-IDF → 模块2 子图 → 模块3 离散化 → 模块4 序列化 → 模块5 LLM 微调

---

## 2. 环境与路径

| 项 | 值 |
|----|-----|
| Python | **3.9**（本机 env：`vqgraph_py39`） |
| 本机项目路径 | `c:\Users\hhp\Desktop\组会\idea\VQGraph\VQGraph` |
| Conda（本机） | `D:\anaconda\envs\vqgraph_py39\python.exe` |
| 服务器路径 | `~/huanghp_2252895/VQGraph` |
| 服务器 Conda | `2252895_vqgraph` |
| Git 最新提交 | `75c3aae` — token predictor + E5b 码本级训练 + P_code 推理融合 |

### SSH（已配置 `~/.ssh/config`）

```text
Host vqgraph-gpu
    HostName 10.199.227.106
    User power
    Port 8122
```

连接：`ssh vqgraph-gpu`  
**注意：** 服务器 **HTTP/HTTPS 出网受限**，须使用本地模型目录，无法在线拉 HF。

---

## 3. 数据集

**优先自动加载：** `data/dataset/{name}/` 存在时走文本版 DGL，否则回退 CPF `data/{name}.npz`。

### 3.1 Cora（文本 DGL）

| 文件 | 说明 |
|------|------|
| `cora_graph.pth` | DGL 图；`feat` [2708,768] |
| `cora_text.pkl` | 每节点 `Title: ... Abstract: ...` |
| `cora_metadata.pth` | `categories` → 7 类 |

**两种划分：**

| 划分 | train/val/test | 用途 |
|------|----------------|------|
| 官方 mask | 140 / 500 / 1000 | 小样本 LLM 微调、与 GNN 对齐 |
| **60/20/20 分层** | 1624 / 542 / 542 | 更多 train；`make_stratified_ratio_split()`，seed=42 |

- 强制文本：`--data_source text`
- `data/dataset/cora/`（~1.2GB）**未进 git**

### 3.2 PubMed（文本 DGL）

| 项 | 值 |
|----|-----|
| 节点 | ~19717 |
| 类 | 3（Diabetes Mellitus / Experimental / Type 1） |
| 划分 | **60/20/20 分层** → 11830 / 3943 / 3944 |
| 码本 | `e5b_pubmed_m4096`（M=4096，no_ltoken） |

- 数据目录：`data/dataset/pubmed/`（未进 git）
- 多数类 baseline ≈ 40%；GNN codebook val ≈ 70%

---

## 4. 模块进度

| 模块 | 状态 | 要点 |
|------|------|------|
| **1 结构码本 (E1)** | ✅ | 语义偏置 VQ；**可不加 L_token**（`--no_token_predictor`） |
| **E5b predictor** | ✅ | 冻结 E1 → factorized + 码本级 KL → 独立 TF-IDF |
| **GCN 分类基线** | ✅ | `train_gcn_baseline.py` |
| **2 子图** | ✅ | `SubgraphExtractor`，默认 **k=2** |
| **3 节点离散化** | ✅ | text_sim × (1 + λ_tfidf·TF-IDF + λ_pred·P_code) + MMR |
| **3b TokenSelector** | ✅ | 可学习重排网络；Gumbel 软选择训练；推理接入 preprocess |
| **4 序列化** | ✅ | `BiasedEulerSerializer`；α/β/γ=0.4/0.3/0.3 |
| **TextTokenbook** | ✅ | `codebook/filtered_tokenbook.npy`，V=13648 |
| **preprocess_data** | ✅ | 支持 `--lambda_pred` / `--p_code_normalize` |
| **5 LLM 微调** | ⚠️ | 代码完成；Cora/PubMed QLoRA **在跑/待汇总**；**无内置 train acc** |
| **struct_token 多模式** | ⚠️ | `id` / `pcode_*` / `struct_summary` 代码已有；**preprocess 未完全接入** |
| **strip struct** | ✅ | `scripts/strip_struct_tokens_jsonl.py` |
| **60/20/20 split** | ✅ | `graph_utils.make_stratified_ratio_split()` |
| **层次粗/细码 + HeteroGen** | ✅ 训练侧 | `--hierarchical_vq`：双通道 GCN + 粗/细 VQ + L_H/L_D；见 **§17**；**LLM JSON 未做** |

---

## 5. 核心结论：P_code 与 E1/E5b 分工

### P_code 曾无效的三层原因

1. **τ'=0.15** → 13648 维 softmax 熵 ≈ log(V)，L_token 卡在均匀解  
2. **训练-推理不对齐** → 训练用节点 `z_q`，推理用 `codebook[c]`；联合训练 linear 头仍输出均匀  
3. **推理尺度失衡** → raw P_code max ≈ 1/V；须 **`p_code_normalize=max`**，且 **`lambda_pred` 不宜过大**（0.3 会 100% 改词）

### 有效配方（E5b）

```text
E1 全量码本（τ'=0.03，可无 L_token）
  → 冻结 E1
  → E5b：predictor_only + factorized + Top-64 KL + τ'=0.03 + λ_token=1.0（码本级 KL，目标 semantic_centers[c]）
  → --compute_tfidf（稳定期 node_codes）
  → preprocess：p_code_normalize=max, lambda_pred=0.05
```

**选词公式：**

```text
score[t] = text_sim[t] × (1 + λ_tfidf·TF-IDF[t] + λ_pred·P_code_norm[t])
P_code_norm = P_code / max(P_code)    # normalize=max 必须
```

### E1 vs E5b「码本」

| | E1 结构码本 | E5b 产物目录 |
|--|------------|-------------|
| 训练什么 | GCN + VQ（可选弱 L_token） | **仅** factorized predictor |
| 产出 `<S_k>` | ✅ | 复用 E1，不变 |
| 产出 P_code | ❌ 弱/不可用 | ✅ 用于选词 |
| E1 去掉 L_token | ✅ 可以 | **不影响** E5b |

**Factorized predictor**：`proj(结构码) · tokenbook_emb^T`，与 text_sim 同构；推理时提供码级词表先验，**不进入 LLM**。

### 5.1 本对话核心诊断：Codebook → LLM 信息链路

**完整链路：**

```text
E1: GNN h → VQ → codebook[c] (256-d) + semantic_centers μ_k
E5b: P_code[t|c] = softmax(proj(c)·tokenbook^T)
模块3: score[t] = text_sim × (1 + λ_tfidf·TF-IDF + λ_pred·P_code) → MMR → 8 text tokens
模块4: 每行 "{level} <S_k> [8 words]" → JSONL → QLoRA
```

**产物是否进入 LLM：**

| 产物 | 含义 | 进入 LLM？ |
|------|------|-----------|
| `codebook[c]` 256-d 向量 | 结构原型 | ❌ 仅 argmax 取 index |
| `semantic_centers[c]` | 码的典型语义 | ❌ 仅 E5b 训练 |
| `P_code` | 结构→词表先验 | ⚠️ 仅 λ_pred=0.05 弱影响选词 |
| `TF-IDF[c][t]` | 结构→词频先验 | ⚠️ 仅影响选词 |
| `<S_k>` | 离散 ID | ✅ 进 prompt，**对 Llama 无语义** |

**根本错配：** Codebook 为 **GNN 单节点分类** 优化，LLM 任务为 **2-hop 子图序列 → 中心节点分类**；结构主通道被压成 **无义 ID + 5% 选词旁路**。

**`<S_k>` 重复问题（Cora 2-hop 常见 150–250 行，同一码可出现 80+ 次）：**

- 原因：每节点独立 argmax，同质邻居共享码；`serialize_tree` 每行仍打印 struct
- 浪费 context，第 2 次起信息增量 ≈ 0
- **推荐改法（未实现）：** 首行 `struct_summary`（码分布 + center P_code 词）+ 行内去掉 `<S_k>`；或每码全文只标注一次

**改进路线图（按成本）：**

1. **Quick win：** `struct_summary` + 行内无 struct；或 `pcode_supplement`（`<S_k|w1,w2,w3>`）
2. **中期：** 自适应 λ_pred、按码聚类行、码游走摘要
3. **长期：** LLM-aware codebook 重训 / codebook→LLM adapter

**对比实验建议：** `e5b`（现状） vs `e5b_no_stoken` vs `e5b+struct_summary+无行内struct` vs `nocode`

---

## 6. 关键配置（`config.py`）

```text
# 模块1 — E1 默认（E5b 命令行覆盖）
enable_token_predictor=True, lambda_token=0.05
token_target_temperature=0.15          # E1/E5b 实验用 0.03
token_predictor_type=linear            # E5b 用 factorized
token_kl_top_k=0                     # E5b 用 64
warmup_epochs=20, tfidf_stats_min_epoch=None  # → epoch>=21 才参与 best/TF-IDF

# 模块3 — 推理融合（已更新）
p_code_normalize=max
lambda_pred=0.05                     # sweep 最佳；原 0.5 过强
lambda_tfidf=0.5
top_k_text_tokens=8, subgraph_k_hop=2, mmr_candidate_pool=96, mmr_lambda=0.5

# 模块3b — TokenSelector（train_token_selector.py）
token_selector_hidden_dim=128, token_selector_lr=1e-3, token_selector_epochs=50
token_selector_batch_size=32
gumbel_tau_init=1.0, gumbel_tau_min=0.5, gumbel_tau_anneal_epochs=30
token_selector_candidate_pool=128   # s0 Top-128 候选池
token_selector_training_mode=distill # distill（默认）| cls
token_selector_distill_weight=1.0     # KL(P_student || P_target_mmr)
token_selector_cls_weight=0.05       # 蒸馏模式弱分类损失
token_selector_student_temperature=1.0
token_selector_kl_weight=0.3         # 仅 cls 模式：KL(w||softmax(s0))
token_selector_entropy_weight=0.01   # 仅 cls 模式
token_selector_vtext_dropout=0.3     # 仅 cls 模式
filter_noise_subwords_at_selection=True  # 推理时过滤 PDF/子词噪声

# 模块5 — config 默认（服务器正式跑可覆盖）
lora_r=8, lora_alpha=16, lora_dropout=0.1, finetune_lr=1e-4
qlora_epochs=2, max_seq_length=768
```

**服务器 QLoRA 超参（历史）：** `qlora_epochs=5, lora_r=16, lora_alpha=32, lora_dropout=0.05, finetune_lr=2e-4, max_seq_length=1024`

**PubMed 当前推荐（本对话）：** `qlora_epochs=3, finetune_lr=5e-5, max_seq_length=1536`（lora 仍用 config 默认 8/16/0.1）

**struct_token_mode（`config.py`）：** `id` | `pcode_supplement` | `pcode_replace` | `struct_summary`  
- `build_subgraph_struct_summary()` 已实现（`node_representation.py`）  
- `struct_summary` 实验 JSONL：`data/llm_finetune_json/e5b_no_ltoken_struct_summary/`（Cora 140/500/1000）  
- ⚠️ `preprocess_data.py` 尚未自动 prepend summary / 切换 instruction，需手动或待改代码

**QLoRA 指标限制：** `--mode qlora` 训练结束只跑 **val acc**，写入 `qlora/finetune_metrics.json`；**无 train acc**。训练过程 `eval_strategy=no`，无 epoch 级 val。事后需脚本加载 adapter 分别 `evaluate_accuracy("train"/"val")`。

---

## 7. 码本与 JSONL 产物矩阵

> 均在 `outputs/experiments/` 与 `data/llm_finetune_json/`，**未 commit**；同步服务器需 `scp`。

### 7.1 Cora（官方 140/500/1000）

| 系列 | E1 目录 | JSONL 目录 | 说明 |
|------|---------|-----------|------|
| no_ltoken baseline | `e5b_no_ltoken/cora/GCN/seed_1` | `llm_finetune_e5b_no_ltoken` | λ=0.05，纯 `<S_k>` |
| nocode 对照 | 同上 | `llm_finetune_nocode_no_ltoken` | λ=0 |
| **no_stoken** | — | `llm_finetune_e5b_no_ltoken_no_stoken` | strip 后 `<level> [words]` |
| struct_summary | 同上 | `e5b_no_ltoken_struct_summary` | 首行摘要 + 行间 `<S_k>` |
| pcode_supplement | 同上 | `e5b_no_ltoken_pcode_supplement` | `<S_k\|w1,w2,w3>` |
| learned / distill | + TokenSelector | `llm_finetune_e5b_learned*` / `llm_finetune_distill` | 可学习选词 |

### 7.2 Cora 60/20/20（1624/542/542）

| JSONL 目录 | 码本 | 说明 |
|-----------|------|------|
| `llm_finetune_e5b_no_ltoken_cora_602020` | `e5b_no_ltoken` | baseline |
| `llm_finetune_e5b_no_ltoken_cora_602020_no_stoken` | — | strip 版 |
| `llm_finetune_nocode_no_ltoken_no_stoken` | 同上 | nocode + strip |

### 7.3 PubMed 60/20/20（11830/3943/3944）

| 项 | 路径 |
|----|------|
| 码本 | `outputs/experiments/e5b_pubmed_m4096/pubmed/GCN/seed_1/` |
| JSONL | `data/llm_finetune_json/llm_finetune_e5b_no_ltoken_pubmed/` |
| 格式 | baseline：纯 `<S_k>`，无 TokenSelector，无 struct_summary |

路径示例：

```text
outputs/experiments/e5b_no_ltoken/cora/GCN/seed_1/
  model.pth, codebook_embeddings.npz, semantic_centers.npz, node_codes.npz
  tfidf_stats.npz, tfidf_stats.vocab.json

data/llm_finetune_json/llm_finetune_e5b_no_ltoken/
  train.jsonl (140), val.jsonl (500), test.jsonl (1000), manifest.json
```

**no_ltoken Cora 指标：** E1 best ep86 val **75.4%**；E5b L_token **0.899**

**PubMed QLoRA 现象（ep5, lr=2e-4, seq=768/1024）：** val acc ≈ **60%**（3 类，优于 40% 多数类基线，但低于 GNN ~70%）。可能原因：11k train 步数过多易过拟合、**seq 截断**（avg input ~1440 chars，`full_ids[:max_len]` 从开头截可能丢 label 侧信息）、结构通道弱。

**实验汇总：** `outputs/experiments/results.json`

**JSONL 格式：** `instruction` + `input`（层级化子图序列）+ `output`（类名）

---

## 8. JSONL 词频分析要点（no_ltoken 对）

- Top 词：`learning`(6%) > `algorithms`(4%) > `classification` > `neural` — 跨类泛化词占比高  
- **P_code(λ=0.05)** 推高：classification、neural、inference、prediction  
- **λ=0** 更多：optimized、gene、recognition、patterns  
- 噪声：minipage、episodio 等 PDF 解析残留  

**选词质量改进方向（部分已实现）：** MMR 目标蒸馏（方案 A，默认）+ s0 候选池 + 噪声过滤；旧 cls 模式见 `--training_mode cls`

### TokenSelector 流程

**默认训练（方案 A：MMR 目标蒸馏）**

```text
预计算：s0 → baseline MMR 硬 Top-k → P_target（均匀 1/k，detach）
训练：s0 → TokenSelector → softmax(s) = P_student
      → L = λ_distill·KL(P_student || P_target) + λ_cls·CrossEntropy（弱）
      Gumbel 路径仅用于弱分类梯度
推理：s0 → TokenSelector（池外 -inf）→ MMR → Top-k
```

**旧模式（`--training_mode cls`）**

```text
训练：s0 Top-128 候选池 → Gumbel → v_text → NodeClassifier
      → L = L_cls + λ_kl·KL(w||softmax(s0)) − λ_ent·H(w)
```

> **注意：** 旧版 checkpoint（无 candidate_pool）在加载时默认 pool=256；旧 learned JSONL 曾出现全图相同噪声 token 的模式坍缩，需用新版重训后再生成 JSONL。

- checkpoint：`outputs/token_selector/{dataset}/seed_{seed}/best.pth`
- preprocess 加 `--token_selector_checkpoint` 即可生成 learned JSONL
- 默认 **冻结** token_predictor；联合训练加 `--train_predictor`

---

## 9. 常用命令

### E1 结构码本（无 L_token，推荐职责分离）

```bash
python train_codebook.py --dataset cora --data_root ./data --data_source text \
  --output_dir ./outputs/experiments/e1_no_ltoken --tokenbook_dir ./codebook \
  --no_token_predictor --lambda_semantic 0.1 --warmup_epochs 20 \
  --seed 1 --device 0 --console_log
```

### E5b predictor（冻结 E1）

```bash
python train_codebook.py --dataset cora --data_root ./data --data_source text \
  --output_dir ./outputs/experiments/e5b_no_ltoken --tokenbook_dir ./codebook \
  --load_checkpoint ./outputs/experiments/e1_no_ltoken/cora/GCN/seed_1/model.pth \
  --init_from_dir ./outputs/experiments/e1_no_ltoken/cora/GCN/seed_1 \
  --predictor_only --predictor_only_epochs 80 --predictor_lr 0.005 \
  --token_predictor_type factorized --token_kl_top_k 64 \
  --lambda_token 1.0 --token_target_temperature 0.03 --seed 1 --device 0 --console_log
```

### TF-IDF（`--tfidf_only` 已 fix：不再误删已有目录）

```bash
python train_codebook.py --tfidf_only --compute_tfidf --dataset cora \
  --output_dir ./outputs/experiments/e5b_no_ltoken --tokenbook_dir ./codebook \
  --seed 1 --device 0
```

### 生成 JSONL

```bash
# 有 P_code
python preprocess_data.py --dataset cora --data_root ./data --data_source text \
  --codebook_dir ./outputs/experiments/e5b_no_ltoken/cora/GCN/seed_1 \
  --tokenbook_path ./codebook --output_dir ./data/llm_finetune_e5b_no_ltoken \
  --lambda_pred 0.05 --p_code_normalize max --seed 1 --device 0

# 无 P_code 对照（同码本）
python preprocess_data.py ... \
  --output_dir ./data/llm_finetune_nocode_no_ltoken --lambda_pred 0 --device 0

# 可学习 TokenSelector 重排（需先 train_token_selector.py）
python preprocess_data.py --dataset cora --data_root ./data --data_source text \
  --codebook_dir ./outputs/experiments/e5b_no_ltoken/cora/GCN/seed_1 \
  --output_dir ./data/llm_finetune_e5b_learned \
  --token_selector_checkpoint ./outputs/token_selector/e5b_no_ltoken/cora/seed_42/best.pth \
  --lambda_pred 0.05 --p_code_normalize max --device 0
```

### 训练 TokenSelector（默认 MMR 蒸馏）

```bash
python train_token_selector.py --dataset cora --data_root ./data --data_source text \
  --codebook_dir ./outputs/experiments/e5b_no_ltoken/cora/GCN/seed_1 \
  --tokenbook_path ./codebook \
  --output_dir ./outputs/token_selector/e5b_distill \
  --lambda_pred 0.05 --p_code_normalize max \
  --epochs 50 --batch_size 32 --lr 1e-3 --seed 42 --device 0 --console_log

# 复现旧 Gumbel+分类目标：
#   ... --training_mode cls --kl_weight 0.3 --entropy_weight 0.01
```

### 选词 / predictor 评估

```bash
PYTHONPATH=. python scripts/compare_token_selection.py \
  --codebook_dir ./outputs/experiments/e5b_no_ltoken/cora/GCN/seed_1 \
  --tokenbook_dir ./codebook --nodes train --lambda_pred 0.05 --p_code_normalize max --device 0

PYTHONPATH=. python scripts/eval_predictor.py \
  --codebook_dir ./outputs/experiments/e5b_no_ltoken/cora/GCN/seed_1 --device 0

PYTHONPATH=. python scripts/eval_token_selector.py \
  --codebook_dir ./outputs/experiments/e5b_no_ltoken/cora/GCN/seed_1 \
  --token_selector_checkpoint ./outputs/token_selector/e5b_no_ltoken/cora/seed_42/best.pth \
  --split val --device 0
```

### Strip `<S_k>`（后处理，不改 preprocess）

```bash
python scripts/strip_struct_tokens_jsonl.py \
  --input_dir ./data/llm_finetune_json/llm_finetune_e5b_no_ltoken \
  --output_dir ./data/llm_finetune_json/llm_finetune_e5b_no_ltoken_no_stoken
```

### 服务器 QLoRA

```bash
cd ~/huanghp_2252895/VQGraph && conda activate 2252895_vqgraph
tmux new -s qlora_pubmed   # 断线：tmux attach -t qlora_pubmed

# Cora baseline 对比（140 train）
python finetune_llm.py \
  --train_jsonl ./data/llm_finetune_json/llm_finetune_e5b_no_ltoken/train.jsonl \
  --val_jsonl ./data/llm_finetune_json/llm_finetune_e5b_no_ltoken/val.jsonl \
  --test_jsonl ./data/llm_finetune_json/llm_finetune_e5b_no_ltoken/test.jsonl \
  --base_model ~/huanghp_2252895/Meta-Llama-3-8B-Instruct \
  --mode qlora --qlora_epochs 5 --device 0 \
  --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 --finetune_lr 2e-4 \
  --max_seq_length 1024 \
  --output_dir ./outputs/llm_qlora_e5b_no_ltoken_ep5

# PubMed（当前推荐超参）
python finetune_llm.py \
  --train_jsonl ./data/llm_finetune_json/llm_finetune_e5b_no_ltoken_pubmed/train.jsonl \
  --val_jsonl ./data/llm_finetune_json/llm_finetune_e5b_no_ltoken_pubmed/val.jsonl \
  --test_jsonl ./data/llm_finetune_json/llm_finetune_e5b_no_ltoken_pubmed/test.jsonl \
  --base_model ~/huanghp_2252895/Meta-Llama-3-8B-Instruct \
  --mode qlora --device 0 \
  --qlora_epochs 3 --max_seq_length 1536 --finetune_lr 5e-5 \
  --output_dir ./outputs/llm_qlora_e5b_no_ltoken_pubmed_ep3_seq1536_lr5e5
```

**查看 val acc：**

```bash
cat ./outputs/llm_qlora_*/qlora/finetune_metrics.json
tail -50 ./outputs/llm_qlora_*/finetune.log
```

**事后 train + val acc（需加载 adapter，train 全量很慢）：**

```bash
python - <<'PY'
from pathlib import Path
from config import reset_config, get_config
from models.llm_finetune import LLMFinetuner
from peft import PeftModel

reset_config()
cfg = get_config()
cfg.base_model_name = str(Path.home() / "huanghp_2252895/Meta-Llama-3-8B-Instruct")
cfg.max_seq_length = 1536

train = Path("data/llm_finetune_json/llm_finetune_e5b_no_ltoken_pubmed/train.jsonl")
val   = Path("data/llm_finetune_json/llm_finetune_e5b_no_ltoken_pubmed/val.jsonl")
adapter = Path("outputs/llm_qlora_e5b_no_ltoken_pubmed_ep3_seq1536_lr5e5/qlora")

f = LLMFinetuner(cfg, device=0)
f.load_base_model(use_qlora=True)
f.apply_lora()
f.model = PeftModel.from_pretrained(f.model, str(adapter))
f._prepare_datasets(train, val, train)
print("train@500", f.evaluate_accuracy("train", max_samples=500))
print("val", f.evaluate_accuracy("val"))
PY
```

`screen -S qlora` 或 **tmux** 防断线。

---

## 10. 基座模型

| 方式 | 说明 |
|------|------|
| 魔搭 ModelScope | `LLM-Research/Meta-Llama-3-8B-Instruct` → 本机 `D:\models\` |
| 服务器 | `~/huanghp_2252895/Meta-Llama-3-8B-Instruct`（需 scp 上传） |
| HF 直连 | 本机常 SSL 失败 |

脚本：`scripts/download_llama_modelscope.py`

---

## 11. Git 状态

**已 push（`75c3aae`）：** token predictor、E5b predictor_only、Top-K KL、factorized 头、P_code 归一化、`preprocess_data` CLI、`compare_token_selection.py`、`eval_predictor.py`、tfidf_only 删目录 bug fix

**未提交（本地）：**

- `data/dataset/cora/`、`outputs/`、各 `data/llm_finetune_*` JSONL 目录

---

## 12. 重要文件索引

| 文件 | 作用 |
|------|------|
| `prompt.md` | 完整规格 |
| `config.py` | 全局超参 |
| `train_codebook.py` | 模块1 + TF-IDF 入口 |
| `models/token_predictor.py` | Factorized 头、Top-K KL、P_code 归一化 |
| `models/codebook_trainer.py` | E5b predictor_only、码本级 KL、TF-IDF |
| `models/node_representation.py` | 选词融合 + MMR + TokenSelector 推理 |
| `models/token_selector.py` | TokenSelector、NodeClassifier、Gumbel 训练 |
| `train_token_selector.py` | 可学习选词训练入口 |
| `preprocess_data.py` | JSONL 生成（`--lambda_pred` / `--token_selector_checkpoint`） |
| `finetune_llm.py` | LLM 微调入口 |
| `scripts/compare_token_selection.py` | λ=0 vs λ>0 选词对比 |
| `scripts/eval_predictor.py` | p_max / KL 快速评估 |
| `scripts/eval_token_selector.py` | 固定 vs 可学习选词对比 |
| `scripts/strip_struct_tokens_jsonl.py` | JSONL 后处理：去掉行内 `<S_k>` |
| `graph_utils.py` | `make_stratified_ratio_split()` 60/20/20 |
| `outputs/experiments/results.json` | E1–E6 实验汇总 |

---

## 13. 本对话完成事项（2026-06-05）

| 事项 | 状态 |
|------|------|
| 可学习 TokenSelector（Gumbel + MMR 蒸馏模式） | ✅ |
| PubMed E1+E5b 码本（M=4096, no_ltoken） | ✅ |
| PubMed JSONL 60/20/20（11830/3943/3944） | ✅ |
| Cora JSONL 60/20/20（1624/542/542） | ✅ |
| `strip_struct_tokens_jsonl.py` → `*_no_stoken` 数据集 | ✅ |
| `struct_summary` / `pcode_supplement` 实验 JSONL（Cora 小 split） | ✅ |
| Codebook→LLM 信息链路多角度分析 | ✅（文档） |
| `<S_k>` 重复问题分析与改法设计 | ✅（**代码未改**） |
| PubMed QLoRA 服务器训练 | ⚠️ 进行中/待汇总 |
| preprocess 接入 struct_summary + 行内 dedup | ❌ 待做 |

---

## 14. 待续任务

1. **汇总 QLoRA：** PubMed（ep3/seq1536/lr5e-5）及 Cora e5b vs nocode vs no_stoken val/test acc  
2. **序列化改进（优先）：** `struct_summary` 首行 + 行内去掉重复 `<S_k>`，重新生成 JSONL 并 QLoRA 对比  
3. **层次粗/细码（§17）：** ✅ 训练侧已实现；下一步 P2 新 JSONL + QLoRA；可选 P3（L_M / 双 decoder / SAGE）  
4. **（可选）** preprocess 加 `--struct_token_mode` CLI；训练时记录 train/val acc  
5. **（可选）** PubMed 超参：若仍 ~60%，试 ep2 + lr1e-4 + 检查截断  
6. **（长期）** ogbn-arxiv 数据管线；LLM-aware codebook 重训  

---

## 15. 新对话开场可复制

```text
项目：VQGraph @ c:\Users\hhp\Desktop\组会\idea\VQGraph\VQGraph
GitHub: https://github.com/cnhhp/VQGraph
服务器：ssh vqgraph-gpu → ~/huanghp_2252895/VQGraph，conda 2252895_vqgraph

请先读 HANDOFF.md（§5.1 Codebook→LLM 诊断；§17 层次粗/细码设计）与 config.py。

已完成：
- E5b 配方：factorized + 码本级 KL + p_code_normalize=max + λ_pred=0.05
- Cora：140/500/1000 + 602020 JSONL；PubMed：11830/3943/3944 JSONL
- strip 版：*_no_stoken；struct_summary 实验 JSONL（preprocess 未完全接入）
- TokenSelector + MMR 蒸馏训练链路
- 核心结论：结构 codebook 信息几乎未进入 LLM 决策；<S_k> 重复浪费 context
- §17 层次粗/细码 + HeteroGen：**训练逻辑已实现**（`--hierarchical_vq`）；LLM JSON 未做

码本：
- Cora: outputs/experiments/e5b_no_ltoken/cora/GCN/seed_1/
- PubMed: outputs/experiments/e5b_pubmed_m4096/pubmed/GCN/seed_1/

JSONL/outputs 未 commit，需 scp 到服务器。

待做：[你的任务]
```

---

## 16. 历史：Git 与 seed 系列（Cora 140 split）

| 系列 | E1 | E5b | JSONL λ=0.05 | JSONL λ=0 |
|------|-----|-----|-------------|-----------|
| seed0 | `e1_tau003` | `e5b_code_level` | `llm_finetune_e5b` | `llm_finetune_v2` |
| seed1 | `e1_tau003_s1` | `e5b_code_level_s1` | `llm_finetune_e5b_s1` | `llm_finetune_nocode_s1` |
| **no_ltoken** | `e1_no_ltoken` | `e5b_no_ltoken` | `llm_finetune_e5b_no_ltoken` | `llm_finetune_nocode_no_ltoken` |

---

## 17. 设计草案：层次粗/细码 + HeteroGen 异配感知

> **状态：码本训练逻辑已实现**（`--hierarchical_vq`，仅 GCN）；**LLM JSON / preprocess 未做**。  
> 参考：*Graph Generation Beyond Homophily Assumptions*（HeteroGen）  
> 目标：缓解「邻域同码 + 异配/边缘节点误伤」；粗码=theme、细码=structure；分类主战场仍在 LLM。

### 17.0 训练 CLI（已实现）

```bash
python train_codebook.py --dataset cora --data_source text --teacher GCN \
  --hierarchical_vq \
  --codebook_size_coarse 16 --codebook_size_fine 128 \
  --lambda_H 1.0 --lambda_D 0.05 --lambda_L 0.1 \
  --lambda_div 0.05 --lambda_div_fi 0.5 --lambda_ico 0.2 \
  --select_min_s_L 0.75 --fine_noise 0.0 --lamb_edge 0.15 \
  --text_fuse 0.5 --lambda_semantic 0.3 --learning_rate 1e-3 \
  --no_token_predictor --epochs 120 --device 0 \
  --output_dir ./outputs/experiments/hier_vq_cora_v5
```

**v2（有主 CE）：** best GNN val **75.2%**；coarse **13/16**、fine **34/256**；细码–标签 NMI≈0.43；异配同细码 **30.5%**。  
**v3（有主 CE + 强 div）：** fine **47/256**；GNN val **71.8%**；异配同细码 **21.5%**。  

**v4 按原定去掉主 CE（2026-07-18）：** trainer 层次模式不再加 `criterion(logits,y)`；保留弱 \(L_L\)（\(\lambda_L=0.1\)，只约束 \(h_L\)）；边重建改回**只绑 \(z_{\mathrm{fi}}\)**；结构通道关闭 dropout / `fine_noise=0`；选模曾偏 unique。  
**Cora v4：** coarse **13/16**、fine **35/256**（\(N_{\mathrm{eff}}\approx10.5\)）；细码–标签 NMI **0.19**；同配/异配同细码 **33.8%/32.8%**（结构偏弱）。产物：`hier_vq_cora_v4/`。

**v5 修 v4 结构弱/选模偏（2026-07-18）：**  
- 边损失：cosine + 正负采样（去掉 min-max MSE）；\(L_H\) 含轻推异配；`lamb_edge=0.15`、`lambda_H=1.0`  
- \(L_{\mathrm{intra\_co}}\)（`lambda_ico=0.2`）：同粗码邻居拉开细码  
- 选模：`s_L≥0.75` 后再最大化 `uniq_fi`  
- \(M_{\mathrm{fi}}=128\)  
**Cora v5（seed_1, best ep119）vs v4：**  

| | v4 | **v5** |
|--|----|--------|
| fine unique | 35/256 (14%) | **90/128 (70%)** |
| fine \(N_{\mathrm{eff}}\) | 10.5 | **46.2** |
| fine–label NMI | 0.19 | 0.31（仍远低于有主 CE 的 ~0.45） |
| 同配/异配同细码 | 33.8%/32.8% (gap≈0) | **31.2%/22.2% (gap=0.09)** |
| coarse Neff / NMI | 4.6 / 0.39 | **11.7 / 0.61** |
| s_L（弱 \(L_L\)） | ~66%（早停） | **~86%** |
| 类内 fine Neff 均值 | 7.3 | **18.0** |

产物：`outputs/experiments/hier_vq_cora_v5/`。

**产物（层次模式）：** `codebook_coarse.npz` / `codebook_fine.npz` / `node_codes_coarse.npz` / `node_codes_fine.npz`；兼容字段 `codebook_embeddings.npz` = fine，`node_codes.npz` = fine。  
**核心文件：** `models/hierarchical_encoder.py`（`HierarchicalGCN` + `DualChannelAgg`）。  
**开关默认关：** 不加 `--hierarchical_vq` 时行为与旧 E1 一致。  
**限制：** SAGE + hierarchical 暂不支持。

### 17.1 动机与现状瓶颈

**灾难链（现 E1 单码本）：**

```text
GNN 邻居一律平滑 → h 像邻居 → VQ 同码 <S_k>
→ 边重建 Â ≈ z·zᵀ（越像越该有边）强化同配
→ TF-IDF / P_code / 行内 <S_k> 重复强化
→ 异配节点被当成「和邻居一类」；LLM 主要靠 8 个 text token
```

**现状代码（`models.py`）：** 单 `GraphConv` → 单 VQ → `decoder_1/2` + `z@z.T` 全图边 MSE；**不区分同配/异配**。  
训练时赋码 = 相似度 argmax；训练结束存 `node_codes.npz`；preprocess 再跑一遍 `_infer_distances` 多为重复（可直接读存盘码）。

### 17.2 从 HeteroGen 借 / 不借

| 借 | 不借 |
|----|------|
| \(h=[h_L\|h_{\bar L}]\) 解耦 | 扩散生成 \(L_P\) / 整图生成 |
| \(N^S\) / \(N^D\) 分路径聚合 | 粗码 = 类名进 prompt |
| \(L_L\)（可选、小权重，只约束连续 \(h_L\)） | 细码走 P_code 改选词 |
| \(L_D\)（dCov 解耦） | 必须用 y 分邻居（可用 proxy） |
| \(L_M\)（结构通道影响平衡，可选） | |
| \(L_H\)（同配/异配边重建平衡，**P1 优先**） | |

原版 VQGraph Tokenizer：`L_Rec + L_CE + L_VQ + L_commit`。  
本方案：`L_Rec/commit` 拆粗细两套；`L_CE` → 可选弱 \(L_L\) 或去掉；最终分类 CE 在 QLoRA。

### 17.3 分工一览

| 线 | 连续表示 | 离散码 | 含义 | 主要约束 |
|----|----------|--------|------|----------|
| 语义/主题 | \(h_L\) | **粗码** \(c^{\mathrm{co}}\)，\(M_{\mathrm{co}}\approx16\) | 邻域 **theme** | SemanticVQ + text；**不用 y 指定码号** |
| 结构 | \(h_{\bar L}\) | **细码** \(c^{\mathrm{fi}}\)，\(M_{\mathrm{fi}}\approx256\) | **结构角色** | 残差 VQ + 边重建；+ \(L_H\) |

**硬原则：**

1. 粗码 = theme，**不是** class name  
2. 细码 = structure，**不**走 P_code 选词  
3. 邻居行 **不写** struct（防同码 repetition）  
4. 分类靠 center text + LLM；结构辅助  
5. \(L_L\) 只约束 \(h_L\)，**不**约束 \(c_{\mathrm{co}}\)  
6. 边重建 / \(L_H\) **只绑细码**；\(M_{\mathrm{co}}\gg\) 类数（防粗码坍成类 ID）

### 17.4 数据流

```text
x_i, A, t_i
  → 双通道 GNN（h_L ← 主要 N^S；h_L̄ ← N^S 低频 + N^D 高频/对比）
  → 可选 Linear_head(h_L) → L_L（弱）
  → SemanticVQ(h_L, t_i) → z_co, c_co     # 粗码 theme
  → r = h_L̄ − stopgrad(z_co)
  → VQ_fi(r) → z_fi, c_fi                   # 细码 structure（无 text）
  → Dec_fi + 边重建(z_fi) + L_H
  → 导出 → preprocess（theme 词 + struct 摘要）→ QLoRA
```

### 17.5 双通道 GNN（量化前，必做）

**\(h_L\)（语义通道）：** 只聚合同组邻居 \(N^S\)（低频），异组不进入 —— 对应 HeteroGen 式 (10)。

**\(h_{\bar L}\)（结构通道）：** \(N^S\) 低频平滑 + \(N^D\) 高频/对比（消息可含 \(-h_j\)）—— 对应式 (11)–(13)。

| 配置 | 预期 |
|------|------|
| 只拆两个 VQ，GNN 仍普通 GraphConv | \(h_{\bar L}\) 已抹平，L_H 事后补课，效果弱 |
| 双通道 + 层次 VQ + L_H | 异配缓解主路径 |

**Proxy（无 y）：** 同组 = 粗码相同或 \(\mathrm{cos}(t_i,t_j)>\tau\)；异组 = 否则。

### 17.6 粗码（Theme）

| 项 | 内容 |
|----|------|
| 内部 | \(c_{\mathrm{co}}\) + \(z_{\mathrm{co}}\) |
| 给 LLM | P_code_co / μ 的 **top 词**，禁止写类名 |
| 损失 | \(L_{\mathrm{co\_commit}}\) + \(L_{\mathrm{co\_sem}}\) + 可选 \(L_{\mathrm{co\_node\_rec}}\) |
| **禁止** | \(L_{\mathrm{co\_cls}}=\mathrm{CE}(c_{\mathrm{co}},y)\) |

**与 \(h_L\) / \(L_L\) / \(y\)：** 粗码与 \(h_L\) **同输入链路**；\(L_L\) 塑形连续 \(h_L\)（有判别力）；粗码含义由 **text/theme** 定，**不是** \(y\) 的离散化。  
防变类码：\(M_{\mathrm{co}}\gg\#\mathrm{class}\)、\(\lambda_L\) 小、无 \(L_{\mathrm{co\_cls}}\)、LLM 看词不看码号。

### 17.7 细码（Structure）

```text
r = h_L̄ − stopgrad(z_co)
z_fi, c_fi = VQ_fi(r)     # 不加 text
```

| 损失 | 作用 |
|------|------|
| \(L_{\mathrm{fi\_commit}}\) / \(L_{\mathrm{fi\_node\_rec}}\) | 量化稳定 + 还原 \(h_{\bar L}\) |
| \(L_{\mathrm{fi\_edge\_rec}}\) | \(\hat A\) 由 \(z_{\mathrm{fi}}\) 建（替换混合 z 的 `z·zᵀ`） |
| \(L_H\) | \((L_{\mathrm{inter}}-L_{\mathrm{intra}})^2\)，强迫异配边也重建好 |
| \(L_{\mathrm{intra\_co}}\)（可选） | 同粗码相邻 → 细码拉开（自监督） |

**边 Decode：** Phase1 仍可用线性 + 内积 + \(L_H\)（decode 结构不大改）。  
Phase2 可选双 decoder：`MLP_intra` / `MLP_inter` 分别拟合同配/异配边（替换全图 `z·zᵀ`）。粗码不参与边 decoder。

### 17.8 总损失

**完整版：**

\[
\begin{aligned}
L &= \lambda_{\mathrm{co},c} L_{\mathrm{co\_commit}} + \lambda_{\mathrm{co},s} L_{\mathrm{co\_sem}} + \lambda_{\mathrm{co},n} L_{\mathrm{co\_node\_rec}} \\
&+ \lambda_{\mathrm{fi},c} L_{\mathrm{fi\_commit}} + \lambda_{\mathrm{fi},n} L_{\mathrm{fi\_node\_rec}} + \lambda_{\mathrm{fi},e} L_{\mathrm{fi\_edge\_rec}} \\
&+ \lambda_H L_H + \lambda_D L_D + \lambda_M L_M + \lambda_L L_L \\
&+ \lambda_{\mathrm{tok}} L_{\mathrm{token}} + \lambda_{\mathrm{ico}} L_{\mathrm{intra\_co}}
\end{aligned}
\]

**Phase1 最小：** \(L_{\mathrm{co\_commit}}+L_{\mathrm{co\_sem}}+L_{\mathrm{fi\_*}}+ \lambda_H L_H + \lambda_D L_D\) + 双通道 GNN。

| 权重 | 初值 | 说明 |
|------|------|------|
| commit / sem / edge | ~1 | |
| node rec | 0.5~1 | 对齐 `lamb_node` |
| **λ_H** | 0.1~1 | 从小扫 |
| λ_D | 0.01~0.1 | |
| λ_M | 0.1~0.5 | 可选 |
| λ_L | ≤0.1 | 弱；只 \(h_L\) |
| λ_tok | 0.05 | **仅粗码**；现有 E5b，可关 |
| λ_ico | 0.1 | 可选，未实现 |

**两项可选辅助损失释义：**

| 损失 | 含义 | 代码 |
|------|------|------|
| \(L_{\mathrm{token}}\) | \(\mathrm{KL}(P_{\mathrm{target}}\|P_{\mathrm{pred}})\)：码向量预测词表分布，供 \(P_{\mathrm{code}}\) 选词 | ✅ `token_predictor.py` |
| \(L_{\mathrm{intra\_co}}\) | 同粗码邻居的细码距离 margin：同 theme 仍要结构可分 | ❌ 未实现 |

**实验配置：** A 纯自监督码本（无 \(L_L\)，proxy 分边） / B 弱 \(L_L\)（推荐） / C + `L_cls_full`（对齐原 L_CE ablation）。

### 17.9 LLM 序列化（与码本配套）

```text
[theme: neural, learning, networks, optimization]
[struct: center=F_42; hist=F_3:4, F_17:2, F_9:1]
Center: Title: ... Abstract: ...
Neighbor_1: ...   # 仅 text，无 struct
Question: Predict the category of the center paper.
```

禁止 `[class: Neural_Networks]` 作粗码显示（除非负对照）。

### 17.10 实施阶段与验收

| Phase | 内容 | 状态 |
|-------|------|------|
| **P0** | 双通道 GNN + 层次 VQ | ✅ `hierarchical_encoder.py` |
| **P1** | + \(L_H\) + \(L_D\) + 弱 \(L_L\) | ✅ 已接入训练损失 |
| **P2** | 新 JSONL + QLoRA | ❌ 未做 |
| **P3** | \(L_M\)、\(L_{\mathrm{intra\_co}}\)、双 decoder、SAGE | ❌ 未做 |

**必测指标：** 异配边同细码率；同配边同细码率（别崩）；按异类邻居比例分桶；粗码–类 NMI（不应≈1）；LLM ablation（full / theme-only / struct-only / nocode / 旧格式）。

### 17.11 改动落点

| 文件 | 改什么 |
|------|--------|
| `models/hierarchical_encoder.py` | **新增** DualChannelAgg + HierarchicalGCN |
| `models.py` | `Model` 支持 `hierarchical_vq` |
| `models/codebook_trainer.py` | labels 传入、双码本产物、分项 loss 日志 |
| `config.py` / `train_codebook.py` | CLI 与超参 |
| `preprocess_data.py` / `node_representation.py` | theme+struct JSONL — **未做** |
| `finetune_llm.py` | instruction 适配 — **未做** |

### 17.12 一句话收束

粗码对 \(h_L\) 做 text 对齐 SemanticVQ（theme）；细码对结构残差 VQ，边重建绑 \(z_{\mathrm{fi}}\) 并用 \(L_H\) 打同配偏置；HeteroGen 贡献解耦通道 + 分路径聚合 + 异配边正则，**不**贡献扩散生成，**不**把粗码变成类标签。
