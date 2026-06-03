# VQGraph 项目交接文档

> 更新日期：2026-06-02  
> 仓库：https://github.com/cnhhp/VQGraph  
> 规格说明：[`prompt.md`](prompt.md)

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

## 3. 数据集（当前实验：文本 DGL Cora）

**优先自动加载：** `data/dataset/cora/` 存在时走文本版，否则回退 CPF `data/cora.npz`。

| 文件 | 说明 |
|------|------|
| `cora_graph.pth` | DGL 图；`feat` [2708,768]；`train/val/test_mask` → 140/500/1000 |
| `cora_text.pkl` | 每节点 `Title: ... Abstract: ...` |
| `cora_metadata.pth` | `categories` → `g.category_names` |

- 强制文本：`--data_source text`
- `data/dataset/cora/`（~1.2GB）**未进 git**，需本机/服务器各自保留

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
| **5 LLM 微调** | ⚠️ | 代码完成；**服务器 Llama-3 QLoRA 多组对比待汇总** |

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
token_selector_kl_weight=0.3        # KL 正则（加大，贴近 s0）
token_selector_entropy_weight=0.01
token_selector_vtext_dropout=0.3
filter_noise_subwords_at_selection=True  # 推理时过滤 PDF/子词噪声

# 模块5 — config 默认（服务器正式跑可覆盖）
lora_r=8, lora_alpha=16, lora_dropout=0.1, finetune_lr=1e-4
qlora_epochs=2, max_seq_length=768
```

**服务器正式 QLoRA 超参（用户指定）：** `qlora_epochs=5, lora_r=16, lora_alpha=32, lora_dropout=0.05, finetune_lr=2e-4, max_seq_length=1024`

**QLoRA 可复现性：** 固定 `--seed 42` + 相同数据 → train_loss 完全一致（正常）。

---

## 7. 码本与 JSONL 产物矩阵

> 均在 `outputs/experiments/` 与 `data/`，**未 commit**；同步服务器需 `scp`。

| 系列 | E1 目录 | E5b+TF-IDF | seed | E1 有 L_token? | JSONL λ=0.05 | JSONL λ=0 |
|------|---------|------------|------|----------------|-------------|-----------|
| 原版 | `e1_tau003` | `e5b_code_level` | 0 | ✅ | `llm_finetune_e5b` | `llm_finetune_v2` |
| s1 | `e1_tau003_s1` | `e5b_code_level_s1` | 1 | ✅ | `llm_finetune_e5b_s1` | `llm_finetune_nocode_s1` |
| **no_ltoken** | `e1_no_ltoken` | `e5b_no_ltoken` | 1 | ❌ | `llm_finetune_e5b_no_ltoken` | `llm_finetune_nocode_no_ltoken` |
| **learned** | — | `e5b_no_ltoken` + TokenSelector | 42 | — | `llm_finetune_e5b_learned` | — |

路径示例：

```text
outputs/experiments/e5b_no_ltoken/cora/GCN/seed_1/
  model.pth, codebook_embeddings.npz, semantic_centers.npz, node_codes.npz
  tfidf_stats.npz, tfidf_stats.vocab.json

data/llm_finetune_e5b_no_ltoken/
  train.jsonl (140), val.jsonl (500), test.jsonl (1000), manifest.json
```

**no_ltoken 指标：** E1 best ep86 val **75.4%**；E5b L_token **0.899**

**实验汇总：** `outputs/experiments/results.json`

**JSONL 格式：** `instruction` + `input`（层级化子图序列）+ `output`（类名）

---

## 8. JSONL 词频分析要点（no_ltoken 对）

- Top 词：`learning`(6%) > `algorithms`(4%) > `classification` > `neural` — 跨类泛化词占比高  
- **P_code(λ=0.05)** 推高：classification、neural、inference、prediction  
- **λ=0** 更多：optimized、gene、recognition、patterns  
- 噪声：minipage、episodio 等 PDF 解析残留  

**选词质量改进方向（部分已实现）：** TokenSelector 可学习重排 + s0 候选池约束 + KL/熵正则 + v_text dropout；其余可选：text_sim 阈值、TF-IDF max_df、洗 tokenbook

### TokenSelector 流程

```text
训练：s0 = 固定公式 → 取 s0 Top-256 候选池
      → TokenSelector 重排（池外 -inf）→ Gumbel-Softmax
      → v_text（训练时 dropout）→ NodeClassifier
      → L = L_cls + λ_kl·KL(w||softmax(s0)) − λ_ent·H(w)
推理：s0 → TokenSelector（池外 -inf）→ MMR → Top-k
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

### 训练 TokenSelector

```bash
python train_token_selector.py --dataset cora --data_root ./data --data_source text \
  --codebook_dir ./outputs/experiments/e5b_no_ltoken/cora/GCN/seed_1 \
  --tokenbook_path ./codebook \
  --output_dir ./outputs/token_selector/e5b_no_ltoken \
  --lambda_pred 0.05 --p_code_normalize max \
  --epochs 50 --batch_size 32 --lr 1e-3 --seed 42 --device 0 --console_log
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

### 服务器 QLoRA（推荐对比 no_ltoken 对）

```bash
cd ~/huanghp_2252895/VQGraph && conda activate 2252895_vqgraph

# 有 P_code
python finetune_llm.py \
  --train_jsonl ./data/llm_finetune_e5b_no_ltoken/train.jsonl \
  --val_jsonl ./data/llm_finetune_e5b_no_ltoken/val.jsonl \
  --test_jsonl ./data/llm_finetune_e5b_no_ltoken/test.jsonl \
  --base_model ~/huanghp_2252895/Meta-Llama-3-8B-Instruct \
  --mode qlora --qlora_epochs 5 --device 0 \
  --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 --finetune_lr 2e-4 \
  --max_seq_length 1024 \
  --output_dir ./outputs/llm_qlora_e5b_no_ltoken_ep5

# 无 P_code 对照
python finetune_llm.py \
  --train_jsonl ./data/llm_finetune_nocode_no_ltoken/train.jsonl \
  --val_jsonl ./data/llm_finetune_nocode_no_ltoken/val.jsonl \
  --test_jsonl ./data/llm_finetune_nocode_no_ltoken/test.jsonl \
  --base_model ~/huanghp_2252895/Meta-Llama-3-8B-Instruct \
  --mode qlora --qlora_epochs 5 --device 0 \
  --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 --finetune_lr 2e-4 \
  --max_seq_length 1024 \
  --output_dir ./outputs/llm_qlora_nocode_no_ltoken_ep5
```

`screen -S qlora` 防断线。指标：`outputs/llm_qlora_*/qlora/finetune_metrics.json`

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
| `scripts/run_phase_ab_server.sh` | 服务器 phase A+B 模板 |
| `outputs/experiments/results.json` | E1–E6 实验汇总 |

---

## 13. 待续任务

1. 服务器 `git pull` + scp 码本 + JSONL → 多组 QLoRA 对比并汇总 val/test acc  
2. （可选）选词质量：text_sim 阈值、TF-IDF max_df、洗 tokenbook  
3. （可选）Llama 本机续传 → scp 服务器  

---

## 14. 新对话开场可复制

```text
项目：VQGraph @ c:\Users\hhp\Desktop\组会\idea\VQGraph\VQGraph
GitHub: https://github.com/cnhhp/VQGraph（main @ 75c3aae）
服务器：ssh vqgraph-gpu → ~/huanghp_2252895/VQGraph

请先读 HANDOFF.md 与 config.py。

已完成：
- P_code 根因 + E5b 配方（factorized + 码本级 KL + normalize=max + λ_pred=0.05）
- 三组码本+JSONL：seed0 / seed1(s1) / seed1(no_ltoken)；每组均有 λ=0.05 与 λ=0 两版
- 推荐对比：llm_finetune_e5b_no_ltoken vs llm_finetune_nocode_no_ltoken
- 码本：outputs/experiments/e5b_no_ltoken/cora/GCN/seed_1/
- JSONL/outputs 未 commit，需 scp 到服务器

待做：[你的任务]
```
