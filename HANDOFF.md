# VQGraph 项目交接文档

> 更新日期：2026-05-28  
> 仓库：https://github.com/cnhhp/VQGraph  
> 规格说明：[`prompt.md`](prompt.md)

---

## 1. 项目目标

将引文网络图节点离散化为「结构 token + 文本 token」序列，经偏置欧拉游走序列化后，供大模型（LLM）做节点分类微调。

**Pipeline：** 模块1 结构码本 → 模块2 子图 → 模块3 离散化 → 模块4 序列化 → 模块5 LLM 微调

---

## 2. 环境与路径

| 项 | 值 |
|----|-----|
| Python | **3.9**（`prompt.md` / README 约定；本机 env：`vqgraph_py39` 3.9.25） |
| 本机项目路径 | `c:\Users\hhp\Desktop\组会\idea\VQGraph\VQGraph` |
| Conda（本机） | `D:\anaconda\envs\vqgraph_py39\python.exe` |
| 服务器 Conda | `2252895_vqgraph`（路径示例：`/home/power/anaconda3/envs/2252895_vqgraph`） |
| Git 最新提交 | `8bf01ef` — LLM pipeline + 文本 Cora + JSONL + 模块5 实现 |

### SSH（已配置 `~/.ssh/config`）

```text
Host vqgraph-gpu
    HostName 10.199.227.106
    User power
    Port 8122
```

连接：`ssh vqgraph-gpu`  
**注意：** 学校服务器 **HTTP/HTTPS 出网受限**，无法在服务器上 `huggingface-cli login` / 在线下模型。

---

## 3. 数据集（当前实验：文本 DGL Cora）

**优先自动加载：** `data/dataset/cora/` 存在时走文本版，否则回退 CPF `data/cora.npz`。

| 文件 | 说明 |
|------|------|
| `cora_graph.pth` | DGL 图；`feat` [2708,768]；`train/val/test_mask` → 140/500/1000 |
| `cora_text.pkl` | 每节点 `Title: ... Abstract: ...` |
| `cora_metadata.pth` | `categories` → `g.category_names` |

- 强制文本：`--data_source text`
- 强制 CPF BoW：`--data_source cpf`（需 `data/cora.npz`，**本地已删未提交**，远程仓库仍有）

**与 CPF 差异：** 768 维预计算嵌入 + 真实文本；SBERT 语义通道 384 维（`all-MiniLM-L6-v2`），与 `feat` 独立。

---

## 4. 模块进度

| 模块 | 状态 | 要点 |
|------|------|------|
| **1 结构码本** | ✅ + Token 预测辅助 | 语义偏置 VQ + **token_predictor**（KL 监督）；推理融合 `P_code` |
| **GCN 分类基线** | ✅ 脚本就绪 | `train_gcn_baseline.py` → `outputs/baseline_gcn/cora/seed_0/baseline_metrics.json`（无 VQ，仅 CE） |
| **2 子图** | ✅ | `SubgraphExtractor`，默认 k=1 |
| **3 节点离散化** | ✅ | TF-IDF + **P_code（token_predictor）** + MMR + 选词阶段停用词 |
| **4 序列化** | ✅ | `BiasedEulerSerializer`；α/β/γ=0.4/0.3/0.3 |
| **TextTokenbook** | ✅ | `codebook/filtered_tokenbook.npy`，V=13648 |
| **preprocess_data** | ✅ 全量 JSONL | `data/llm_finetune/` 140/500/1000；`manifest.json` |
| **5 LLM 微调** | ⚠️ 代码完成，正式训练未完成 | 本机 CPU 冒烟 Qwen2.5-0.5B 通过；**全量 QLoRA Llama-3 待做** |

---

## 5. 关键配置（`config.py`，阶段 A+B 默认）

```text
# 模块1 — Token 预测辅助（默认开启）
enable_token_predictor=True, lambda_token=0.05, token_pred_temperature=1.0
token_target_temperature=0.15, lambda_pred=0.5

# 阶段 B — 序列化
subgraph_k_hop=2, top_k_text_tokens=8, mmr_candidate_pool=96

# 阶段 A — QLoRA 正则
lora_r=8, lora_alpha=16, lora_dropout=0.1, finetune_lr=1e-4
qlora_epochs=2, warmup_ratio=0.06, max_seq_length=768

# 其它
tokenbook: filtered_tokenbook.npy, lambda_tfidf=0.5, batch=4, grad_accum=2
```

**损失（模块 1）：** `L_total = L_recon + L_commit + α·L_token`（`L_token = KL(P_target ‖ P_pred)`，P_target 来自 SBERT 余弦相似度）

**推理融合（模块 3）：** `score[t] = text_sim[t] × (1 + λ_tfidf·TF-IDF + λ_pred·P_code[t])`

**TF-IDF 统计时机：** 默认仅在 `epoch >= warmup_epochs + 1` 后更新 best checkpoint 并分配 node_codes 做 TF-IDF（warmup 阶段码分配不稳定，不参与统计）。可调 `--tfidf_stats_min_epoch`。

**重训 checklist（启用 token_predictor 后）：**
1. `train_codebook.py`（需 `codebook/filtered_tokenbook.npy`）
2. `--compute_tfidf` 重算 TF-IDF
3. `preprocess_data.py` 重生成 JSONL

**v1 旧数据（k=1, top_k=5）：** `data/llm_finetune/`（仍可用于对比）  
**v2 新数据（k=2, top_k=8）：** 需用当前 config 重跑 `preprocess_data.py` → `data/llm_finetune_v2/`

---

## 6. 码本与 JSONL 产物

### 结构码本（本地有效，未进 git）

```text
outputs/codebook/cora/GCN/seed_0/
  model.pth, codebook_embeddings.npz, semantic_centers.npz, node_codes.npz
  tfidf_stats.npz, tfidf_stats.vocab.json
  train_conf.json, config.json
```

### LLM 训练数据

```text
data/llm_finetune/          # v1：k=1, top_k=5（GitHub 已有）
data/llm_finetune_v2/       # v2：k=2, top_k=8（服务器本地生成，勿提交大文件）
  train.jsonl (140), val.jsonl (500), test.jsonl (1000)
  manifest.json
```

**JSONL 格式：** `instruction` + `input`（序列化子图文本）+ `output`（类名，如 `Neural_Networks`，来自 `resolve_node_class_name`）

**模块 4 示例（节点 0，停用词过滤后）：**

```text
0 <S_546> [nodes, walk, learning, networks, graph]
1 <S_907> [networks, processor, parallel, control, neural]
...
```

---

## 7. 模块 5 实现说明

**文件：** `models/llm_finetune.py`，入口 `finetune_llm.py`

- **训练：** 因果 LM + prompt 部分 label=-100；自定义 collator 做 padding
- **评估：** `generate` + `normalize_prediction` 与 gold 精确匹配
- **模式：** `--mode qlora | lora | both`；QLoRA 需 `bitsandbytes` + CUDA
- **离线模型：** `--base_model /path/to/local`（服务器无法访问 HF 时必须）

**本机冒烟（已通过）：**

```bash
python finetune_llm.py ... --base_model Qwen/Qwen2.5-0.5B-Instruct --mode lora \
  --max_samples 10 --lora_epochs 1 --device -1 --output_dir ./outputs/llm_smoke
```

产物：`outputs/llm_smoke/lora/`（1 epoch 准确率 0 属正常）

---

## 8. 常用命令

```bash
# 模块1 重训码本（文本 Cora，含 Token 预测辅助任务）
python train_codebook.py --dataset cora --data_root ./data --data_source text \
  --tokenbook_dir ./codebook --warmup_epochs 20 --lambda_semantic 0.1 \
  --lambda_token 0.05 --token_target_temperature 0.15 \
  --compute_tfidf --seed 0 --device 0 --console_log

# 禁用 Token 预测（与旧版行为一致）
python train_codebook.py ... --no_token_predictor

# 仅重算 TF-IDF
python train_codebook.py --tfidf_only --dataset cora --data_root ./data \
  --tokenbook_dir ./codebook --device 0

# 生成 JSONL v2（阶段 B，默认 k=2 top_k=8 → llm_finetune_v2）
python preprocess_data.py --dataset cora --data_root ./data --data_source text \
  --codebook_dir ./outputs/codebook/cora/GCN/seed_0 \
  --tokenbook_path ./codebook --device 0

# 生成 JSONL v1（旧版对比）
python preprocess_data.py ... --output_dir ./data/llm_finetune --k 1 --top_k 5

# GCN 正式基线（文本 Cora，与 LLM 相同 140/500/1000 划分）
python train_gcn_baseline.py --dataset cora --data_root ./data --device 0 --console_log --save_model

# 模块4 单节点冒烟
python -m models.serialization --codebook_dir ./outputs/codebook/cora/GCN/seed_0 \
  --tokenbook_path ./codebook --dataset cora --data_root ./data \
  --data_source text --node 0 --device 0

# LLM QLoRA 阶段 A+B（v2 数据 + 默认正则超参）
python finetune_llm.py \
  --train_jsonl ./data/llm_finetune_v2/train.jsonl \
  --val_jsonl ./data/llm_finetune_v2/val.jsonl \
  --test_jsonl ./data/llm_finetune_v2/test.jsonl \
  --base_model ~/huanghp_2252895/Meta-Llama-3-8B-Instruct \
  --mode qlora --device 0 \
  --output_dir ./outputs/llm_qlora_v2

# v1 基线对比（71% val 那次）
# --train_jsonl ./data/llm_finetune/... --output_dir ./outputs/llm_qlora
```

---

## 9. 基座模型下载（当前卡点）

| 方式 | 结果 |
|------|------|
| HuggingFace 官方 / 镜像 | 本机 **SSL 失败**（`TLS connection closed`），与是否热点无关时仍可能失败 |
| **魔搭 ModelScope** | 可访问；曾启动 `LLM-Research/Meta-Llama-3-8B-Instruct` → `D:\models\Meta-Llama-3-8B-Instruct` |

**本机下载状态（交接时）：** `D:\models\Meta-Llama-3-8B-Instruct` 约 **1.1GB / ~16GB**，**未完成**（仅部分分片）。需在本机续下后 `scp` 到服务器：

```powershell
# 魔搭续传（本机）
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('LLM-Research/Meta-Llama-3-8B-Instruct', local_dir=r'D:\models\Meta-Llama-3-8B-Instruct')"

scp -P 8122 -r D:\models\Meta-Llama-3-8B-Instruct vqgraph-gpu:~/models/
```

**安全：** 曾在对话中泄露 HF token，**必须撤销并换新**，勿再粘贴 token。

**备选：** `Qwen/Qwen2.5-7B-Instruct`（魔搭/HF 更易，24GB QLoRA 够用，无 Meta 审批）

---

## 10. 服务器正式微调清单（24GB + Python 3.9）

```bash
conda activate 2252895_vqgraph   # 或你的环境名
cd ~/VQGraph && git pull

pip install -r requirements.txt
pip install bitsandbytes   # QLoRA 必装

# 无需 HF 在线登录；使用本地模型目录
python finetune_llm.py \
  --train_jsonl ./data/llm_finetune/train.jsonl \
  --val_jsonl ./data/llm_finetune/val.jsonl \
  --test_jsonl ./data/llm_finetune/test.jsonl \
  --base_model ~/models/Meta-Llama-3-8B-Instruct \
  --mode qlora --qlora_epochs 5 --device 0 \
  --output_dir ./outputs/llm_qlora

# 建议先 pilot
# --max_samples 100 --qlora_epochs 1 --output_dir ./outputs/llm_pilot
```

`screen -S qlora` 防断线。

---

## 11. Git 状态说明

**已 push（`8bf01ef`）：** 代码 + `data/llm_finetune/*.jsonl` + README 模块5

**未提交（本地）：**

- `data/dataset/cora/`（~1.2GB，太大）
- `outputs/`（`.gitignore`）
- 删除的 `data/*.npz`（本地删了，远程仍有）
- `data/llm_finetune_smoke*`、冒烟日志

---

## 12. 重要文件索引

| 文件 | 作用 |
|------|------|
| `prompt.md` | 完整规格 |
| `config.py` | 全局超参 |
| `graph_utils.py` | `load_graph_data`、`load_text_dgl_dataset`、`resolve_node_class_name`、停用词 |
| `train_codebook.py` | 模块1 入口 |
| `models/token_predictor.py` | Token 预测头 + KL 损失 |
| `preprocess_data.py` | JSONL 主流水线 |
| `models/node_representation.py` | MMR + TF-IDF 选词 |
| `models/serialization.py` | 模块4 |
| `models/llm_finetune.py` | 模块5 |
| `finetune_llm.py` | 微调入口 |
| `train_gcn_baseline.py` | 标准 GCN 节点分类基线（无 VQ） |
| `models/gcn_baseline.py` | 基线模型定义 |
| `requirements.txt` | 含 transformers/peft/datasets/accelerate；bitsandbytes 注释为可选 |

---

## 13. 新对话建议接续任务

1. **确认并完成** `D:\models\Meta-Llama-3-8B-Instruct` 魔搭下载 → `scp` 到 `vqgraph-gpu:~/models/`
2. 服务器 **`pip install bitsandbytes`**，`--max_samples 100` pilot 后全量 QLoRA
3. 分析 `outputs/llm_qlora/qlora/finetune_metrics.json` 与 val 错误样例
4. 可选：评估加速（批量 generate）、README 补充离线模型说明
5. 可选：git 提交本地 `graph_utils` 等若还有未 push 改动（当前 main 已较新）

---

## 14. 新对话开场可复制

```text
项目：VQGraph @ c:\Users\hhp\Desktop\组会\idea\VQGraph\VQGraph
GitHub: https://github.com/cnhhp/VQGraph，请先读 HANDOFF.md 与 prompt.md
数据：文本 DGL Cora，JSONL 已在 data/llm_finetune/
码本：outputs/codebook/cora/GCN/seed_0/（本地，未上传）
模块5 已实现，待服务器 QLoRA：模型在下载/上传 ~/models/Meta-Llama-3-8B-Instruct
服务器：ssh vqgraph-gpu（10.199.227.106:8122，user power），无外网 HF
请继续：[你的具体任务]
```
