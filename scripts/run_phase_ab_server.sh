#!/usr/bin/env bash
# 阶段 A+B：在服务器上生成 v2 JSONL 并启动 QLoRA（需已有码本与 Cora 数据）
set -euo pipefail

ROOT="${VQGRAPH_ROOT:-$HOME/huanghp_2252895/VQGraph}"
CODEBOOK="${CODEBOOK_DIR:-$ROOT/outputs/codebook/cora/GCN/seed_0}"
BASE_MODEL="${BASE_MODEL:-$HOME/huanghp_2252895/Meta-Llama-3-8B-Instruct}"

cd "$ROOT"
conda activate "${CONDA_ENV:-2252895_vqgraph}"

echo "=== Phase B: preprocess → data/llm_finetune_v2 ==="
python preprocess_data.py \
  --dataset cora \
  --data_root ./data \
  --data_source text \
  --codebook_dir "$CODEBOOK" \
  --tokenbook_path ./codebook \
  --device 0

echo "=== Phase A: QLoRA (config defaults: epoch=2, dropout=0.1, lr=1e-4, r=8) ==="
python finetune_llm.py \
  --train_jsonl ./data/llm_finetune_v2/train.jsonl \
  --val_jsonl ./data/llm_finetune_v2/val.jsonl \
  --test_jsonl ./data/llm_finetune_v2/test.jsonl \
  --base_model "$BASE_MODEL" \
  --mode qlora \
  --device 0 \
  --output_dir ./outputs/llm_qlora_v2

echo "Done. Metrics: $ROOT/outputs/llm_qlora_v2/qlora/finetune_metrics.json"
