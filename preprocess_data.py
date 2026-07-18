"""
入口：生成大模型微调 JSONL（模块2 + 3 + 4 串联）。
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from config import get_config, reset_config
from graph_utils import (
    extract_sentence_bert_embeddings,
    load_graph_data,
    resolve_node_class_name,
    set_seed,
    setup_logging,
)

logger = logging.getLogger(__name__)


@dataclass
class PreprocessPipeline:
    """模块 2/3/4 与码本产物的共享上下文，避免重复加载。"""

    graph: Any
    node_features: torch.Tensor
    text_embeddings: torch.Tensor
    labels: torch.Tensor
    extractor: Any
    tokenizer: Any
    serializer: Any
    dataset_name: str
    instruction: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="生成指令微调 JSONL 数据")
    p.add_argument("--dataset", type=str, default="cora")
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--codebook_dir", type=str, required=True, help="模块1 产物目录")
    p.add_argument("--tokenbook_path", type=str, default="./codebook")
    p.add_argument("--tfidf_path", type=str, default=None, help="默认 codebook_dir/tfidf_stats.npz")
    p.add_argument(
        "--output_dir",
        type=str,
        default="./data/llm_finetune_v2",
        help="默认 v2（k=2, top_k=8）；v1 见 ./data/llm_finetune",
    )
    p.add_argument("--top_k", type=int, default=None, help="文本 token 数，默认读 config.top_k_text_tokens")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split_idx", type=int, default=0)
    p.add_argument("--labelrate_train", type=int, default=20)
    p.add_argument("--labelrate_val", type=int, default=30)
    p.add_argument("--k", type=int, default=None, help="子图 hop 数，默认读 config")
    p.add_argument("--device", type=int, default=0, help="-1 表示 CPU")
    p.add_argument("--no_tfidf", action="store_true")
    p.add_argument("--max_nodes", type=int, default=None, help="调试用：每个 split 最多处理节点数")
    p.add_argument("--sentence_bert", type=str, default=None)
    p.add_argument(
        "--data_source",
        type=str,
        default=None,
        choices=["auto", "text", "cpf"],
        help="数据来源：auto=优先 data/dataset/{name}/ 文本版",
    )
    p.add_argument("--node_text_csv", type=str, default=None, help="可选，覆盖部分节点文本")
    p.add_argument(
        "--lambda_pred",
        type=float,
        default=None,
        help="P_code 融合强度；0 表示禁用 P_code（仅 TF-IDF + text_sim）",
    )
    p.add_argument(
        "--p_code_normalize",
        type=str,
        default=None,
        choices=["none", "max", "minmax"],
        help="推理时 P_code 归一化方式",
    )
    p.add_argument(
        "--token_selector_checkpoint",
        type=str,
        default=None,
        help="可选：训练好的 TokenSelector checkpoint（best.pth）",
    )
    p.add_argument(
        "--struct_token_mode",
        type=str,
        default=None,
        choices=["id", "pcode_supplement", "pcode_replace", "struct_summary", "projected_vector"],
        help="结构 token 展示模式；projected_vector 需 --projected_codebook_dir",
    )
    p.add_argument(
        "--projected_codebook_dir",
        type=str,
        default=None,
        help="PCA 投影码本目录（struct_token_mode=projected_vector 时必需）",
    )
    return p.parse_args()


def build_pipeline(
    args: argparse.Namespace,
    cfg,
    graph: Any,
    node_features: torch.Tensor,
    labels: torch.Tensor,
    text_dict: Dict[int, str],
) -> PreprocessPipeline:
    from models.codebook_trainer import CodebookTrainer, TFIDFComputer
    from models.llm_finetune import instruction_for_struct_mode
    from models.node_representation import NodeRepresentationTokenizer
    from models.serialization import BiasedEulerSerializer
    from models.subgraph_extraction import SubgraphExtractor
    from text_tokenizers.text_tokenbook import TextTokenbook

    device = torch.device(
        "cpu"
        if args.device < 0
        else (f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    )

    codebook_dir = Path(args.codebook_dir)
    artifacts = CodebookTrainer.load_artifacts(codebook_dir, device)

    tfidf = None
    if not args.no_tfidf:
        tfidf_path = Path(args.tfidf_path) if args.tfidf_path else codebook_dir / "tfidf_stats.npz"
        if tfidf_path.exists():
            tfidf = TFIDFComputer.load(tfidf_path)
            logger.info("Loaded TF-IDF from %s", tfidf_path)
        else:
            logger.warning("TF-IDF not found at %s; text selection uses similarity only.", tfidf_path)

    sbert = args.sentence_bert or cfg.sentence_bert_model
    tokenbook = TextTokenbook.load(
        Path(args.tokenbook_path),
        model_name=sbert,
        device=device,
    )
    logger.info("Loaded tokenbook: V=%d", len(tokenbook))

    text_emb = extract_sentence_bert_embeddings(
        text_dict,
        model_name=sbert,
        device=device,
    )

    if args.k is not None:
        cfg.subgraph_k_hop = args.k

    if args.p_code_normalize is not None:
        cfg.p_code_normalize = args.p_code_normalize
    if args.token_selector_checkpoint:
        cfg.token_selector_checkpoint = Path(args.token_selector_checkpoint)
    if args.struct_token_mode is not None:
        cfg.struct_token_mode = args.struct_token_mode
    if args.projected_codebook_dir is not None:
        cfg.projected_codebook_dir = Path(args.projected_codebook_dir)
    elif cfg.struct_token_mode == "projected_vector":
        default_proj = codebook_dir / "projected_k8"
        if default_proj.exists():
            cfg.projected_codebook_dir = default_proj
            logger.info("Using default projected_codebook_dir: %s", default_proj)

    extractor = SubgraphExtractor(cfg)
    token_selector = None
    if getattr(cfg, "token_selector_checkpoint", None):
        from models.token_selector import load_token_selector

        token_selector = load_token_selector(cfg.token_selector_checkpoint, device)
        logger.info("Loaded TokenSelector from %s", cfg.token_selector_checkpoint)

    tokenizer = NodeRepresentationTokenizer(
        artifacts=artifacts,
        tokenbook=tokenbook,
        tfidf=tfidf,
        cfg=cfg,
        token_selector=token_selector,
    )
    serializer = BiasedEulerSerializer(cfg)

    return PreprocessPipeline(
        graph=graph,
        node_features=node_features,
        text_embeddings=text_emb,
        labels=labels,
        extractor=extractor,
        tokenizer=tokenizer,
        serializer=serializer,
        dataset_name=cfg.dataset_name,
        instruction=instruction_for_struct_mode(cfg.struct_token_mode),
    )


def build_sample_for_node(
    node_id: int,
    pipeline: PreprocessPipeline,
) -> "InstructionSample":
    from models.llm_finetune import InstructionSample

    subgraph = pipeline.extractor.extract_from_dgl_graph(
        pipeline.graph,
        node_id,
        k=pipeline.extractor.k,
    )
    tokenized = pipeline.tokenizer.tokenize_subgraph(
        subgraph,
        pipeline.graph,
        pipeline.node_features,
        pipeline.text_embeddings,
    )
    result = pipeline.serializer.serialize(
        subgraph,
        tokenized,
        pipeline.text_embeddings,
    )

    class_name = resolve_node_class_name(
        pipeline.dataset_name,
        node_id,
        pipeline.labels,
        pipeline.graph,
    )

    return InstructionSample(
        instruction=pipeline.instruction,
        input=result.text,
        output=class_name,
    )


def build_samples_for_split(
    split_name: str,
    node_indices: List[int],
    pipeline: PreprocessPipeline,
    max_nodes: Optional[int] = None,
) -> List:
    """
    对 split 中每个目标节点 v_t：
    子图提取 -> 节点离散化 -> 序列化 -> InstructionSample
    """
    from models.llm_finetune import InstructionSample

    if max_nodes is not None:
        node_indices = node_indices[:max_nodes]

    samples: List[InstructionSample] = []
    total = len(node_indices)

    try:
        from tqdm import tqdm

        iterator = tqdm(node_indices, desc=f"{split_name}", unit="node")
    except ImportError:
        iterator = node_indices
        logger.info("Processing %s: %d nodes (install tqdm for progress bar)", split_name, total)

    for i, node_id in enumerate(iterator):
        try:
            sample = build_sample_for_node(node_id, pipeline)
            samples.append(sample)
        except Exception:
            logger.exception("Failed on node %d in split %s", node_id, split_name)
            raise
        if not hasattr(iterator, "set_postfix") and (i + 1) % 50 == 0:
            logger.info("%s: %d / %d", split_name, i + 1, total)

    logger.info("Split %s: built %d samples", split_name, len(samples))
    return samples


def main() -> None:
    args = parse_args()
    reset_config()
    cfg = get_config()
    cfg.dataset_name = args.dataset
    cfg.data_root = Path(args.data_root)
    cfg.codebook_checkpoint_dir = Path(args.codebook_dir)
    cfg.tokenbook_path = Path(args.tokenbook_path)
    if args.sentence_bert:
        cfg.sentence_bert_model = args.sentence_bert
    if args.data_source is not None:
        cfg.data_source = None if args.data_source == "auto" else args.data_source
    if args.node_text_csv:
        cfg.node_text_csv = Path(args.node_text_csv)
    if args.top_k is not None:
        cfg.top_k_text_tokens = args.top_k
    if args.lambda_pred is not None:
        cfg.lambda_pred = args.lambda_pred
    if args.p_code_normalize is not None:
        cfg.p_code_normalize = args.p_code_normalize
    if args.struct_token_mode is not None:
        cfg.struct_token_mode = args.struct_token_mode
    if args.projected_codebook_dir is not None:
        cfg.projected_codebook_dir = Path(args.projected_codebook_dir)

    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(__name__, log_file=out_dir / "preprocess.log")

    from models.llm_finetune import InstructionDatasetBuilder

    g, feats, labels, idx_train, idx_val, idx_test, text_dict = load_graph_data(
        cfg.dataset_name,
        root=cfg.data_root,
        node_text_path=cfg.node_text_csv,
        seed=args.seed,
        labelrate_train=args.labelrate_train,
        labelrate_val=args.labelrate_val,
        split_idx=args.split_idx,
        data_source=cfg.data_source,
        text_dataset_subdir=cfg.text_dataset_subdir,
    )

    split_indices: Dict[str, List[int]] = {
        "train": idx_train.tolist(),
        "val": idx_val.tolist(),
        "test": idx_test.tolist(),
    }

    pipeline = build_pipeline(args, cfg, g, feats, labels, text_dict)

    manifest: Dict[str, Any] = {
        "dataset": cfg.dataset_name,
        "data_root": str(cfg.data_root),
        "codebook_dir": str(args.codebook_dir),
        "tokenbook_path": str(args.tokenbook_path),
        "data_source": args.data_source or "auto",
        "serialization": {
            "subgraph_k_hop": cfg.subgraph_k_hop,
            "top_k_text_tokens": cfg.top_k_text_tokens,
            "mmr_candidate_pool": cfg.mmr_candidate_pool,
            "max_seq_length_recommended": 768,
            "lambda_pred": cfg.lambda_pred,
            "p_code_normalize": cfg.p_code_normalize,
            "lambda_tfidf": cfg.lambda_tfidf,
            "struct_token_mode": cfg.struct_token_mode,
            "projected_codebook_dir": (
                str(cfg.projected_codebook_dir)
                if getattr(cfg, "projected_codebook_dir", None)
                else None
            ),
            "token_selector_checkpoint": (
                str(cfg.token_selector_checkpoint)
                if getattr(cfg, "token_selector_checkpoint", None)
                else None
            ),
        },
        "splits": {},
    }
    t_all = time.perf_counter()

    for split in args.splits:
        if split not in split_indices:
            logger.warning("Unknown split %s, skip.", split)
            continue

        node_ids = split_indices[split]
        logger.info("Processing split %s (%d nodes)", split, len(node_ids))

        t0 = time.perf_counter()
        samples = build_samples_for_split(
            split,
            node_ids,
            pipeline,
            max_nodes=args.max_nodes,
        )
        out_path = out_dir / f"{split}.jsonl"
        InstructionDatasetBuilder.save_jsonl(samples, out_path)
        elapsed = time.perf_counter() - t0
        manifest["splits"][split] = {
            "path": str(out_path),
            "num_samples": len(samples),
            "num_nodes_requested": len(node_ids),
            "elapsed_sec": round(elapsed, 2),
        }
        logger.info(
            "Split %s finished in %.1fs (%.2f nodes/s)",
            split,
            elapsed,
            len(samples) / elapsed if elapsed > 0 else 0.0,
        )

    manifest["total_elapsed_sec"] = round(time.perf_counter() - t_all, 2)
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    logger.info("Wrote manifest to %s", manifest_path)
    logger.info("Done. Output dir: %s", out_dir)


if __name__ == "__main__":
    main()
