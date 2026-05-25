"""
入口：生成大模型微调 JSONL（模块2 + 3 + 4 串联）。
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from config import get_config, reset_config
from graph_utils import load_graph_data, set_seed, setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="生成指令微调 JSONL 数据")
    p.add_argument("--dataset", type=str, default="cora")
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--codebook_dir", type=str, required=True, help="模块1 产物目录")
    p.add_argument("--tokenbook_path", type=str, default="./codebook")
    p.add_argument("--tfidf_path", type=str, default=None)
    p.add_argument("--output_dir", type=str, default="./data/llm_finetune")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_nodes", type=int, default=None, help="调试用：最多处理节点数")
    return p.parse_args()


def build_samples_for_split(
    split_name: str,
    node_indices: list,
    cfg,
) -> list:
    """
    对 split 中每个目标节点 v_t：
    子图提取 -> 节点离散化 -> 序列化 -> InstructionSample
    """
    from models.llm_finetune import InstructionSample

    raise NotImplementedError(f"build_samples_for_split({split_name}) 待实现")


def main() -> None:
    args = parse_args()
    reset_config()
    cfg = get_config()
    cfg.dataset_name = args.dataset
    cfg.data_root = Path(args.data_root)
    cfg.codebook_checkpoint_dir = Path(args.codebook_dir)
    cfg.tokenbook_path = Path(args.tokenbook_path)

    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(__name__, log_file=out_dir / "preprocess.log")

    from models.structural_codebook import StructuralCodebookTrainer, TFIDFComputer
    from models.subgraph_extraction import SubgraphExtractor
    from models.node_representation import NodeRepresentationTokenizer
    from models.serialization import BiasedEulerSerializer
    from text_tokenizers.text_tokenbook import TextTokenbook

    bundle = load_graph_data(cfg.dataset_name, root=cfg.data_root)
    _, edge_index, labels, train_mask, val_mask, test_mask, _ = bundle

    # artifacts = StructuralCodebookTrainer.load_artifacts(...)
    tokenbook = TextTokenbook.load(cfg.tokenbook_path)
    logger.info("Loaded tokenbook: V=%d", len(tokenbook))
    # tfidf = TFIDFComputer.load(...) if args.tfidf_path else None

    split_masks = {
        "train": train_mask,
        "val": val_mask,
        "test": test_mask,
    }

    for split in args.splits:
        if split not in split_masks:
            logger.warning("Unknown split %s, skip.", split)
            continue
        logger.info("Processing split: %s", split)
        # samples = build_samples_for_split(...)
        # InstructionDatasetBuilder.save_jsonl(samples, out_dir / f"{split}.jsonl")
        raise NotImplementedError("preprocess_data 主流水线待实现")

    logger.info("Done. Output dir: %s", out_dir)


if __name__ == "__main__":
    main()
