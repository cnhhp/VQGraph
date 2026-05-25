"""
入口：大模型微调（模块5）— QLoRA 验证 + 标准 LoRA。
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from config import get_config, reset_config
from graph_utils import set_seed, setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="微调大模型（QLoRA / LoRA）")
    p.add_argument("--train_jsonl", type=str, required=True)
    p.add_argument("--val_jsonl", type=str, required=True)
    p.add_argument("--test_jsonl", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="./outputs/llm")
    p.add_argument("--base_model", type=str, default=None)
    p.add_argument("--mode", choices=["qlora", "lora", "both"], default="both")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    reset_config()
    cfg = get_config()
    if args.base_model:
        cfg.base_model_name = args.base_model
    cfg.use_qlora = args.mode in ("qlora", "both")

    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(__name__, log_file=out_dir / "finetune.log")

    from models.llm_finetune import LLMFinetuner

    finetuner = LLMFinetuner(cfg)
    train_path = Path(args.train_jsonl)
    val_path = Path(args.val_jsonl)
    test_path = Path(args.test_jsonl)

    if args.mode == "qlora":
        acc = finetuner.run_qlora_phase(train_path, val_path, out_dir / "qlora")
        logger.info("QLoRA val accuracy: %.4f", acc)
    elif args.mode == "lora":
        acc = finetuner.run_lora_phase(
            train_path, val_path, test_path, out_dir / "lora"
        )
        logger.info("LoRA test accuracy: %.4f", acc)
    else:
        metrics = finetuner.run_pipeline(
            train_path, val_path, test_path, out_dir, skip_qlora=False
        )
        logger.info("Pipeline metrics: %s", metrics)

    # 当前为脚手架，调用将触发 NotImplementedError
    logger.info("Finetune entry ready; implement LLMFinetuner for execution.")


if __name__ == "__main__":
    main()
