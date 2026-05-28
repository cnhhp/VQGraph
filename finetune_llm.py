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
    p.add_argument("--device", type=int, default=0, help="-1 表示 CPU")
    p.add_argument("--max_samples", type=int, default=None, help="每个 split 最多取 N 条（冒烟）")
    p.add_argument("--skip_qlora", action="store_true", help="both 模式下跳过 QLoRA")
    p.add_argument("--qlora_threshold", type=float, default=None, help="QLoRA val 准确率门槛")
    p.add_argument("--qlora_epochs", type=int, default=None)
    p.add_argument("--lora_epochs", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None, help="同时覆盖 qlora/lora 单模式 epoch")
    p.add_argument("--max_seq_length", type=int, default=None)
    p.add_argument("--lora_r", type=int, default=None, help="LoRA rank（默认读 config）")
    p.add_argument("--lora_alpha", type=int, default=None)
    p.add_argument("--lora_dropout", type=float, default=None)
    p.add_argument("--finetune_lr", type=float, default=None)
    p.add_argument("--warmup_ratio", type=float, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    reset_config()
    cfg = get_config()
    if args.base_model:
        cfg.base_model_name = args.base_model
    cfg.use_qlora = args.mode in ("qlora", "both")
    if args.qlora_threshold is not None:
        cfg.qlora_val_acc_threshold = args.qlora_threshold
    if args.max_seq_length is not None:
        cfg.max_seq_length = args.max_seq_length
    if args.qlora_epochs is not None:
        cfg.qlora_epochs = args.qlora_epochs
    if args.lora_epochs is not None:
        cfg.lora_epochs = args.lora_epochs
    if args.epochs is not None:
        if args.mode == "qlora":
            cfg.qlora_epochs = args.epochs
        elif args.mode == "lora":
            cfg.lora_epochs = args.epochs
    if args.lora_r is not None:
        cfg.lora_r = args.lora_r
    if args.lora_alpha is not None:
        cfg.lora_alpha = args.lora_alpha
    if args.lora_dropout is not None:
        cfg.lora_dropout = args.lora_dropout
    if args.finetune_lr is not None:
        cfg.finetune_lr = args.finetune_lr
    if args.warmup_ratio is not None:
        cfg.warmup_ratio = args.warmup_ratio

    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(__name__, log_file=out_dir / "finetune.log")

    from models.llm_finetune import LLMFinetuner

    device = "cpu" if args.device < 0 else args.device
    finetuner = LLMFinetuner(cfg, device=device)
    train_path = Path(args.train_jsonl)
    val_path = Path(args.val_jsonl)
    test_path = Path(args.test_jsonl)

    qlora_epochs = args.qlora_epochs or args.epochs
    lora_epochs = args.lora_epochs or args.epochs

    if args.mode == "qlora":
        acc = finetuner.run_qlora_phase(
            train_path,
            val_path,
            out_dir / "qlora",
            max_samples=args.max_samples,
            num_epochs=qlora_epochs,
        )
        logger.info("QLoRA val accuracy: %.4f", acc)
    elif args.mode == "lora":
        acc = finetuner.run_lora_phase(
            train_path,
            val_path,
            test_path,
            out_dir / "lora",
            max_samples=args.max_samples,
            num_epochs=lora_epochs,
        )
        logger.info("LoRA test accuracy: %.4f", acc)
    else:
        metrics = finetuner.run_pipeline(
            train_path,
            val_path,
            test_path,
            out_dir,
            skip_qlora=args.skip_qlora,
            max_samples=args.max_samples,
            qlora_epochs=qlora_epochs,
            lora_epochs=lora_epochs,
        )
        logger.info("Pipeline metrics: %s", metrics)


if __name__ == "__main__":
    main()
