"""
入口：训练结构码本（模块1）+ 可选离线 TF-IDF。
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from config import get_config, reset_config
from graph_utils import (
    extract_sentence_bert_embeddings,
    get_device,
    load_graph_data,
    set_seed,
    setup_logging,
)
from models.codebook_trainer import CodebookTrainer, TFIDFComputer
from utils import check_writable

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="训练结构码本（模块1，语义偏置 VQ）")
    p.add_argument("--dataset", type=str, default="cora")
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--output_dir", type=str, default="./outputs/codebook")
    p.add_argument("--node_text_csv", type=str, default=None)
    p.add_argument("--teacher", type=str, default="GCN", choices=["GCN", "SAGE"])
    p.add_argument("--model_config_path", type=str, default="./train.conf.yaml")
    p.add_argument("--lambda_semantic", type=float, default=None)
    p.add_argument("--warmup_epochs", type=int, default=None)
    p.add_argument("--codebook_size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--device", type=int, default=0, help="-1 for CPU")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--labelrate_train", type=int, default=20)
    p.add_argument("--labelrate_val", type=int, default=30)
    p.add_argument("--lamb_node", type=float, default=0.001)
    p.add_argument("--lamb_edge", type=float, default=0.03)
    p.add_argument("--norm_type", type=str, default="none")
    p.add_argument("--dropout_ratio", type=float, default=None)
    p.add_argument("--hidden_dim", type=int, default=None)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--fan_out", type=str, default="5,5")
    p.add_argument("--eval_interval", type=int, default=1)
    p.add_argument("--sentence_bert", type=str, default=None)
    p.add_argument("--compute_tfidf", action="store_true")
    p.add_argument(
        "--tokenbook_dir",
        type=str,
        default="./codebook",
        help="预置文本 tokenbook 目录（--compute_tfidf 时使用）",
    )
    p.add_argument("--console_log", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    reset_config()
    cfg = get_config()
    cfg.dataset_name = args.dataset
    cfg.data_root = Path(args.data_root)
    cfg.output_root = Path(args.output_dir)
    if args.lambda_semantic is not None:
        cfg.lambda_semantic = args.lambda_semantic
    if args.warmup_epochs is not None:
        cfg.warmup_epochs = args.warmup_epochs
    if args.codebook_size is not None:
        cfg.codebook_size = args.codebook_size
    if args.epochs is not None:
        cfg.codebook_train_epochs = args.epochs
    if args.sentence_bert:
        cfg.sentence_bert_model = args.sentence_bert

    set_seed(args.seed)
    output_dir = Path(args.output_dir) / args.dataset / args.teacher / f"seed_{args.seed}"
    check_writable(output_dir, overwrite=False)
    setup_logging(__name__, log_file=output_dir / "train_codebook.log")
    if args.console_log:
        logging.getLogger(__name__).setLevel(logging.INFO)

    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device("cpu")

    logger.info("Device: %s", device)

    g, feats, labels, idx_train, idx_val, idx_test, text_dict = load_graph_data(
        args.dataset,
        root=args.data_root,
        node_text_path=args.node_text_csv,
        seed=args.seed,
        labelrate_train=args.labelrate_train,
        labelrate_val=args.labelrate_val,
    )

    logger.info("Extracting Sentence-BERT embeddings...")
    text_emb = extract_sentence_bert_embeddings(
        text_dict,
        model_name=cfg.sentence_bert_model,
        device=device,
    )

    conf = CodebookTrainer.build_conf_from_args(
        args,
        feat_dim=feats.shape[1],
        label_dim=int(labels.max().item()) + 1,
        device=device,
    )
    conf["lamb_node"] = args.lamb_node
    conf["lamb_edge"] = args.lamb_edge
    conf["patience"] = args.patience
    conf["seed"] = args.seed
    conf["dataset"] = args.dataset

    trainer = CodebookTrainer(cfg)
    artifacts = trainer.fit(
        g=g,
        feats=feats,
        labels=labels,
        text_embeddings=text_emb,
        idx_train=idx_train,
        idx_val=idx_val,
        idx_test=idx_test,
        conf=conf,
        output_dir=output_dir,
        logger_inst=logger,
    )

    if args.compute_tfidf:
        from text_tokenizers.text_tokenbook import TextTokenbook

        train_ids = idx_train.cpu().numpy()
        tfidf_path = output_dir / "tfidf_stats.npz"
        tokenbook = TextTokenbook.load(
            Path(args.tokenbook_dir),
            model_name=cfg.sentence_bert_model,
            device=device,
        )
        TFIDFComputer(cfg).run_offline(
            artifacts,
            text_dict,
            train_ids,
            tfidf_path,
            tokenbook=tokenbook,
        )
        logger.info(
            "TF-IDF recomputed with tokenbook vocab (%d tokens). "
            "Re-run this step after switching tokenbook.",
            len(tokenbook),
        )

    logger.info("Training complete. Artifacts: %s", output_dir)


if __name__ == "__main__":
    main()
