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
    p.add_argument(
        "--tfidf_stats_min_epoch",
        type=int,
        default=None,
        help="仅 epoch >= 此值时更新 best checkpoint 并统计 TF-IDF（默认 warmup_epochs+1）",
    )
    p.add_argument("--codebook_size", type=int, default=None)
    p.add_argument(
        "--hierarchical_vq",
        action="store_true",
        help="启用层次粗/细码（双通道 GCN + SemanticVQ 粗码 + 残差细码 + L_H/L_D）",
    )
    p.add_argument(
        "--codebook_size_coarse",
        type=int,
        default=None,
        help="粗码本大小 M_co（默认 16）",
    )
    p.add_argument(
        "--codebook_size_fine",
        type=int,
        default=None,
        help="细码本大小 M_fi（默认 256）",
    )
    p.add_argument("--lambda_H", type=float, default=None, help="异配边重建平衡权重")
    p.add_argument("--lambda_D", type=float, default=None, help="双通道解耦权重")
    p.add_argument(
        "--lambda_L",
        type=float,
        default=None,
        help="弱监督 L_L 权重（仅约束连续 h_L）",
    )
    p.add_argument(
        "--lambda_div",
        type=float,
        default=None,
        help="粗码使用率 KL 权重",
    )
    p.add_argument(
        "--lambda_div_fi",
        type=float,
        default=None,
        help="细码使用率 KL 权重（soft+hard）",
    )
    p.add_argument(
        "--lambda_ico",
        type=float,
        default=None,
        help="同粗码邻居细码拉开 L_intra_co 权重",
    )
    p.add_argument(
        "--select_min_s_L",
        type=float,
        default=None,
        help="层次选模：弱 L_L val 门槛，达标后再比 unique_fi",
    )
    p.add_argument(
        "--fine_noise",
        type=float,
        default=None,
        help="细码量化前训练噪声标准差",
    )
    p.add_argument(
        "--text_fuse",
        type=float,
        default=None,
        help="text 投影注入粗通道强度",
    )
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
        "--tfidf_only",
        action="store_true",
        help="仅重算 TF-IDF（加载已有码本产物，不重新训练）",
    )
    p.add_argument(
        "--tokenbook_dir",
        type=str,
        default="./codebook",
        help="预置文本 tokenbook 目录（训练 token 预测头 / TF-IDF 时使用）",
    )
    p.add_argument("--lambda_token", type=float, default=None, help="Token KL 损失权重 α")
    p.add_argument(
        "--token_pred_temperature",
        type=float,
        default=None,
        help="预测分布温度 τ",
    )
    p.add_argument(
        "--token_target_temperature",
        type=float,
        default=None,
        help="目标分布温度 τ'",
    )
    p.add_argument(
        "--token_kl_top_k",
        type=int,
        default=None,
        help="Top-K KL（0=全词表）",
    )
    p.add_argument(
        "--token_predictor_type",
        type=str,
        default=None,
        choices=["linear", "factorized"],
        help="Token 预测头类型",
    )
    p.add_argument(
        "--p_code_normalize",
        type=str,
        default=None,
        choices=["none", "max", "minmax"],
        help="推理时 P_code 归一化方式",
    )
    p.add_argument(
        "--load_checkpoint",
        type=str,
        default=None,
        help="从已有 model.pth 初始化权重",
    )
    p.add_argument(
        "--init_from_dir",
        type=str,
        default=None,
        help="predictor_only 时复用该目录的码本/node_codes",
    )
    p.add_argument(
        "--predictor_only",
        action="store_true",
        help="冻结 VQ，仅训练 token_predictor（E5 ablation）",
    )
    p.add_argument(
        "--predictor_only_epochs",
        type=int,
        default=None,
        help="predictor_only 训练 epoch 数",
    )
    p.add_argument(
        "--predictor_lr",
        type=float,
        default=1e-3,
        help="predictor_only 学习率",
    )
    p.add_argument(
        "--no_token_predictor",
        action="store_true",
        help="禁用 Token 预测辅助任务",
    )
    p.add_argument("--console_log", action="store_true")
    p.add_argument(
        "--data_source",
        type=str,
        default=None,
        choices=["auto", "text", "cpf"],
        help="数据来源：auto=优先 data/dataset/{name}/ 文本版",
    )
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
    if args.tfidf_stats_min_epoch is not None:
        cfg.tfidf_stats_min_epoch = args.tfidf_stats_min_epoch
    if args.codebook_size is not None:
        cfg.codebook_size = args.codebook_size
    if args.hierarchical_vq:
        cfg.hierarchical_vq = True
    if args.codebook_size_coarse is not None:
        cfg.codebook_size_coarse = args.codebook_size_coarse
    if args.codebook_size_fine is not None:
        cfg.codebook_size_fine = args.codebook_size_fine
    if args.lambda_H is not None:
        cfg.lambda_H = args.lambda_H
    if args.lambda_D is not None:
        cfg.lambda_D = args.lambda_D
    if args.lambda_L is not None:
        cfg.lambda_L = args.lambda_L
    if getattr(args, "lambda_div", None) is not None:
        cfg.lambda_div = args.lambda_div
    if getattr(args, "lambda_div_fi", None) is not None:
        cfg.lambda_div_fi = args.lambda_div_fi
    if getattr(args, "lambda_ico", None) is not None:
        cfg.lambda_ico = args.lambda_ico
    if getattr(args, "select_min_s_L", None) is not None:
        cfg.select_min_s_L = args.select_min_s_L
    if getattr(args, "fine_noise", None) is not None:
        cfg.fine_noise = args.fine_noise
    if getattr(args, "text_fuse", None) is not None:
        cfg.text_fuse = args.text_fuse
    if args.epochs is not None:
        cfg.codebook_train_epochs = args.epochs
    if args.sentence_bert:
        cfg.sentence_bert_model = args.sentence_bert
    if args.data_source is not None:
        cfg.data_source = None if args.data_source == "auto" else args.data_source
    if args.node_text_csv:
        cfg.node_text_csv = Path(args.node_text_csv)
    if args.lambda_token is not None:
        cfg.lambda_token = args.lambda_token
    if args.token_pred_temperature is not None:
        cfg.token_pred_temperature = args.token_pred_temperature
    if args.token_target_temperature is not None:
        cfg.token_target_temperature = args.token_target_temperature
    if args.token_kl_top_k is not None:
        cfg.token_kl_top_k = args.token_kl_top_k
    if args.token_predictor_type is not None:
        cfg.token_predictor_type = args.token_predictor_type
    if args.p_code_normalize is not None:
        cfg.p_code_normalize = args.p_code_normalize
    if args.predictor_only_epochs is not None:
        cfg.predictor_only_epochs = args.predictor_only_epochs
    if args.no_token_predictor:
        cfg.enable_token_predictor = False

    tokenbook_vocab = Path(args.tokenbook_dir) / cfg.tokenbook_vocab_filename
    if cfg.enable_token_predictor and cfg.lambda_token > 0 and not args.tfidf_only:
        if not tokenbook_vocab.exists():
            raise FileNotFoundError(
                f"Token predictor enabled but vocabulary not found: {tokenbook_vocab}"
            )

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
        node_text_path=args.node_text_csv or cfg.node_text_csv,
        seed=args.seed,
        labelrate_train=args.labelrate_train,
        labelrate_val=args.labelrate_val,
        data_source=cfg.data_source,
        text_dataset_subdir=cfg.text_dataset_subdir,
    )
    logger.info(
        "Graph loaded: feat_dim=%d train=%d val=%d test=%d",
        feats.shape[1],
        len(idx_train),
        len(idx_val),
        len(idx_test),
    )
    if feats.shape[1] != 1433 and args.data_source != "cpf":
        logger.warning(
            "Node feature dim is %d (text DGL Cora uses 768). "
            "Retrain codebook if switching from CPF BoW (1433).",
            feats.shape[1],
        )

    if args.tfidf_only:
        if not (output_dir / "model.pth").exists():
            raise FileNotFoundError(
                f"No trained codebook at {output_dir}; run training first or drop --tfidf_only."
            )
        artifacts = CodebookTrainer.load_artifacts(output_dir, device)
        logger.info("Loaded artifacts from %s (tfidf_only mode)", output_dir)
    else:
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
        conf["tokenbook_dir"] = args.tokenbook_dir
        conf["hierarchical_vq"] = bool(args.hierarchical_vq or cfg.hierarchical_vq)
        if conf["hierarchical_vq"]:
            conf["codebook_size_coarse"] = (
                args.codebook_size_coarse
                if args.codebook_size_coarse is not None
                else cfg.codebook_size_coarse
            )
            conf["codebook_size_fine"] = (
                args.codebook_size_fine
                if args.codebook_size_fine is not None
                else cfg.codebook_size_fine
            )
            conf["lambda_H"] = (
                args.lambda_H if args.lambda_H is not None else cfg.lambda_H
            )
            conf["lambda_D"] = (
                args.lambda_D if args.lambda_D is not None else cfg.lambda_D
            )
            conf["lambda_L"] = (
                args.lambda_L if args.lambda_L is not None else cfg.lambda_L
            )
            conf["lambda_div"] = (
                args.lambda_div
                if getattr(args, "lambda_div", None) is not None
                else getattr(cfg, "lambda_div", 0.05)
            )
            conf["lambda_div_fi"] = (
                args.lambda_div_fi
                if getattr(args, "lambda_div_fi", None) is not None
                else getattr(cfg, "lambda_div_fi", 0.5)
            )
            conf["lambda_ico"] = (
                args.lambda_ico
                if getattr(args, "lambda_ico", None) is not None
                else getattr(cfg, "lambda_ico", 0.2)
            )
            conf["select_min_s_L"] = (
                args.select_min_s_L
                if getattr(args, "select_min_s_L", None) is not None
                else getattr(cfg, "select_min_s_L", 0.75)
            )
            conf["fine_noise"] = (
                args.fine_noise
                if getattr(args, "fine_noise", None) is not None
                else getattr(cfg, "fine_noise", 0.0)
            )
            conf["text_fuse"] = (
                args.text_fuse
                if getattr(args, "text_fuse", None) is not None
                else getattr(cfg, "text_fuse", 0.5)
            )
            # v2：层次模式默认略强语义偏置
            if args.lambda_semantic is None and conf.get("lambda_semantic") is None:
                conf["lambda_semantic"] = max(
                    getattr(cfg, "lambda_semantic", 0.1), 0.3
                )
            if args.teacher != "GCN":
                raise ValueError("hierarchical_vq currently requires --teacher GCN")
        if cfg.tfidf_stats_min_epoch is not None:
            conf["tfidf_stats_min_epoch"] = cfg.tfidf_stats_min_epoch
        if args.load_checkpoint:
            conf["load_checkpoint"] = args.load_checkpoint
        if args.init_from_dir:
            conf["init_from_dir"] = args.init_from_dir
        if args.predictor_only:
            conf["predictor_only"] = True
            conf["predictor_only_epochs"] = (
                args.predictor_only_epochs or cfg.predictor_only_epochs
            )
            conf["predictor_lr"] = args.predictor_lr
            if not args.load_checkpoint and not args.init_from_dir:
                raise ValueError(
                    "predictor_only requires --load_checkpoint or --init_from_dir"
                )
            if args.load_checkpoint and not args.init_from_dir:
                conf["init_from_dir"] = str(Path(args.load_checkpoint).parent)

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

    if args.compute_tfidf or args.tfidf_only:
        from text_tokenizers.text_tokenbook import TextTokenbook

        train_ids = idx_train.cpu().numpy()
        tfidf_path = output_dir / "tfidf_stats.npz"
        tokenbook = TextTokenbook.load(
            Path(args.tokenbook_dir),
            model_name=cfg.sentence_bert_model,
            device=device,
            build_embeddings=False,
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

    if args.tfidf_only:
        logger.info("TF-IDF recompute complete. Artifacts: %s", output_dir)
    else:
        logger.info("Training complete. Artifacts: %s", output_dir)


if __name__ == "__main__":
    main()
