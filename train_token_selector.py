"""
训练可学习 TokenSelector：Gumbel-Softmax 软选择 + 节点分类监督。

用法::

    python train_token_selector.py \\
      --codebook_dir ./outputs/experiments/e5b_no_ltoken/cora/GCN/seed_1 \\
      --tokenbook_path ./codebook \\
      --output_dir ./outputs/token_selector/e5b_no_ltoken \\
      --lambda_pred 0.05 --p_code_normalize max \\
      --device 0
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim

from config import get_config, reset_config
from graph_utils import (
    build_selection_valid_mask,
    extract_sentence_bert_embeddings,
    load_graph_data,
    set_seed,
    setup_logging,
)
from models.codebook_trainer import CodebookTrainer, TFIDFComputer, assign_node_codes
from models.node_representation import NodeRepresentationTokenizer
from models.token_selector import (
    NodeClassifier,
    TokenSelectionTrainer,
    TokenSelector,
    build_tfidf_table,
    gumbel_tau_for_epoch,
    save_token_selector_checkpoint,
)
from text_tokenizers.text_tokenbook import TextTokenbook
from utils import check_writable

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train learnable TokenSelector for text token ranking")
    p.add_argument("--dataset", type=str, default="cora")
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--codebook_dir", type=str, required=True)
    p.add_argument("--tokenbook_path", type=str, default="./codebook")
    p.add_argument("--tfidf_path", type=str, default=None)
    p.add_argument("--output_dir", type=str, default="./outputs/token_selector")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=int, default=0, help="-1 for CPU")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--predictor_lr", type=float, default=5e-4)
    p.add_argument("--train_predictor", action="store_true")
    p.add_argument("--no_tfidf", action="store_true")
    p.add_argument("--lambda_pred", type=float, default=None)
    p.add_argument("--p_code_normalize", type=str, default=None, choices=["none", "max", "minmax"])
    p.add_argument("--gumbel_tau_init", type=float, default=None)
    p.add_argument("--gumbel_tau_min", type=float, default=None)
    p.add_argument("--gumbel_tau_anneal_epochs", type=int, default=None)
    p.add_argument("--top_k_hard", type=int, default=None, help="验证时硬 Top-k")
    p.add_argument("--candidate_pool", type=int, default=None, help="s0 Top-K 候选池大小")
    p.add_argument("--kl_weight", type=float, default=None, help="KL(w||softmax(s0)) 权重")
    p.add_argument("--entropy_weight", type=float, default=None, help="选择熵正则权重")
    p.add_argument("--vtext_dropout", type=float, default=None, help="训练时 v_text dropout")
    p.add_argument(
        "--data_source",
        type=str,
        default=None,
        choices=["auto", "text", "cpf"],
    )
    p.add_argument("--console_log", action="store_true")
    return p.parse_args()


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return float((pred == labels).float().mean().item())


def _run_epoch_soft(
    trainer: TokenSelectionTrainer,
    node_indices: torch.Tensor,
    text_emb: torch.Tensor,
    z_q_all: torch.Tensor,
    struct_codes: torch.Tensor,
    labels: torch.Tensor,
    optimizer: optim.Optimizer,
    batch_size: int,
    tau: float,
) -> Tuple[float, float, Dict[str, float]]:
    trainer.train()
    perm = node_indices[torch.randperm(node_indices.shape[0])]
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0
    aux_sums: Dict[str, float] = {}

    for start in range(0, perm.shape[0], batch_size):
        batch_idx = perm[start : start + batch_size]
        batch_text = text_emb[batch_idx]
        batch_z = z_q_all[batch_idx]
        batch_codes = struct_codes[batch_idx]
        batch_labels = labels[batch_idx]

        optimizer.zero_grad()
        loss, logits, aux = trainer.forward_soft(
            batch_text, batch_z, batch_codes, batch_labels, tau=tau
        )
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        total_acc += _accuracy(logits.detach(), batch_labels)
        for key, val in aux.items():
            aux_sums[key] = aux_sums.get(key, 0.0) + val
        n_batches += 1

    if n_batches == 0:
        return 0.0, 0.0, {}
    aux_avg = {k: v / n_batches for k, v in aux_sums.items()}
    return total_loss / n_batches, total_acc / n_batches, aux_avg


@torch.no_grad()
def _evaluate_hard(
    trainer: TokenSelectionTrainer,
    node_indices: torch.Tensor,
    text_emb: torch.Tensor,
    z_q_all: torch.Tensor,
    struct_codes: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
) -> Tuple[float, float]:
    trainer.eval()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for start in range(0, node_indices.shape[0], batch_size):
        batch_idx = node_indices[start : start + batch_size]
        batch_text = text_emb[batch_idx]
        batch_z = z_q_all[batch_idx]
        batch_codes = struct_codes[batch_idx]
        batch_labels = labels[batch_idx]

        loss, logits, _ = trainer.forward_hard(
            batch_text, batch_z, batch_codes, batch_labels
        )
        total_loss += float(loss.item())
        total_acc += _accuracy(logits, batch_labels)
        n_batches += 1

    if n_batches == 0:
        return 0.0, 0.0
    return total_loss / n_batches, total_acc / n_batches


def main() -> None:
    args = parse_args()
    reset_config()
    cfg = get_config()
    cfg.dataset_name = args.dataset
    cfg.data_root = Path(args.data_root)

    if args.lambda_pred is not None:
        cfg.lambda_pred = args.lambda_pred
    if args.p_code_normalize is not None:
        cfg.p_code_normalize = args.p_code_normalize

    epochs = args.epochs or cfg.token_selector_epochs
    batch_size = args.batch_size or cfg.token_selector_batch_size
    lr = args.lr or cfg.token_selector_lr
    tau_init = args.gumbel_tau_init if args.gumbel_tau_init is not None else cfg.gumbel_tau_init
    tau_min = args.gumbel_tau_min if args.gumbel_tau_min is not None else cfg.gumbel_tau_min
    anneal_epochs = (
        args.gumbel_tau_anneal_epochs
        if args.gumbel_tau_anneal_epochs is not None
        else cfg.gumbel_tau_anneal_epochs
    )
    top_k_hard = args.top_k_hard or cfg.top_k_text_tokens
    candidate_pool = args.candidate_pool or cfg.token_selector_candidate_pool
    kl_weight = args.kl_weight if args.kl_weight is not None else cfg.token_selector_kl_weight
    entropy_weight = (
        args.entropy_weight
        if args.entropy_weight is not None
        else cfg.token_selector_entropy_weight
    )
    vtext_dropout = (
        args.vtext_dropout if args.vtext_dropout is not None else cfg.token_selector_vtext_dropout
    )

    set_seed(args.seed)
    device = torch.device(
        "cpu" if args.device < 0 else (f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    )

    codebook_dir = Path(args.codebook_dir)
    output_dir = Path(args.output_dir) / args.dataset / f"seed_{args.seed}"
    check_writable(output_dir, overwrite=True)
    setup_logging(__name__, log_file=output_dir / "train_token_selector.log")
    if args.console_log:
        logging.getLogger(__name__).setLevel(logging.INFO)

    data_source = None if (args.data_source is None or args.data_source == "auto") else args.data_source
    g, feats, labels, idx_train, idx_val, idx_test, text_dict = load_graph_data(
        args.dataset,
        root=args.data_root,
        seed=args.seed,
        data_source=data_source,
    )
    num_classes = int(labels.max().item()) + 1

    artifacts = CodebookTrainer.load_artifacts(codebook_dir, device)
    tfidf = None
    if not args.no_tfidf:
        tfidf_path = Path(args.tfidf_path) if args.tfidf_path else codebook_dir / "tfidf_stats.npz"
        if tfidf_path.exists():
            tfidf = TFIDFComputer.load(tfidf_path)

    tokenbook = TextTokenbook.load(args.tokenbook_path, device=device)
    text_emb = extract_sentence_bert_embeddings(
        text_dict,
        model_name=cfg.sentence_bert_model,
        device=device,
    ).to(device)

    tokenizer_helper = NodeRepresentationTokenizer(
        artifacts=artifacts,
        tokenbook=tokenbook,
        tfidf=tfidf,
        cfg=cfg,
    )
    encoder_model = tokenizer_helper.load_encoder()
    predictor = getattr(encoder_model.encoder, "token_predictor", None)
    tokenbook_emb_buf = getattr(encoder_model.encoder, "tokenbook_embeddings", None)
    if tokenbook_emb_buf is not None:
        tokenbook_emb = tokenbook_emb_buf.float().to(device)
    else:
        tokenbook_emb = tokenbook.get_embedding_matrix().float().to(device)

    if artifacts.node_code_assignments is not None:
        struct_codes_np = artifacts.node_code_assignments.astype(np.int64)
        logger.info("Using precomputed node_codes from artifacts")
    else:
        logger.info("node_codes.npz missing; running full-graph inference")
        struct_codes_np = assign_node_codes(
            encoder_model, g, feats, text_emb, device
        )
    struct_codes = torch.tensor(struct_codes_np, dtype=torch.long, device=device)

    codebook = artifacts.codebook_embeddings.to(device)
    z_q_all = codebook[struct_codes]

    num_codes = codebook.shape[0]
    vocab_size = len(tokenbook)
    d_text = tokenbook_emb.shape[1]
    d_struct = codebook.shape[1]

    tfidf_table = build_tfidf_table(
        tfidf,
        tokenbook.token_to_id,
        num_codes,
        vocab_size,
        device,
    )

    if predictor is not None:
        if args.train_predictor:
            predictor.train()
            for p in predictor.parameters():
                p.requires_grad = True
        else:
            predictor.eval()
            for p in predictor.parameters():
                p.requires_grad = False

    token_selector = TokenSelector(d_text, hidden_dim=cfg.token_selector_hidden_dim).to(device)
    node_classifier = NodeClassifier(
        d_text,
        d_struct,
        num_classes,
        vtext_dropout=vtext_dropout,
    ).to(device)

    selection_valid_mask = None
    if cfg.filter_noise_subwords_at_selection:
        valid_np = build_selection_valid_mask(
            tokenbook.id_to_token,
            vocab_size,
            filter_stopwords=False,
            filter_noise_subwords=True,
        )
        selection_valid_mask = torch.tensor(valid_np, dtype=torch.bool, device=device)
        logger.info(
            "Noise filter: %d / %d tokens blocked for candidate pool",
            int((~valid_np).sum()),
            vocab_size,
        )

    trainer = TokenSelectionTrainer(
        token_selector=token_selector,
        node_classifier=node_classifier,
        tokenbook_emb=tokenbook_emb,
        tfidf_table=tfidf_table,
        predictor=predictor,
        lambda_tfidf=cfg.lambda_tfidf,
        lambda_pred=cfg.lambda_pred,
        token_pred_tau=cfg.token_pred_temperature,
        p_code_normalize=cfg.p_code_normalize,
        train_predictor=args.train_predictor,
        top_k_hard=top_k_hard,
        candidate_pool=candidate_pool,
        kl_weight=kl_weight,
        entropy_weight=entropy_weight,
        vtext_dropout=vtext_dropout,
        selection_valid_mask=selection_valid_mask,
    ).to(device)

    param_groups: List[Dict[str, Any]] = [
        {"params": list(token_selector.parameters()) + list(node_classifier.parameters()), "lr": lr},
    ]
    if args.train_predictor and predictor is not None:
        param_groups.append({"params": list(predictor.parameters()), "lr": args.predictor_lr})
    optimizer = optim.Adam(param_groups)

    labels_dev = labels.to(device)
    idx_train_dev = idx_train.to(device)
    idx_val_dev = idx_val.to(device)
    idx_test_dev = idx_test.to(device)

    best_val_acc = -1.0
    best_state = copy.deepcopy(token_selector.state_dict())
    best_metrics: Dict[str, Any] = {}
    history: List[List[float]] = []

    logger.info(
        "Train TokenSelector: N=%d V=%d train=%d val=%d device=%s "
        "pool=%d kl=%.3f ent=%.3f vtext_drop=%.2f",
        g.num_nodes(),
        vocab_size,
        len(idx_train),
        len(idx_val),
        device,
        candidate_pool,
        kl_weight,
        entropy_weight,
        vtext_dropout,
    )

    for epoch in range(1, epochs + 1):
        tau = gumbel_tau_for_epoch(epoch, tau_init, tau_min, anneal_epochs)
        train_loss, train_acc, train_aux = _run_epoch_soft(
            trainer,
            idx_train_dev,
            text_emb,
            z_q_all,
            struct_codes,
            labels_dev,
            optimizer,
            batch_size,
            tau,
        )
        val_loss, val_acc = _evaluate_hard(
            trainer,
            idx_val_dev,
            text_emb,
            z_q_all,
            struct_codes,
            labels_dev,
            batch_size,
        )
        history.append([epoch, train_loss, train_acc, val_loss, val_acc, tau])
        kl_log = train_aux.get("kl_loss", 0.0)
        ent_log = train_aux.get("entropy", 0.0)
        logger.info(
            "Ep %3d | tau=%.3f | train loss %.4f acc %.4f kl %.4f ent %.3f | "
            "val loss %.4f acc %.4f",
            epoch,
            tau,
            train_loss,
            train_acc,
            kl_log,
            ent_log,
            val_loss,
            val_acc,
        )
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(token_selector.state_dict())
            best_metrics = {
                "best_epoch": epoch,
                "train_loss": train_loss,
                "train_acc_soft": train_acc,
                "val_loss_hard": val_loss,
                "val_acc_hard": val_acc,
            }

    token_selector.load_state_dict(best_state)
    test_loss, test_acc = _evaluate_hard(
        trainer,
        idx_test_dev,
        text_emb,
        z_q_all,
        struct_codes,
        labels_dev,
        batch_size,
    )
    best_metrics["test_loss_hard"] = test_loss
    best_metrics["test_acc_hard"] = test_acc

    ckpt_config = {
        "d_text": d_text,
        "d_struct": d_struct,
        "hidden_dim": cfg.token_selector_hidden_dim,
        "lambda_tfidf": cfg.lambda_tfidf,
        "lambda_pred": cfg.lambda_pred,
        "token_pred_tau": cfg.token_pred_temperature,
        "p_code_normalize": cfg.p_code_normalize,
        "top_k_hard": top_k_hard,
        "candidate_pool": candidate_pool,
        "kl_weight": kl_weight,
        "entropy_weight": entropy_weight,
        "vtext_dropout": vtext_dropout,
        "filter_noise_subwords_at_selection": cfg.filter_noise_subwords_at_selection,
        "codebook_dir": str(codebook_dir),
        "tokenbook_path": str(args.tokenbook_path),
        "train_predictor": args.train_predictor,
        "seed": args.seed,
    }
    metrics_out = {
        "dataset": args.dataset,
        "split": {
            "train": int(len(idx_train)),
            "val": int(len(idx_val)),
            "test": int(len(idx_test)),
        },
        "accuracy": best_metrics,
        "hyperparameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "predictor_lr": args.predictor_lr,
            "gumbel_tau_init": tau_init,
            "gumbel_tau_min": tau_min,
            "gumbel_tau_anneal_epochs": anneal_epochs,
            "candidate_pool": candidate_pool,
            "kl_weight": kl_weight,
            "entropy_weight": entropy_weight,
            "vtext_dropout": vtext_dropout,
            "train_predictor": args.train_predictor,
        },
    }

    save_token_selector_checkpoint(
        output_dir / "best.pth",
        token_selector,
        node_classifier,
        ckpt_config,
        metrics_out,
    )
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2, ensure_ascii=False)
    if history:
        np.savez(output_dir / "history.npz", history=np.array(history))

    logger.info(
        "Done | best val=%.4f test=%.4f | saved %s",
        best_val_acc,
        test_acc,
        output_dir / "best.pth",
    )
    print(
        f"TokenSelector | val={best_val_acc:.4f} test={test_acc:.4f} | {output_dir / 'best.pth'}"
    )


if __name__ == "__main__":
    main()
