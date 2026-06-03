"""
评估 TokenSelector：对比固定 s0+MMR vs TokenSelector+MMR 的选词差异与分类准确率。

用法::

    PYTHONPATH=. python scripts/eval_token_selector.py \\
      --codebook_dir ./outputs/experiments/e5b_no_ltoken/cora/GCN/seed_1 \\
      --token_selector_checkpoint ./outputs/token_selector/e5b_no_ltoken/cora/seed_42/best.pth \\
      --split val --device 0
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Set, Tuple

import torch

from config import get_config, reset_config
from graph_utils import extract_sentence_bert_embeddings, load_graph_data, mask_tokens_for_selection, setup_logging
from models.codebook_trainer import CodebookTrainer, TFIDFComputer
from models.node_representation import NodeRepresentationTokenizer, select_diverse_tokens
from models.token_selector import build_tfidf_table, load_token_selection_trainer
from text_tokenizers.text_tokenbook import TextTokenbook

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate learnable TokenSelector vs fixed scoring")
    p.add_argument("--dataset", type=str, default="cora")
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--codebook_dir", type=str, required=True)
    p.add_argument("--tokenbook_path", type=str, default="./codebook")
    p.add_argument("--token_selector_checkpoint", type=str, required=True)
    p.add_argument("--tfidf_path", type=str, default=None)
    p.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    p.add_argument("--max_nodes", type=int, default=None)
    p.add_argument("--device", type=int, default=0)
    p.add_argument(
        "--data_source",
        type=str,
        default=None,
        choices=["auto", "text", "cpf"],
    )
    p.add_argument("--lambda_pred", type=float, default=None)
    p.add_argument("--p_code_normalize", type=str, default=None, choices=["none", "max", "minmax"])
    return p.parse_args()


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def _select_with_scores(
    tokenizer: NodeRepresentationTokenizer,
    node_text_emb: torch.Tensor,
    struct_code: int,
    scores,
) -> List[str]:
    cfg = tokenizer.cfg
    k = cfg.top_k_text_tokens
    id_to_token = tokenizer.tokenbook.id_to_token
    selection_scores = scores
    filter_stop = getattr(cfg, "filter_stopwords_at_selection", True)
    filter_noise = getattr(cfg, "filter_noise_subwords_at_selection", True)
    if filter_stop or filter_noise:
        masked, n_valid = mask_tokens_for_selection(
            scores,
            id_to_token,
            filter_stopwords=filter_stop,
            filter_noise_subwords=filter_noise,
        )
        if int((masked > -1e11).sum()) >= k:
            selection_scores = masked

    book_emb = tokenizer.tokenbook.get_embedding_matrix()
    top_ids = select_diverse_tokens(
        selection_scores,
        book_emb,
        k=k,
        mmr_lambda=cfg.mmr_lambda,
        candidate_pool=cfg.mmr_candidate_pool,
    )
    return [id_to_token[int(i)] for i in top_ids if int(i) in id_to_token]


def main() -> None:
    args = parse_args()
    reset_config()
    cfg = get_config()
    if args.lambda_pred is not None:
        cfg.lambda_pred = args.lambda_pred
    if args.p_code_normalize is not None:
        cfg.p_code_normalize = args.p_code_normalize

    device = torch.device(
        "cpu" if args.device < 0 else (f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    )
    setup_logging(__name__)

    data_source = None if (args.data_source is None or args.data_source == "auto") else args.data_source
    g, feats, labels, idx_train, idx_val, idx_test, text_dict = load_graph_data(
        args.dataset,
        root=args.data_root,
        data_source=data_source,
    )
    split_map = {"train": idx_train, "val": idx_val, "test": idx_test}
    node_indices = split_map[args.split].tolist()
    if args.max_nodes:
        node_indices = node_indices[: args.max_nodes]

    codebook_dir = Path(args.codebook_dir)
    artifacts = CodebookTrainer.load_artifacts(codebook_dir, device)
    tfidf = None
    tfidf_path = Path(args.tfidf_path) if args.tfidf_path else codebook_dir / "tfidf_stats.npz"
    if tfidf_path.exists():
        tfidf = TFIDFComputer.load(tfidf_path)

    tokenbook = TextTokenbook.load(args.tokenbook_path, device=device)
    text_emb = extract_sentence_bert_embeddings(
        text_dict,
        model_name=cfg.sentence_bert_model,
        device=device,
    )

    tokenizer = NodeRepresentationTokenizer(
        artifacts=artifacts,
        tokenbook=tokenbook,
        tfidf=tfidf,
        cfg=cfg,
    )
    encoder_model = tokenizer.load_encoder()
    predictor = getattr(encoder_model.encoder, "token_predictor", None)
    tokenbook_emb_buf = getattr(encoder_model.encoder, "tokenbook_embeddings", None)
    if tokenbook_emb_buf is not None:
        tokenbook_emb = tokenbook_emb_buf.float().to(device)
    else:
        tokenbook_emb = tokenbook.get_embedding_matrix().float().to(device)

    num_codes = artifacts.codebook_embeddings.shape[0]
    tfidf_table = build_tfidf_table(
        tfidf,
        tokenbook.token_to_id,
        num_codes,
        len(tokenbook),
        device,
    )

    num_classes = int(labels.max().item()) + 1
    d_struct = artifacts.codebook_embeddings.shape[1]
    trainer = load_token_selection_trainer(
        args.token_selector_checkpoint,
        num_classes=num_classes,
        d_struct=d_struct,
        tokenbook_emb=tokenbook_emb,
        tfidf_table=tfidf_table,
        predictor=predictor,
        device=device,
        cfg=cfg,
        id_to_token=tokenbook.id_to_token,
    )

    dist = tokenizer._infer_distances(g, feats, text_emb)
    struct_codes = dist.argmax(dim=1)

    jaccards: List[float] = []
    changed = 0
    idx_tensor = torch.tensor(node_indices, dtype=torch.long, device=device)
    labels_dev = labels.to(device)
    codebook = artifacts.codebook_embeddings.to(device)
    z_q = codebook[struct_codes[idx_tensor]]
    batch_text = text_emb[idx_tensor].to(device)
    batch_codes = struct_codes[idx_tensor]

    _, logits, _ = trainer.forward_hard(
        batch_text,
        z_q,
        batch_codes,
        labels_dev[idx_tensor],
    )
    cls_acc = float((logits.argmax(dim=-1) == labels_dev[idx_tensor]).float().mean().item())

    for node_id in node_indices:
        struct_code = int(struct_codes[node_id].item())
        node_emb = text_emb[node_id]
        s0 = tokenizer._compute_initial_scores(node_emb, struct_code)
        tokens_fixed = _select_with_scores(tokenizer, node_emb, struct_code, s0)

        s0_t = torch.tensor(s0, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            s_adj = trainer.predict_hard_scores(s0_t).squeeze(0).cpu().numpy()
        tokens_learned = _select_with_scores(tokenizer, node_emb, struct_code, s_adj)

        j = _jaccard(set(tokens_fixed), set(tokens_learned))
        jaccards.append(j)
        if set(tokens_fixed) != set(tokens_learned):
            changed += 1

    mean_jaccard = sum(jaccards) / max(len(jaccards), 1)
    print(f"Split={args.split} nodes={len(node_indices)}")
    print(f"NodeClassifier hard-topk acc={cls_acc:.4f}")
    print(f"Token change rate={changed / max(len(node_indices), 1):.2%}")
    print(f"Mean Jaccard (fixed vs learned)={mean_jaccard:.4f}")


if __name__ == "__main__":
    main()
