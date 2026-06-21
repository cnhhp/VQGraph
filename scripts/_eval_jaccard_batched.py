"""Batched Jaccard eval for TokenSelector (avoids full-split forward_hard OOM)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Set

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_config, reset_config
from graph_utils import extract_sentence_bert_embeddings, load_graph_data, mask_tokens_for_selection
from models.codebook_trainer import CodebookTrainer, TFIDFComputer
from models.node_representation import NodeRepresentationTokenizer, select_diverse_tokens
from models.token_selector import build_tfidf_table, load_token_selection_trainer


def _jaccard(a: Set[str], b: Set[str]) -> float:
    u = a | b
    return len(a & b) / len(u) if u else 1.0


def _select_with_scores(tokenizer, scores, k) -> List[str]:
    id_to_token = tokenizer.tokenbook.id_to_token
    cfg = tokenizer.cfg
    selection_scores = scores
    if getattr(cfg, "filter_stopwords_at_selection", True) or getattr(
        cfg, "filter_noise_subwords_at_selection", True
    ):
        masked, _ = mask_tokens_for_selection(
            scores,
            id_to_token,
            filter_stopwords=getattr(cfg, "filter_stopwords_at_selection", True),
            filter_noise_subwords=getattr(cfg, "filter_noise_subwords_at_selection", True),
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
    p = argparse.ArgumentParser()
    p.add_argument("--codebook_dir", required=True)
    p.add_argument("--token_selector_checkpoint", required=True)
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--lambda_pred", type=float, default=0.05)
    p.add_argument("--p_code_normalize", default="max")
    args = p.parse_args()

    reset_config()
    cfg = get_config()
    cfg.lambda_pred = args.lambda_pred
    cfg.p_code_normalize = args.p_code_normalize
    device = torch.device(
        "cpu" if args.device < 0 else (f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    )

    g, feats, labels, idx_train, idx_val, idx_test, text_dict = load_graph_data("cora", root="./data")
    split_map = {"train": idx_train, "val": idx_val, "test": idx_test}
    node_indices = split_map[args.split].tolist()

    artifacts = CodebookTrainer.load_artifacts(args.codebook_dir, device)
    tfidf = TFIDFComputer.load(Path(args.codebook_dir) / "tfidf_stats.npz")
    from text_tokenizers.text_tokenbook import TextTokenbook

    tokenbook = TextTokenbook.load("./codebook", device=device)
    text_emb = extract_sentence_bert_embeddings(text_dict, model_name=cfg.sentence_bert_model, device=device)
    tokenizer = NodeRepresentationTokenizer(artifacts=artifacts, tokenbook=tokenbook, tfidf=tfidf, cfg=cfg)
    encoder_model = tokenizer.load_encoder()
    predictor = getattr(encoder_model.encoder, "token_predictor", None)
    tokenbook_emb_buf = getattr(encoder_model.encoder, "tokenbook_embeddings", None)
    tokenbook_emb = (
        tokenbook_emb_buf.float().to(device)
        if tokenbook_emb_buf is not None
        else tokenbook.get_embedding_matrix().float().to(device)
    )
    tfidf_table = build_tfidf_table(
        tfidf, tokenbook.token_to_id, artifacts.codebook_embeddings.shape[0], len(tokenbook), device
    )
    num_classes = int(labels.max().item()) + 1
    trainer = load_token_selection_trainer(
        args.token_selector_checkpoint,
        num_classes=num_classes,
        d_struct=artifacts.codebook_embeddings.shape[1],
        tokenbook_emb=tokenbook_emb,
        tfidf_table=tfidf_table,
        predictor=predictor,
        device=device,
        cfg=cfg,
        id_to_token=tokenbook.id_to_token,
    )

    dist = tokenizer._infer_distances(g, feats, text_emb)
    struct_codes = dist.argmax(dim=1)
    codebook = artifacts.codebook_embeddings.to(device)
    k = cfg.top_k_text_tokens

    jaccards: List[float] = []
    changed = 0
    correct = 0
    labels_dev = labels.to(device)

    for node_id in node_indices:
        struct_code = int(struct_codes[node_id].item())
        node_emb = text_emb[node_id]
        s0 = tokenizer._compute_initial_scores(node_emb, struct_code)
        tokens_fixed = _select_with_scores(tokenizer, s0, k)

        s0_t = torch.tensor(s0, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            s_adj = trainer.predict_hard_scores(s0_t).squeeze(0).cpu().numpy()
        tokens_learned = _select_with_scores(tokenizer, s_adj, k)

        j = _jaccard(set(tokens_fixed), set(tokens_learned))
        jaccards.append(j)
        if set(tokens_fixed) != set(tokens_learned):
            changed += 1

        z_q = codebook[struct_codes[node_id : node_id + 1]]
        _, logits, _ = trainer.forward_hard(
            node_emb.unsqueeze(0).to(device),
            z_q,
            struct_codes[node_id : node_id + 1],
            labels_dev[node_id : node_id + 1],
        )
        if logits.argmax(dim=-1).item() == labels_dev[node_id].item():
            correct += 1

    n = len(node_indices)
    print(f"Split={args.split} nodes={n}")
    print(f"NodeClassifier hard-topk acc={correct/n:.4f}")
    print(f"Token change rate={changed/n:.2%}")
    print(f"Mean Jaccard (fixed vs learned)={sum(jaccards)/n:.4f}")


if __name__ == "__main__":
    main()
