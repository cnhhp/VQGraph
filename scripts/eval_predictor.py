"""
评估 token predictor：p_max、L_token、选词变化率。

用法::
  PYTHONPATH=. python scripts/eval_predictor.py \\
    --codebook_dir ./outputs/experiments/e1_tau003/cora/GCN/seed_0 \\
    --tokenbook_dir ./codebook --device 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from config import get_config, reset_config
from graph_utils import extract_sentence_bert_embeddings, load_graph_data
from models.codebook_trainer import CodebookTrainer
from models.token_predictor import (
    FactorizedTokenPredictor,
    build_target_distribution,
    compute_p_code,
)


def _load_predictor_from_state(state: dict, device: torch.device):
    from models.token_predictor import FactorizedTokenPredictor, TokenPredictorHead

    if "encoder.token_predictor.proj.weight" in state:
        w = state["encoder.token_predictor.proj.weight"]
        text_dim = state["encoder.tokenbook_embeddings"].shape[1]
        pred = FactorizedTokenPredictor(w.shape[1], text_dim).to(device)
        pred.load_state_dict(
            {
                k.replace("encoder.token_predictor.", ""): v
                for k, v in state.items()
                if k.startswith("encoder.token_predictor.")
            }
        )
        return pred
    if "encoder.token_predictor.weight" in state:
        w = state["encoder.token_predictor.weight"]
        pred = TokenPredictorHead(w.shape[1], w.shape[0]).to(device)
        pred.weight.data = w
        pred.bias.data = state["encoder.token_predictor.bias"]
        return pred
    return None


def eval_checkpoint(
    codebook_dir: Path,
    tokenbook_dir: Path,
    device: torch.device,
    tau_prime: float | None = None,
) -> dict:
    reset_config()
    cfg = get_config()
    artifacts = CodebookTrainer.load_artifacts(codebook_dir, device)
    state = artifacts.encoder_state_dict

    cfg_path = codebook_dir / "config.json"
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as f:
            snap = json.load(f)
        for k in (
            "token_target_temperature",
            "token_pred_temperature",
            "lambda_token",
            "lambda_pred",
        ):
            if k in snap:
                setattr(cfg, k, snap[k])
    if tau_prime is not None:
        cfg.token_target_temperature = tau_prime

    predictor = _load_predictor_from_state(state, device)
    if predictor is None:
        return {"error": "no token_predictor in checkpoint"}

    book = state.get("encoder.tokenbook_embeddings")
    if book is None:
        from text_tokenizers.text_tokenbook import TextTokenbook

        tb = TextTokenbook.load(
            tokenbook_dir, cfg=cfg, device=device, build_embeddings=True
        )
        book = F.normalize(tb.get_embedding_matrix().float(), dim=-1)
    else:
        book = F.normalize(book.float(), dim=-1)

    cb = artifacts.codebook_embeddings.numpy()
    pmax_list = []
    with torch.no_grad():
        for i in range(min(len(cb), 500)):
            z = torch.tensor(cb[i : i + 1], dtype=torch.float32, device=device)
            p = compute_p_code(
                z,
                predictor,
                tau=cfg.token_pred_temperature,
                tokenbook_emb=book.to(device),
                normalize=getattr(cfg, "p_code_normalize", "none"),
            )
            pmax_list.append(float(p.max().cpu()))

    g, feats, labels, idx_train, idx_val, idx_test, text_dict = load_graph_data(
        "cora", root=Path("./data"), data_source="text"
    )
    text_emb = extract_sentence_bert_embeddings(
        text_dict, model_name=cfg.sentence_bert_model, device=device
    )
    train_idx = idx_train.cpu().numpy()
    sample_te = text_emb[train_idx[:50]]

    with torch.no_grad():
        target = build_target_distribution(
            sample_te, book, tau_prime=cfg.token_target_temperature
        )
        target_ent = -(target * target.clamp(min=1e-12).log()).sum(-1).mean()
        z_sample = torch.tensor(cb[:50], dtype=torch.float32, device=device)
        if isinstance(predictor, FactorizedTokenPredictor):
            logits = predictor(z_sample, book.to(device))
        else:
            logits = predictor(z_sample.float())
        log_pred = F.log_softmax(logits / cfg.token_pred_temperature, dim=-1)
        kl = F.kl_div(log_pred, target[: len(z_sample)], reduction="batchmean")

    V = book.shape[0]
    return {
        "codebook_dir": str(codebook_dir),
        "tau_prime": cfg.token_target_temperature,
        "lambda_token": cfg.lambda_token,
        "p_max_mean": float(np.mean(pmax_list)),
        "p_max_median": float(np.median(pmax_list)),
        "p_max_max": float(np.max(pmax_list)),
        "uniform_p_max": 1.0 / V,
        "target_entropy": float(target_ent.item()),
        "log_vocab": float(np.log(V)),
        "kl_sample": float(kl.item()),
        "num_codes_sampled": len(pmax_list),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="评估 token predictor 指标")
    p.add_argument("--codebook_dir", type=str, required=True)
    p.add_argument("--tokenbook_dir", type=str, default="./codebook")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--tau_prime", type=float, default=None)
    p.add_argument("--json_out", type=str, default=None)
    args = p.parse_args()

    device = torch.device(
        f"cuda:{args.device}" if torch.cuda.is_available() and args.device >= 0 else "cpu"
    )
    metrics = eval_checkpoint(
        Path(args.codebook_dir), Path(args.tokenbook_dir), device, args.tau_prime
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
