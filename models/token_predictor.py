"""
Token 预测辅助头：从 STE 码向量 z_q 预测文本 token 分布（KL 监督）。
"""

from __future__ import annotations

from typing import Optional, Protocol, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


class TokenPredictorHead(nn.Linear):
    """轻量线性头：码向量 d -> tokenbook 词表 V。"""

    def __init__(self, code_dim: int, vocab_size: int) -> None:
        super().__init__(code_dim, vocab_size)
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)


class FactorizedTokenPredictor(nn.Module):
    """因子分解头：logits = (proj(z_q) · tokenbook_emb.T) / τ，与 text_sim 同构。"""

    def __init__(self, code_dim: int, text_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(code_dim, text_dim, bias=False)
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(
        self,
        z_q: torch.Tensor,
        tokenbook_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.proj(z_q.float())
        if tokenbook_emb is None:
            raise ValueError("FactorizedTokenPredictor requires tokenbook_emb")
        book = F.normalize(tokenbook_emb.float(), dim=-1)
        h = F.normalize(h, dim=-1)
        return h @ book.t()


class _PredictorLike(Protocol):
    def forward(self, z_q: torch.Tensor, tokenbook_emb: Optional[torch.Tensor] = ...) -> torch.Tensor: ...


def _predictor_logits(
    predictor: nn.Module,
    z_q: torch.Tensor,
    tokenbook_emb: torch.Tensor,
) -> torch.Tensor:
    if isinstance(predictor, FactorizedTokenPredictor):
        return predictor(z_q, tokenbook_emb)
    return predictor(z_q.float())


def _align_batch_text(
    text_emb: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    """将 text_emb 对齐到 batch_size（与 z_q 节点数一致）。"""
    if text_emb.shape[0] == batch_size:
        return text_emb
    if text_emb.shape[0] == 1:
        return text_emb.expand(batch_size, -1)
    return text_emb[:batch_size]


def build_target_distribution(
    text_emb: torch.Tensor,
    tokenbook_emb: torch.Tensor,
    tau_prime: float = 0.15,
    top_k: int = 0,
) -> torch.Tensor:
    """
    P_target = softmax(cosine_sim(t, tokenbook) / tau')，shape [B, V]。

    top_k > 0 时仅在 cosine Top-K 上归一化，其余为 0。
    """
    t = F.normalize(text_emb.float(), dim=-1)
    book = F.normalize(tokenbook_emb.float(), dim=-1)
    cos_sim = t @ book.t()
    if top_k and top_k < cos_sim.shape[-1]:
        vals, idx = torch.topk(cos_sim, k=top_k, dim=-1)
        target = torch.zeros_like(cos_sim)
        target.scatter_(-1, idx, F.softmax(vals / tau_prime, dim=-1))
        return target
    return F.softmax(cos_sim / tau_prime, dim=-1)


def compute_token_kl_loss(
    z_q: torch.Tensor,
    text_emb: torch.Tensor,
    tokenbook_emb: torch.Tensor,
    predictor: nn.Module,
    tau: float = 1.0,
    tau_prime: float = 0.15,
    top_k: int = 0,
) -> torch.Tensor:
    """
    L_token = KL(P_target || P_pred)。

    top_k > 0 时仅在 Top-K 索引上计算 KL（按行 renormalize）。
    """
    batch_size = z_q.shape[0]
    text_emb = _align_batch_text(text_emb, batch_size)

    logits = _predictor_logits(predictor, z_q, tokenbook_emb)
    target = build_target_distribution(text_emb, tokenbook_emb, tau_prime, top_k=top_k)

    if top_k and top_k < logits.shape[-1]:
        _, idx = torch.topk(
            F.normalize(text_emb, dim=-1) @ F.normalize(tokenbook_emb, dim=-1).t(),
            k=top_k,
            dim=-1,
        )
        pred_sub = logits.gather(-1, idx)
        log_pred = F.log_softmax(pred_sub / tau, dim=-1)
        tgt_sub = target.gather(-1, idx)
        tgt_sub = tgt_sub / tgt_sub.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        return F.kl_div(log_pred, tgt_sub, reduction="batchmean")

    log_pred = F.log_softmax(logits / tau, dim=-1)
    return F.kl_div(log_pred, target, reduction="batchmean")


def compute_p_code(
    code_vectors: torch.Tensor,
    predictor: nn.Module,
    tau: float = 1.0,
    tokenbook_emb: Optional[torch.Tensor] = None,
    normalize: str = "none",
) -> torch.Tensor:
    """
    推理：P_code = softmax(logits / τ)，可选归一化到 [0,1]。

    normalize: none | max | minmax
    """
    single = code_vectors.dim() == 1
    if single:
        code_vectors = code_vectors.unsqueeze(0)
    if isinstance(predictor, FactorizedTokenPredictor):
        if tokenbook_emb is None:
            raise ValueError("FactorizedTokenPredictor needs tokenbook_emb at inference")
        logits = predictor(code_vectors.float(), tokenbook_emb)
    else:
        logits = predictor(code_vectors.float())
    probs = F.softmax(logits / tau, dim=-1)
    if normalize == "max":
        probs = probs / probs.max(dim=-1, keepdim=True).values.clamp(min=1e-12)
    elif normalize == "minmax":
        pmin = probs.min(dim=-1, keepdim=True).values
        pmax = probs.max(dim=-1, keepdim=True).values
        probs = (probs - pmin) / (pmax - pmin + 1e-12)
    return probs.squeeze(0) if single else probs


def attach_token_predictor_to_encoder(
    encoder: nn.Module,
    vocab_size: int,
    tokenbook_embeddings: torch.Tensor,
    cfg: Union[Config, dict],
) -> None:
    """在 GCN/SAGE encoder 上挂载 token_predictor 与固定 tokenbook buffer。"""
    if isinstance(cfg, Config):
        lambda_token = cfg.lambda_token
        tau = cfg.token_pred_temperature
        tau_prime = cfg.token_target_temperature
        top_k = getattr(cfg, "token_kl_top_k", 0)
        pred_type = getattr(cfg, "token_predictor_type", "linear")
    else:
        lambda_token = cfg.get("lambda_token", 0.05)
        tau = cfg.get("token_pred_temperature", 1.0)
        tau_prime = cfg.get("token_target_temperature", 0.15)
        top_k = cfg.get("token_kl_top_k", 0)
        pred_type = cfg.get("token_predictor_type", "linear")

    code_dim = getattr(encoder, "input_dim", None)
    if code_dim is None:
        code_dim = encoder.graph_layer_1._out_feats
    device = next(encoder.parameters()).device
    text_dim = tokenbook_embeddings.shape[1]

    if pred_type == "factorized":
        predictor = FactorizedTokenPredictor(code_dim, text_dim).to(device)
    else:
        predictor = TokenPredictorHead(code_dim, vocab_size).to(device)
    encoder.token_predictor = predictor

    emb = F.normalize(tokenbook_embeddings.float().to(device), dim=-1)
    encoder.register_buffer("tokenbook_embeddings", emb)

    encoder.lambda_token = float(lambda_token)
    encoder.token_pred_tau = float(tau)
    encoder.token_target_tau = float(tau_prime)
    encoder.token_kl_top_k = int(top_k)
    encoder.token_predictor_type = pred_type


def maybe_add_token_loss(
    encoder: nn.Module,
    quantized: torch.Tensor,
    text_emb: Optional[torch.Tensor],
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """若 token_predictor 已挂载，计算 L_token 并返回 (weighted_loss, raw_token_loss)。"""
    predictor = getattr(encoder, "token_predictor", None)
    lambda_token = getattr(encoder, "lambda_token", 0.0)
    tokenbook_emb = getattr(encoder, "tokenbook_embeddings", None)

    if (
        predictor is None
        or lambda_token <= 0
        or text_emb is None
        or tokenbook_emb is None
    ):
        return torch.tensor(0.0, device=quantized.device), None

    token_loss = compute_token_kl_loss(
        quantized,
        text_emb,
        tokenbook_emb,
        predictor,
        tau=getattr(encoder, "token_pred_tau", 1.0),
        tau_prime=getattr(encoder, "token_target_tau", 0.15),
        top_k=getattr(encoder, "token_kl_top_k", 0),
    )
    return lambda_token * token_loss, token_loss
