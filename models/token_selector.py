"""
可学习文本 Token 选择：TokenSelector + NodeClassifier + Gumbel-Softmax 软选择训练。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config, get_config
from models.structural_codebook import TFIDFStatistics
from models.token_predictor import compute_p_code

logger = logging.getLogger(__name__)


class TokenSelector(nn.Module):
    """共享 MLP：对每个 token 将初始得分与嵌入拼接后输出调整得分。"""

    def __init__(self, d_text: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.d_text = d_text
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(d_text + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, s0: torch.Tensor, token_embs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        s0 : (B, V) 初始得分
        token_embs : (V, d_text) 冻结 tokenbook 嵌入

        Returns
        -------
        adjusted_s : (B, V)
        """
        b, v = s0.shape
        if token_embs.shape[0] != v:
            raise ValueError(
                f"s0 vocab {v} != token_embs rows {token_embs.shape[0]}"
            )
        token_embs_expanded = token_embs.unsqueeze(0).expand(b, -1, -1)
        s0_expanded = s0.unsqueeze(-1)
        x = torch.cat([s0_expanded, token_embs_expanded], dim=-1)
        return self.net(x).squeeze(-1)


class NodeClassifier(nn.Module):
    """节点分类头：拼接连续文本向量与结构码向量。"""

    def __init__(
        self,
        d_text: int,
        d_struct: int,
        num_classes: int,
        vtext_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.vtext_dropout = float(vtext_dropout)
        self.fc = nn.Linear(d_text + d_struct, num_classes)

    def forward(
        self,
        v_text: torch.Tensor,
        z_q: torch.Tensor,
        *,
        apply_vtext_dropout: bool = True,
    ) -> torch.Tensor:
        if apply_vtext_dropout and self.training and self.vtext_dropout > 0:
            v_text = F.dropout(v_text, p=self.vtext_dropout, training=True)
        feat = torch.cat([v_text, z_q], dim=-1)
        return self.fc(feat)


def build_tfidf_table(
    tfidf: Optional[TFIDFStatistics],
    tokenbook_vocab: Dict[str, int],
    num_codes: int,
    vocab_size: int,
    device: torch.device,
) -> torch.Tensor:
    """
    预构建 TF-IDF 先验表 [M, V]，行对齐结构码、列对齐 tokenbook。
    """
    table = torch.zeros(num_codes, vocab_size, dtype=torch.float32, device=device)
    if tfidf is None:
        return table

    tfidf_vocab = tfidf.token_to_id
    for code_idx in range(min(num_codes, tfidf.num_codes)):
        row = tfidf.get_prior_weights(code_idx)
        if row.max() < 1e-8:
            continue
        if tfidf_vocab is None:
            n = min(len(row), vocab_size)
            table[code_idx, :n] = torch.tensor(row[:n], dtype=torch.float32, device=device)
            continue
        if tfidf_vocab == tokenbook_vocab and len(row) == vocab_size:
            table[code_idx] = torch.tensor(row, dtype=torch.float32, device=device)
            continue
        tfidf_id_to_token = {i: t for t, i in tfidf_vocab.items()}
        for tfidf_id, weight in enumerate(row):
            if weight <= 0:
                continue
            token = tfidf_id_to_token.get(tfidf_id)
            if token is None:
                continue
            book_id = tokenbook_vocab.get(token)
            if book_id is not None:
                table[code_idx, book_id] = float(weight)
    return table


def compute_text_similarity_batch(
    node_text_emb: torch.Tensor,
    tokenbook_emb: torch.Tensor,
) -> torch.Tensor:
    """节点嵌入与 tokenbook 余弦相似度，归一化到 [0,1]，shape [B, V]。"""
    book_norm = F.normalize(tokenbook_emb.float(), dim=1)
    node_norm = F.normalize(node_text_emb.float(), dim=1)
    cos = node_norm @ book_norm.t()
    return (cos + 1.0) / 2.0


def compute_p_code_batch(
    code_vectors: torch.Tensor,
    predictor: Optional[nn.Module],
    tokenbook_emb: torch.Tensor,
    tau: float,
    normalize: str,
    lambda_pred: float,
    detach: bool,
) -> torch.Tensor:
    """批量 P_code，shape [B, V]；无 predictor 或 lambda_pred=0 时返回零。"""
    batch_size, vocab_size = code_vectors.shape[0], tokenbook_emb.shape[0]
    zeros = torch.zeros(batch_size, vocab_size, device=code_vectors.device)
    if predictor is None or lambda_pred <= 0:
        return zeros

    probs = compute_p_code(
        code_vectors,
        predictor,
        tau=tau,
        tokenbook_emb=tokenbook_emb,
        normalize=normalize,
    )
    if detach:
        probs = probs.detach()
    if probs.shape[-1] != vocab_size:
        aligned = torch.zeros(batch_size, vocab_size, device=probs.device)
        n = min(probs.shape[-1], vocab_size)
        aligned[:, :n] = probs[:, :n]
        return aligned
    return probs


def compute_initial_scores(
    node_text_emb: torch.Tensor,
    struct_code_idx: torch.Tensor,
    tokenbook_emb: torch.Tensor,
    tfidf_table: torch.Tensor,
    code_vectors: torch.Tensor,
    predictor: Optional[nn.Module],
    lambda_tfidf: float,
    lambda_pred: float,
    token_pred_tau: float = 1.0,
    p_code_normalize: str = "max",
    detach_p_code: bool = True,
) -> torch.Tensor:
    """
    score[t] = text_sim[t] * (1 + λ_tfidf * TF-IDF[c][t] + λ_pred * P_code[t])
    """
    text_sim = compute_text_similarity_batch(node_text_emb, tokenbook_emb)
    prior = tfidf_table[struct_code_idx]
    p_code = compute_p_code_batch(
        code_vectors,
        predictor,
        tokenbook_emb,
        tau=token_pred_tau,
        normalize=p_code_normalize,
        lambda_pred=lambda_pred,
        detach=detach_p_code,
    )

    multiplier = torch.ones_like(text_sim)
    if lambda_tfidf > 0 and prior.max() >= 1e-8:
        multiplier = multiplier + lambda_tfidf * prior
    if lambda_pred > 0 and p_code.max() >= 1e-8:
        multiplier = multiplier + lambda_pred * p_code
    return text_sim * multiplier


def compute_initial_scores_numpy(
    node_text_emb: torch.Tensor,
    struct_code_idx: int,
    tokenbook_emb: torch.Tensor,
    tfidf_table: torch.Tensor,
    code_vectors: torch.Tensor,
    predictor: Optional[nn.Module],
    lambda_tfidf: float,
    lambda_pred: float,
    token_pred_tau: float = 1.0,
    p_code_normalize: str = "max",
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    """单节点 numpy 包装，供 NodeRepresentationTokenizer 推理使用。"""
    if node_text_emb.dim() == 1:
        node_text_emb = node_text_emb.unsqueeze(0)
    idx = torch.tensor([struct_code_idx], dtype=torch.long, device=device)
    if code_vectors.dim() == 1:
        code_vectors = code_vectors.unsqueeze(0)

    with torch.no_grad():
        s0 = compute_initial_scores(
            node_text_emb.to(device),
            idx,
            tokenbook_emb.to(device),
            tfidf_table.to(device),
            code_vectors.to(device),
            predictor,
            lambda_tfidf=lambda_tfidf,
            lambda_pred=lambda_pred,
            token_pred_tau=token_pred_tau,
            p_code_normalize=p_code_normalize,
            detach_p_code=True,
        )
    return s0.squeeze(0).cpu().numpy()


def gumbel_tau_for_epoch(
    epoch: int,
    tau_init: float,
    tau_min: float,
    anneal_epochs: int,
) -> float:
    """epoch 为 1-indexed；第 1 epoch 使用 tau_init，之后线性退火至 tau_min。"""
    if anneal_epochs <= 0:
        return tau_min
    progress = min(1.0, max(0, epoch - 1) / anneal_epochs)
    return tau_init + (tau_min - tau_init) * progress


def build_s0_candidate_mask(
    s0: torch.Tensor,
    candidate_pool: int,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """s0 Top-K 候选池布尔掩码（可选排除噪声/停用词），shape [B, V]。"""
    ranked = s0
    if valid_mask is not None:
        if valid_mask.dim() == 1:
            valid_mask = valid_mask.unsqueeze(0)
        ranked = s0.masked_fill(~valid_mask, float("-inf"))
    pool_size = min(int(candidate_pool), s0.shape[-1])
    topk_idx = ranked.topk(pool_size, dim=-1).indices
    mask = torch.zeros_like(s0, dtype=torch.bool)
    mask.scatter_(1, topk_idx, True)
    return mask


def mask_scores_to_candidate_pool(
    scores: torch.Tensor,
    s0: torch.Tensor,
    candidate_pool: int,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """将非 s0 Top-K 候选位置置为 -inf。"""
    mask = build_s0_candidate_mask(s0, candidate_pool, valid_mask=valid_mask)
    return scores.masked_fill(~mask, float("-inf"))


def s0_target_distribution(s0: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """候选池内 softmax(s0)，作为 KL 正则目标分布。"""
    s0_pool = s0.masked_fill(~mask, float("-inf"))
    return F.softmax(s0_pool, dim=-1)


def selection_kl_loss(w: torch.Tensor, p_s0: torch.Tensor) -> torch.Tensor:
    """KL(w || p_s0)，batch mean。"""
    w_safe = w.clamp(min=1e-8)
    p_safe = p_s0.clamp(min=1e-8)
    return (w_safe * (w_safe.log() - p_safe.log())).sum(dim=-1).mean()


def selection_entropy(w: torch.Tensor) -> torch.Tensor:
    """选择分布 Shannon 熵，batch mean。"""
    w_safe = w.clamp(min=1e-8)
    return -(w_safe * w_safe.log()).sum(dim=-1).mean()


def compute_mmr_target_topk_indices(
    scores_np: np.ndarray,
    tokenbook_emb: torch.Tensor,
    id_to_token: Dict[int, str],
    top_k: int,
    mmr_lambda: float,
    mmr_candidate_pool: int,
    filter_stopwords: bool = True,
    filter_noise_subwords: bool = True,
) -> List[int]:
    """baseline s0 + 过滤 + MMR 硬 Top-k，与 node_representation 选词一致。"""
    from graph_utils import mask_tokens_for_selection
    from models.node_representation import select_diverse_tokens

    selection_scores = np.asarray(scores_np, dtype=np.float64)
    k = min(int(top_k), selection_scores.shape[0])
    if k <= 0:
        return []

    if filter_stopwords or filter_noise_subwords:
        masked, _ = mask_tokens_for_selection(
            selection_scores,
            id_to_token,
            filter_stopwords=filter_stopwords,
            filter_noise_subwords=filter_noise_subwords,
        )
        n_valid = int((masked > -1e11).sum())
        if n_valid >= k:
            selection_scores = masked

    return select_diverse_tokens(
        selection_scores,
        tokenbook_emb,
        k=k,
        mmr_lambda=mmr_lambda,
        candidate_pool=mmr_candidate_pool,
    )


def build_mmr_target_distribution(
    topk_idx: torch.Tensor,
    vocab_size: int,
) -> torch.Tensor:
    """MMR 硬 Top-k 均匀目标分布，shape [B, V]。"""
    batch_size, k = topk_idx.shape
    p_target = torch.zeros(
        batch_size,
        vocab_size,
        device=topk_idx.device,
        dtype=torch.float32,
    )
    if k <= 0:
        return p_target
    weight = 1.0 / float(k)
    p_target.scatter_(1, topk_idx, weight)
    return p_target


@torch.no_grad()
def precompute_mmr_target_topk_indices(
    trainer: "TokenSelectionTrainer",
    text_emb: torch.Tensor,
    z_q_all: torch.Tensor,
    struct_codes: torch.Tensor,
    id_to_token: Dict[int, str],
    num_nodes: int,
    top_k: int,
    mmr_lambda: float,
    mmr_candidate_pool: int,
    filter_stopwords: bool,
    filter_noise_subwords: bool,
) -> torch.Tensor:
    """全图预计算 MMR teacher indices，shape [N, k]。"""
    k = int(top_k)
    indices = torch.zeros(num_nodes, k, dtype=torch.long, device=text_emb.device)
    was_training = trainer.training
    trainer.eval()
    try:
        for node_id in range(num_nodes):
            if node_id % 500 == 0:
                logger.info("Precomputing MMR targets: %d / %d", node_id, num_nodes)
            s0 = trainer.compute_s0(
                text_emb[node_id : node_id + 1],
                struct_codes[node_id : node_id + 1],
                z_q_all[node_id : node_id + 1],
            )
            ids = compute_mmr_target_topk_indices(
                s0.squeeze(0).cpu().numpy(),
                trainer.tokenbook_emb,
                id_to_token,
                top_k=k,
                mmr_lambda=mmr_lambda,
                mmr_candidate_pool=mmr_candidate_pool,
                filter_stopwords=filter_stopwords,
                filter_noise_subwords=filter_noise_subwords,
            )
            if not ids:
                ids = [0] * k
            elif len(ids) < k:
                ids = ids + [ids[-1]] * (k - len(ids))
            indices[node_id] = torch.tensor(ids[:k], dtype=torch.long, device=text_emb.device)
    finally:
        trainer.train(was_training)
    logger.info("Precomputed MMR targets for %d nodes", num_nodes)
    return indices


class TokenSelectionTrainer(nn.Module):
    """TokenSelector + NodeClassifier 训练包装。"""

    def __init__(
        self,
        token_selector: TokenSelector,
        node_classifier: NodeClassifier,
        tokenbook_emb: torch.Tensor,
        tfidf_table: torch.Tensor,
        predictor: Optional[nn.Module] = None,
        lambda_tfidf: float = 0.5,
        lambda_pred: float = 0.05,
        token_pred_tau: float = 1.0,
        p_code_normalize: str = "max",
        train_predictor: bool = False,
        top_k_hard: int = 8,
        candidate_pool: int = 256,
        kl_weight: float = 0.1,
        entropy_weight: float = 0.01,
        vtext_dropout: float = 0.3,
        selection_valid_mask: Optional[torch.Tensor] = None,
        training_mode: str = "distill",
        distill_weight: float = 1.0,
        cls_weight: float = 0.05,
        student_temperature: float = 1.0,
        target_topk_indices: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.token_selector = token_selector
        self.node_classifier = node_classifier
        self.register_buffer("tokenbook_emb", tokenbook_emb.float())
        self.register_buffer("tfidf_table", tfidf_table.float())
        if selection_valid_mask is not None:
            self.register_buffer(
                "selection_valid_mask",
                selection_valid_mask.bool(),
            )
        else:
            self.selection_valid_mask = None
        if target_topk_indices is not None:
            self.register_buffer("target_topk_indices", target_topk_indices.long())
        self.predictor = predictor
        self.lambda_tfidf = lambda_tfidf
        self.lambda_pred = lambda_pred
        self.token_pred_tau = token_pred_tau
        self.p_code_normalize = p_code_normalize
        self.train_predictor = train_predictor
        self.top_k_hard = top_k_hard
        self.candidate_pool = int(candidate_pool)
        self.kl_weight = float(kl_weight)
        self.entropy_weight = float(entropy_weight)
        self.vtext_dropout = float(vtext_dropout)
        self.training_mode = str(training_mode)
        self.distill_weight = float(distill_weight)
        self.cls_weight = float(cls_weight)
        self.student_temperature = float(student_temperature)

    def compute_s0(
        self,
        node_text_emb: torch.Tensor,
        struct_code_idx: torch.Tensor,
        z_q: torch.Tensor,
    ) -> torch.Tensor:
        return compute_initial_scores(
            node_text_emb,
            struct_code_idx,
            self.tokenbook_emb,
            self.tfidf_table,
            z_q,
            self.predictor,
            lambda_tfidf=self.lambda_tfidf,
            lambda_pred=self.lambda_pred,
            token_pred_tau=self.token_pred_tau,
            p_code_normalize=self.p_code_normalize,
            detach_p_code=not self.train_predictor,
        )

    def _valid_mask_for_batch(self, batch_size: int, device: torch.device) -> Optional[torch.Tensor]:
        if self.selection_valid_mask is None:
            return None
        return self.selection_valid_mask.to(device).unsqueeze(0).expand(batch_size, -1)

    def predict_hard_scores(self, s0: torch.Tensor) -> torch.Tensor:
        """推理：TokenSelector 调整得分并限制在 s0 候选池内，无 Gumbel。"""
        self.token_selector.eval()
        with torch.no_grad():
            s = self.token_selector(s0, self.tokenbook_emb)
            valid = self._valid_mask_for_batch(s0.shape[0], s0.device)
            return mask_scores_to_candidate_pool(
                s, s0, self.candidate_pool, valid_mask=valid
            )

    def forward_soft(
        self,
        node_text_emb: torch.Tensor,
        z_q: torch.Tensor,
        struct_code_idx: torch.Tensor,
        labels: torch.Tensor,
        tau: float,
        node_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        if self.training_mode == "distill":
            return self._forward_soft_distill(
                node_text_emb,
                z_q,
                struct_code_idx,
                labels,
                tau,
                node_indices=node_indices,
            )
        return self._forward_soft_cls(
            node_text_emb,
            z_q,
            struct_code_idx,
            labels,
            tau,
        )

    def _forward_soft_cls(
        self,
        node_text_emb: torch.Tensor,
        z_q: torch.Tensor,
        struct_code_idx: torch.Tensor,
        labels: torch.Tensor,
        tau: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        s0 = self.compute_s0(node_text_emb, struct_code_idx, z_q)
        s = self.token_selector(s0, self.tokenbook_emb)
        valid = self._valid_mask_for_batch(s0.shape[0], s0.device)
        mask = build_s0_candidate_mask(s0, self.candidate_pool, valid_mask=valid)
        s_pool = s.masked_fill(~mask, float("-inf"))
        w = F.gumbel_softmax(s_pool, tau=tau, hard=False, dim=-1)
        v_text = w @ self.tokenbook_emb
        logits = self.node_classifier(v_text, z_q, apply_vtext_dropout=True)
        cls_loss = F.cross_entropy(logits, labels)
        loss = cls_loss
        aux: Dict[str, float] = {"cls_loss": float(cls_loss.item())}

        if self.kl_weight > 0:
            p_s0 = s0_target_distribution(s0, mask)
            kl = selection_kl_loss(w, p_s0)
            loss = loss + self.kl_weight * kl
            aux["kl_loss"] = float(kl.item())

        if self.entropy_weight > 0:
            ent = selection_entropy(w)
            loss = loss - self.entropy_weight * ent
            aux["entropy"] = float(ent.item())

        aux["total_loss"] = float(loss.item())
        return loss, logits, aux

    def _forward_soft_distill(
        self,
        node_text_emb: torch.Tensor,
        z_q: torch.Tensor,
        struct_code_idx: torch.Tensor,
        labels: torch.Tensor,
        tau: float,
        node_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        if not hasattr(self, "target_topk_indices") or self.target_topk_indices is None:
            raise ValueError("distill mode requires target_topk_indices buffer")
        if node_indices is None:
            raise ValueError("distill mode requires node_indices for target lookup")

        s0 = self.compute_s0(node_text_emb, struct_code_idx, z_q)
        s = self.token_selector(s0, self.tokenbook_emb)
        valid = self._valid_mask_for_batch(s0.shape[0], s0.device)
        mask = build_s0_candidate_mask(s0, self.candidate_pool, valid_mask=valid)
        s_pool = s.masked_fill(~mask, float("-inf"))

        temp = max(self.student_temperature, 1e-6)
        p_student = F.softmax(s_pool / temp, dim=-1)
        topk_batch = self.target_topk_indices[node_indices]
        p_target = build_mmr_target_distribution(topk_batch, s0.shape[-1])
        distill_loss = selection_kl_loss(p_student, p_target.detach())

        w = F.gumbel_softmax(s_pool, tau=tau, hard=False, dim=-1)
        v_text = w @ self.tokenbook_emb
        logits = self.node_classifier(v_text, z_q, apply_vtext_dropout=True)
        cls_loss = F.cross_entropy(logits, labels)

        loss = self.distill_weight * distill_loss + self.cls_weight * cls_loss
        aux: Dict[str, float] = {
            "distill_loss": float(distill_loss.item()),
            "cls_loss": float(cls_loss.item()),
            "total_loss": float(loss.item()),
        }
        return loss, logits, aux

    @torch.no_grad()
    def forward_hard(
        self,
        node_text_emb: torch.Tensor,
        z_q: torch.Tensor,
        struct_code_idx: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        """硬 Top-k 选词 + 分类，用于验证。"""
        self.eval()
        s0 = self.compute_s0(node_text_emb, struct_code_idx, z_q)
        s = self.token_selector(s0, self.tokenbook_emb)
        valid = self._valid_mask_for_batch(s0.shape[0], s0.device)
        s = mask_scores_to_candidate_pool(
            s, s0, self.candidate_pool, valid_mask=valid
        )
        k = min(self.top_k_hard, s.shape[-1])
        topk_idx = s.topk(k, dim=-1).indices
        w_hard = torch.zeros_like(s)
        w_hard.scatter_(1, topk_idx, 1.0 / k)
        v_text = w_hard @ self.tokenbook_emb
        logits = self.node_classifier(v_text, z_q, apply_vtext_dropout=False)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        return loss, logits, topk_idx


def save_token_selector_checkpoint(
    path: Union[str, Path],
    token_selector: TokenSelector,
    node_classifier: Optional[NodeClassifier],
    config: Dict[str, Any],
    metrics: Optional[Dict[str, Any]] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "token_selector_state_dict": token_selector.state_dict(),
        "config": config,
        "metrics": metrics or {},
    }
    if node_classifier is not None:
        payload["node_classifier_state_dict"] = node_classifier.state_dict()
    torch.save(payload, path)
    meta_path = path.with_suffix(".json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {"config": config, "metrics": metrics or {}, "checkpoint": str(path)},
            f,
            indent=2,
            ensure_ascii=False,
        )


def load_token_selector(
    path: Union[str, Path],
    device: torch.device,
) -> TokenSelector:
    """加载 TokenSelector（推理/preprocess 用）。"""
    path = Path(path)
    payload = torch.load(path, map_location=device)
    config = payload.get("config", {})
    d_text = int(config["d_text"])
    hidden_dim = int(config.get("hidden_dim", 128))
    selector = TokenSelector(d_text=d_text, hidden_dim=hidden_dim).to(device)
    selector.load_state_dict(payload["token_selector_state_dict"])
    selector.eval()
    selector.candidate_pool = int(config.get("candidate_pool", 256))
    return selector


def load_token_selection_trainer(
    path: Union[str, Path],
    num_classes: int,
    d_struct: int,
    tokenbook_emb: torch.Tensor,
    tfidf_table: torch.Tensor,
    predictor: Optional[nn.Module],
    device: torch.device,
    cfg: Optional[Config] = None,
    id_to_token: Optional[Dict[int, str]] = None,
) -> TokenSelectionTrainer:
    """加载完整训练模块（评估脚本用）。"""
    cfg = cfg or get_config()
    path = Path(path)
    payload = torch.load(path, map_location=device)
    config = payload.get("config", {})
    d_text = int(config.get("d_text", tokenbook_emb.shape[1]))
    hidden_dim = int(config.get("hidden_dim", cfg.token_selector_hidden_dim))

    selector = TokenSelector(d_text=d_text, hidden_dim=hidden_dim).to(device)
    classifier = NodeClassifier(
        d_text,
        d_struct,
        num_classes,
        vtext_dropout=float(config.get("vtext_dropout", cfg.token_selector_vtext_dropout)),
    ).to(device)
    selector.load_state_dict(payload["token_selector_state_dict"])
    if "node_classifier_state_dict" in payload:
        classifier.load_state_dict(payload["node_classifier_state_dict"])

    selection_valid_mask = None
    if config.get("filter_noise_subwords_at_selection", cfg.filter_noise_subwords_at_selection):
        if id_to_token is not None:
            from graph_utils import build_selection_valid_mask

            vocab_size = tokenbook_emb.shape[0]
            valid_np = build_selection_valid_mask(
                id_to_token,
                vocab_size,
                filter_stopwords=False,
                filter_noise_subwords=True,
            )
            selection_valid_mask = torch.tensor(valid_np, dtype=torch.bool)

    trainer = TokenSelectionTrainer(
        token_selector=selector,
        node_classifier=classifier,
        tokenbook_emb=tokenbook_emb,
        tfidf_table=tfidf_table,
        predictor=predictor,
        lambda_tfidf=float(config.get("lambda_tfidf", cfg.lambda_tfidf)),
        lambda_pred=float(config.get("lambda_pred", cfg.lambda_pred)),
        token_pred_tau=float(config.get("token_pred_tau", cfg.token_pred_temperature)),
        p_code_normalize=str(config.get("p_code_normalize", cfg.p_code_normalize)),
        train_predictor=False,
        top_k_hard=int(config.get("top_k_hard", cfg.top_k_text_tokens)),
        candidate_pool=int(config.get("candidate_pool", cfg.token_selector_candidate_pool)),
        kl_weight=float(config.get("kl_weight", cfg.token_selector_kl_weight)),
        entropy_weight=float(config.get("entropy_weight", cfg.token_selector_entropy_weight)),
        vtext_dropout=float(config.get("vtext_dropout", cfg.token_selector_vtext_dropout)),
        selection_valid_mask=selection_valid_mask,
        training_mode=str(config.get("training_mode", cfg.token_selector_training_mode)),
        distill_weight=float(config.get("distill_weight", cfg.token_selector_distill_weight)),
        cls_weight=float(config.get("cls_weight", cfg.token_selector_cls_weight)),
        student_temperature=float(
            config.get("student_temperature", cfg.token_selector_student_temperature)
        ),
    ).to(device)
    trainer.eval()
    return trainer
