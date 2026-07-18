"""
层次粗/细码编码器：双通道聚合 + SemanticVQ(粗) + 细码 VQ + L_H/L_D。

v2 改进（相对首版）：
- 减弱同标签过度平滑，保留 h_L 残差
- 细码直接量化 h_S（两通道已解耦），不再用 detach(z_co) 残差掐断粗码梯度
- 分类头融合 z_co+z_fi，让粗码参与主 CE 反传
- 码本使用率 KL（防塌缩）
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv

from models.semantic_vq import SemanticVectorQuantize
from models.token_predictor import attach_token_predictor_to_encoder, maybe_add_token_loss
from vq import VectorQuantize


def _squeeze_dist(dist: torch.Tensor) -> torch.Tensor:
    dist = torch.squeeze(dist)
    if dist.dim() == 3 and dist.shape[0] == 1:
        dist = dist.squeeze(0)
    return dist


def _to_nm(dist: torch.Tensor, n: int) -> torch.Tensor:
    """Normalize VQ dist to [N, M]."""
    d = _squeeze_dist(dist)
    if d.dim() == 1:
        d = d.unsqueeze(0)
    if d.dim() == 2 and d.shape[0] != n and d.shape[1] == n:
        d = d.t()
    if d.dim() == 2 and d.shape[0] == 1 and n > 1:
        # ambiguous; leave
        pass
    return d


def codebook_usage_kl(dist_nm: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
    """
    KL(mean_softmax(dist/τ) || Uniform)，鼓励码本均匀使用。
    dist 越大越优（余弦分数）。
    """
    if dist_nm.dim() != 2 or dist_nm.shape[0] < 2:
        return dist_nm.new_zeros(())
    soft = F.softmax(dist_nm / max(tau, 1e-4), dim=-1)
    avg = soft.mean(dim=0).clamp(min=1e-8)
    m = avg.numel()
    # KL(avg || U) = sum avg * (log avg - log(1/M))
    return torch.sum(avg * (avg.log() + torch.log(torch.tensor(float(m), device=avg.device))))


def compute_structure_edge_losses(
    z_edge: torch.Tensor,
    adj: torch.Tensor,
    labels: torch.Tensor,
    tau: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    细码边损失（替换全图 min-max MSE）：
    - 正边：拉高 cosine sim；等量负采样：压低 sim
    - L_H：同配/异配正边损失平衡 (L_inter - L_intra)^2
    """
    n = z_edge.shape[0]
    device = z_edge.device
    z_n = F.normalize(z_edge, dim=-1)
    sim = torch.matmul(z_n, z_n.t())
    eye = torch.eye(n, dtype=torch.bool, device=device)
    adj_bool = (adj > 0.5) & ~eye
    label_eq = labels.view(n, 1) == labels.view(1, n)
    intra = adj_bool & label_eq
    inter = adj_bool & ~label_eq

    def _pos_loss(mask: torch.Tensor) -> torch.Tensor:
        if not mask.any():
            return z_edge.new_zeros(())
        return F.softplus(-sim[mask] / tau).mean()

    l_intra = _pos_loss(intra)
    l_inter = _pos_loss(inter)
    l_pos = _pos_loss(adj_bool)

    neg = ~adj_bool & ~eye
    n_pos = int(adj_bool.sum().item())
    if n_pos > 0 and neg.any():
        neg_idx = neg.nonzero(as_tuple=False)
        n_neg = min(n_pos, neg_idx.shape[0])
        choice = torch.randperm(neg_idx.shape[0], device=device)[:n_neg]
        sel = neg_idx[choice]
        l_neg = F.softplus(sim[sel[:, 0], sel[:, 1]] / tau).mean()
    else:
        l_neg = z_edge.new_zeros(())

    edge_rec = l_pos + l_neg
    if (not intra.any()) or (not inter.any()):
        l_h = z_edge.new_zeros(())
    else:
        # 平衡 + 轻微强调异配边也要重建好
        l_h = (l_inter - l_intra).pow(2) + 0.25 * l_inter
    return edge_rec, l_h, l_intra, l_inter


def compute_intra_co_loss(
    z_fi: torch.Tensor,
    idx_co: torch.Tensor,
    adj: torch.Tensor,
    margin: float = 0.2,
) -> torch.Tensor:
    """同粗码邻居的细码应可分：推低 cosine（margin hinge）。"""
    n = z_fi.shape[0]
    device = z_fi.device
    eye = torch.eye(n, dtype=torch.bool, device=device)
    adj_bool = (adj > 0.5) & ~eye
    same_co = (idx_co.view(n, 1) == idx_co.view(1, n)) & adj_bool
    if not same_co.any():
        return z_fi.new_zeros(())
    z_n = F.normalize(z_fi, dim=-1)
    sim = torch.matmul(z_n, z_n.t())
    return F.relu(sim[same_co] - margin).mean()


class DualChannelAgg(nn.Module):
    """
    双通道聚合（减弱邻居权重，保留自身残差，缓解 Cora 同配过度平滑）。
    """

    def __init__(self, dim: int, dropout: float = 0.0, nbr_scale: float = 0.25):
        super().__init__()
        self.w_i_l = nn.Linear(dim, dim, bias=False)
        self.w_n_l = nn.Linear(dim, dim, bias=False)
        self.w_i_s = nn.Linear(dim, dim, bias=False)
        self.w_low = nn.Linear(dim, dim, bias=False)
        self.w_high = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()
        self.nbr_scale = nbr_scale

    def forward(
        self,
        h_l: torch.Tensor,
        h_s: torch.Tensor,
        adj: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n = h_l.shape[0]
        device = h_l.device
        label_eq = labels.view(n, 1) == labels.view(1, n)
        eye = torch.eye(n, dtype=torch.bool, device=device)
        adj_bool = (adj > 0) & ~eye

        same = (adj_bool & label_eq).float()
        diff = (adj_bool & ~label_eq).float()

        deg_s = same.sum(dim=1, keepdim=True).clamp(min=1.0)
        deg_d = diff.sum(dim=1, keepdim=True).clamp(min=1.0)
        ns = self.nbr_scale

        msg_l = (same @ self.w_n_l(h_l)) / deg_s
        h_l_out = self.act(self.w_i_l(h_l) + ns * msg_l) + 0.5 * h_l
        h_l_out = self.dropout(h_l_out)

        msg_low = (same @ self.w_low(h_s)) / deg_s
        msg_high = (diff @ self.w_high(-h_s)) / deg_d
        h_s_out = self.act(self.w_i_s(h_s) + ns * msg_low + ns * msg_high) + 0.5 * h_s
        # 结构通道不 dropout：避免 train unique 虚高、eval 细码塌缩
        return h_l_out, h_s_out


class HierarchicalGCN(nn.Module):
    """双通道 GCN + 粗 SemanticVQ + 细 VQ（结构通道）。"""

    def __init__(
        self,
        num_layers: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout_ratio: float,
        activation,
        norm_type: str,
        codebook_size_coarse: int,
        codebook_size_fine: int,
        lamb_edge: float,
        lamb_node: float,
        lambda_H: float = 0.5,
        lambda_D: float = 0.05,
        lambda_L: float = 0.1,
        lambda_div: float = 0.1,
        lambda_div_fi: float = 0.5,
        lambda_ico: float = 0.2,
        lambda_semantic: float = 0.3,
        text_dim: int = 384,
        ema_beta: float = 0.99,
        text_fuse: float = 0.5,
        fine_noise: float = 0.0,
        select_min_s_L: float = 0.75,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.norm_type = norm_type
        self.dropout = nn.Dropout(dropout_ratio)
        self.lamb_edge = lamb_edge
        self.lamb_node = lamb_node
        self.lambda_H = lambda_H
        self.lambda_D = lambda_D
        self.lambda_L = lambda_L
        self.lambda_div = lambda_div
        self.lambda_div_fi = lambda_div_fi
        self.lambda_ico = lambda_ico
        self.text_fuse = text_fuse
        self.fine_noise = fine_noise
        self.select_min_s_L = select_min_s_L
        self.codebook_size_coarse = codebook_size_coarse
        self.codebook_size_fine = codebook_size_fine
        self.codebook_size = codebook_size_fine

        self.graph_layer_1 = GraphConv(input_dim, input_dim, activation=activation)
        self.proj_l = nn.Linear(input_dim, input_dim)
        self.proj_s = nn.Linear(input_dim, input_dim)
        self.text_proj = nn.Linear(text_dim, input_dim)
        self.dual_agg = DualChannelAgg(input_dim, dropout=dropout_ratio, nbr_scale=0.25)
        self.struct_norm = nn.LayerNorm(input_dim)

        vq_co_raw = VectorQuantize(
            dim=input_dim,
            codebook_size=codebook_size_coarse,
            decay=0.8,
            commitment_weight=1.0,
            use_cosine_sim=True,
            threshold_ema_dead_code=2,
        )
        self.vq_co = SemanticVectorQuantize(
            vq_co_raw,
            text_dim=text_dim,
            lambda_semantic=lambda_semantic,
            ema_beta=ema_beta,
        )
        # 细码：更快 EMA + 死码复活 + 更强 commit
        self.vq_fi = VectorQuantize(
            dim=input_dim,
            codebook_size=codebook_size_fine,
            decay=0.6,
            commitment_weight=1.0,
            use_cosine_sim=True,
            threshold_ema_dead_code=2,
        )
        self.vq = self.vq_fi

        self.fuse = nn.Linear(input_dim * 2, input_dim)
        self.decoder_edge = nn.Linear(input_dim, input_dim)
        self.decoder_node = nn.Linear(input_dim, input_dim)
        self.graph_layer_2 = GraphConv(input_dim, hidden_dim, activation=activation)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.label_head = nn.Linear(input_dim, output_dim)

        self.token_predictor: Optional[nn.Module] = None
        self.lambda_token: float = 0.0
        self.token_pred_tau: float = 1.0
        self.token_target_tau: float = 0.15
        self.token_kl_top_k: int = 0

        self._last_assign: Dict[str, Any] = {}
        self._last_loss_aux: Dict[str, float] = {}
        self._lambda_H_eff: float = float(lambda_H)

    def set_lambda_H_effective(self, lam: float) -> None:
        self._lambda_H_eff = float(lam)

    def attach_token_predictor(
        self,
        vocab_size: int,
        tokenbook_embeddings: torch.Tensor,
        cfg: Union[object, dict],
    ) -> None:
        attach_token_predictor_to_encoder(self, vocab_size, tokenbook_embeddings, cfg)

    def forward(
        self,
        g,
        feats: torch.Tensor,
        text_emb: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        if labels is None:
            raise ValueError(
                "HierarchicalGCN requires labels for dual-channel aggregation and L_H"
            )
        labels = labels.to(feats.device).long()
        n = feats.shape[0]
        adj = g.adjacency_matrix().to_dense().to(feats.device)
        h_list = []

        h0 = self.graph_layer_1(g, feats)
        h0 = self.dropout(h0)
        h_l = self.proj_l(h0)
        h_s = self.proj_s(h0)
        # 文本注入粗通道，拉开 theme 差异（clone：Sentence-BERT 输出可能是 inference tensor）
        if text_emb is not None and self.text_fuse > 0:
            te = text_emb.detach().clone()
            h_l = h_l + self.text_fuse * self.text_proj(te)
        h_l, h_s = self.dual_agg(h_l, h_s, adj, labels)
        h_list.append(torch.cat([h_l, h_s], dim=-1))

        if text_emb is not None:
            z_co, idx_co, commit_co, dist_co, codebook_co = self.vq_co(
                h_l, text_emb=text_emb
            )
        else:
            z_co, idx_co, commit_co, dist_co, codebook_co = self.vq_co(h_l)

        # 细码：结构通道 LayerNorm + 训练噪声拉开表示，再 VQ
        h_s_q = self.struct_norm(h_s)
        if self.training and self.fine_noise > 0:
            h_s_q = h_s_q + self.fine_noise * torch.randn_like(h_s_q)
        z_fi, idx_fi, commit_fi, dist_fi, codebook_fi = self.vq_fi(h_s_q)

        # 原定：边重建 / L_H 只绑细码
        z_edge = self.decoder_edge(z_fi)
        z_node = self.decoder_node(z_fi)
        z_fused = self.fuse(torch.cat([z_co, z_fi], dim=-1))

        node_rec = self.lamb_node * F.mse_loss(h_s, z_node)
        edge_raw, l_h, l_intra, l_inter = compute_structure_edge_losses(
            z_edge, adj, labels
        )
        edge_rec = self.lamb_edge * edge_raw

        if idx_co.dim() > 1:
            idx_co_flat = idx_co.view(-1)
        else:
            idx_co_flat = idx_co
        l_ico = compute_intra_co_loss(z_fi, idx_co_flat, adj)

        cos_ls = F.cosine_similarity(h_l, h_s, dim=-1)
        l_d = (cos_ls * cos_ls).mean()

        logits_l = self.label_head(h_l)
        l_l = F.cross_entropy(logits_l, labels)  # 弱 L_L：只塑形 h_L，非主分类 CE

        dist_co_nm = _to_nm(dist_co, n)
        dist_fi_nm = _to_nm(dist_fi, n)
        l_div_co = codebook_usage_kl(dist_co_nm, tau=0.5)
        l_div_fi = codebook_usage_kl(dist_fi_nm, tau=0.2) + 0.5 * codebook_usage_kl(
            dist_fi_nm, tau=1.0
        )

        loss = (
            node_rec
            + edge_rec
            + commit_co
            + commit_fi
            + self._lambda_H_eff * l_h
            + self.lambda_D * l_d
            + self.lambda_L * l_l
            + self.lambda_div * l_div_co
            + self.lambda_div_fi * l_div_fi
            + self.lambda_ico * l_ico
        )

        # 主分类头：融合粗+细，粗码获得 CE 梯度
        h2 = self.graph_layer_2(g, z_edge)
        h_list.append(z_fused)
        h_list.append(h2)
        logits = self.linear(h2)

        token_loss_weighted, token_loss_raw = maybe_add_token_loss(
            self, z_co, text_emb
        )
        loss = loss + token_loss_weighted

        dist_fi_out = _squeeze_dist(dist_fi)
        dist_co_out = _squeeze_dist(dist_co)
        if idx_co.dim() > 1:
            idx_co = idx_co.view(-1)
        if idx_fi.dim() > 1:
            idx_fi = idx_fi.view(-1)

        n_co = int(torch.unique(idx_co).numel())
        n_fi = int(torch.unique(idx_fi).numel())

        self._last_assign = {
            "idx_co": idx_co.detach(),
            "idx_fi": idx_fi.detach(),
            "logits_l": logits_l.detach(),
            "dist_co": dist_co_out.detach()
            if isinstance(dist_co_out, torch.Tensor)
            else dist_co_out,
            "dist_fi": dist_fi_out.detach()
            if isinstance(dist_fi_out, torch.Tensor)
            else dist_fi_out,
            "codebook_co": codebook_co.detach()
            if isinstance(codebook_co, torch.Tensor)
            else codebook_co,
            "codebook_fi": codebook_fi.detach()
            if isinstance(codebook_fi, torch.Tensor)
            else codebook_fi,
        }
        self._last_loss_aux = {
            "node_rec": float(node_rec.detach().item()),
            "edge_rec": float(edge_rec.detach().item()),
            "commit_co": float(commit_co.detach().item())
            if commit_co.numel()
            else 0.0,
            "commit_fi": float(commit_fi.detach().item())
            if commit_fi.numel()
            else 0.0,
            "L_H": float(l_h.detach().item()),
            "L_D": float(l_d.detach().item()),
            "L_L": float(l_l.detach().item()),
            "L_div": float(
                (self.lambda_div * l_div_co + self.lambda_div_fi * l_div_fi)
                .detach()
                .item()
            ),
            "L_div_fi": float(l_div_fi.detach().item()),
            "L_ico": float(l_ico.detach().item()),
            "L_intra": float(l_intra.detach().item()),
            "L_inter": float(l_inter.detach().item()),
            "lambda_H_eff": float(self._lambda_H_eff),
            "unique_co": float(n_co),
            "unique_fi": float(n_fi),
        }

        return h_list, logits, loss, dist_fi_out, codebook_fi, token_loss_raw
