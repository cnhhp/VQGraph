"""
语义偏置向量量化：在余弦结构码分配上叠加 λ·sim(t_i, μ_k)，并 EMA 更新语义中心。
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import einsum

from vq import (
    CosineSimCodebook,
    VectorQuantize,
    batched_embedding,
    exists,
    gumbel_sample,
    l2norm,
)


class SemanticVectorQuantize(nn.Module):
    """
    包装现有 VectorQuantize，在码字分配时引入语义偏置。

    分配（实现为 dist 最大化）::
        effective = cos_sim(h, e) - struct_norm_cost + λ_eff * sem_sim
    其中 struct_norm_cost 由 1-cos_sim 逐节点 min-max 到 [0,1]。
    预热期 λ_eff=0，退化为原版余弦 VQ。
    """

    def __init__(
        self,
        vq: VectorQuantize,
        text_dim: int,
        lambda_semantic: float = 0.1,
        ema_beta: float = 0.99,
    ) -> None:
        super().__init__()
        if not isinstance(vq._codebook, CosineSimCodebook):
            raise TypeError("SemanticVectorQuantize requires use_cosine_sim=True")
        self.vq = vq
        self.text_dim = text_dim
        self.lambda_semantic = lambda_semantic
        self.ema_beta = ema_beta
        self._lambda_effective = 0.0
        M = vq.codebook_size
        self.register_buffer(
            "semantic_centers",
            torch.zeros(M, text_dim),
        )

    @property
    def codebook_size(self) -> int:
        return self.vq.codebook_size

    @property
    def codebook(self) -> torch.Tensor:
        return self.vq.codebook

    def set_lambda_effective(self, lam: float) -> None:
        self._lambda_effective = float(lam)

    def _semantic_similarity_matrix(
        self, text_emb: torch.Tensor
    ) -> torch.Tensor:
        """text_emb [B, D] -> sem_sim [B, M] in [0, 1]."""
        t = l2norm(text_emb.float())
        mu = l2norm(self.semantic_centers.float())
        cos = torch.matmul(t, mu.t())
        return (cos + 1.0) * 0.5

    def _apply_semantic_bias(
        self,
        struct_dist: torch.Tensor,
        text_emb: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        struct_dist: 余弦相似度 [*, M]，越大越优。
        转换为 struct_norm 代价后叠加语义项。
        """
        if text_emb is None or self._lambda_effective <= 0:
            return struct_dist

        # struct_cost = 1 - cos_sim，逐行归一化到 [0,1]
        struct_cost = 1.0 - struct_dist
        if struct_cost.dim() == 1:
            struct_cost = struct_cost.unsqueeze(0)
            struct_dist = struct_dist.unsqueeze(0)
        row_min = struct_cost.min(dim=-1, keepdim=True).values
        row_max = struct_cost.max(dim=-1, keepdim=True).values
        struct_norm = (struct_cost - row_min) / (row_max - row_min + 1e-8)

        sem = self._semantic_similarity_matrix(text_emb)
        if sem.shape[0] != struct_norm.shape[0]:
            # 对齐 batch 维（SAGE 子图 batch）
            if sem.shape[0] == 1:
                sem = sem.expand(struct_norm.shape[0], -1)
            elif struct_norm.shape[0] == 1:
                struct_norm = struct_norm.expand(sem.shape[0], -1)
                struct_dist = struct_dist.expand(sem.shape[0], -1)

        # 最大化: -struct_norm + λ * sem
        effective = -struct_norm + self._lambda_effective * sem
        return effective

    @torch.no_grad()
    def update_semantic_centers(
        self,
        indices: torch.Tensor,
        text_emb: torch.Tensor,
    ) -> None:
        """对本 batch 内各码本的文本嵌入均值做 EMA 更新 μ_k。"""
        if text_emb is None:
            return
        idx = indices.view(-1).long()
        te = text_emb.view(-1, text_emb.shape[-1]).float()
        if idx.shape[0] != te.shape[0]:
            te = te[: idx.shape[0]]
        beta = self.ema_beta
        for k in idx.unique().tolist():
            mask = idx == k
            if not mask.any():
                continue
            mean_t = te[mask].mean(dim=0).to(self.semantic_centers.device)
            self.semantic_centers[k].mul_(beta).add_(mean_t * (1.0 - beta))

    def _codebook_forward_semantic(
        self,
        x: torch.Tensor,
        text_emb: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """CosineSimCodebook forward，分配前注入语义偏置。"""
        cb: CosineSimCodebook = self.vq._codebook
        needs_codebook_dim = x.ndim < 4
        x = x.float()
        if needs_codebook_dim:
            x = rearrange(x, "... -> 1 ...")
        shape, dtype = x.shape, x.dtype
        flatten = rearrange(x, "h ... d -> h (...) d")
        flatten = l2norm(flatten)
        cb.init_embed_(flatten)
        embed = cb.embed if not cb.learnable_codebook else cb.embed.detach()
        embed = l2norm(embed)
        dist = einsum("h n d, h c d -> h n c", flatten, embed)
        # 语义偏置（在 gumbel/argmax 之前）
        if text_emb is not None and dist.numel() > 0:
            flat_n = dist.shape[1]
            te = text_emb
            if te.shape[0] != flat_n:
                if te.shape[0] == 1:
                    te = te.expand(flat_n, -1)
                else:
                    te = te[:flat_n]
            # dist: [h, n, c] -> 对 n 维逐节点偏置
            h_dim = dist.shape[0]
            biased_list = []
            for hi in range(h_dim):
                d_hi = dist[hi]  # [n, c]
                biased_list.append(
                    self._apply_semantic_bias(d_hi, te)
                )
            dist = torch.stack(biased_list, dim=0)

        embed_ind = gumbel_sample(
            dist, dim=-1, temperature=cb.sample_codebook_temp
        )
        embed_onehot = F.one_hot(embed_ind, cb.codebook_size).type(dtype)
        embed_ind = embed_ind.view(*shape[:-1])
        quantize = batched_embedding(embed_ind, cb.embed)

        if self.training:
            bins = embed_onehot.sum(dim=1)
            cb.all_reduce_fn(bins)
            cb.cluster_size.data.lerp_(bins, 1 - cb.decay)
            zero_mask = bins == 0
            bins = bins.masked_fill(zero_mask, 1.0)
            embed_sum = einsum("h n d, h n c -> h c d", flatten, embed_onehot)
            cb.all_reduce_fn(embed_sum)
            embed_normalized = embed_sum / rearrange(bins, "... -> ... 1")
            embed_normalized = l2norm(embed_normalized)
            embed_normalized = torch.where(
                rearrange(zero_mask, "... -> ... 1"),
                embed,
                embed_normalized,
            )
            cb.embed.data.lerp_(embed_normalized, 1 - cb.decay)
            cb.expire_codes_(x)

        if needs_codebook_dim:
            quantize, embed_ind = map(
                lambda t: rearrange(t, "1 ... -> ..."), (quantize, embed_ind)
            )
        return quantize, embed_ind, dist, cb.embed

    def forward(
        self,
        x: torch.Tensor,
        text_emb: Optional[torch.Tensor] = None,
        mask=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        与 VectorQuantize 相同返回值:
        quantize, embed_ind, loss, dist, embed
        """
        vq = self.vq
        only_one = x.ndim == 2
        if only_one:
            x = rearrange(x, "b d -> b 1 d")
        shape, device = x.shape, x.device
        heads = vq.heads
        is_multiheaded = heads > 1
        need_transpose = not vq.channel_last and not vq.accept_image_fmap
        if vq.accept_image_fmap:
            height, width = x.shape[-2:]
            x = rearrange(x, "b c h w -> b (h w) c")
        if need_transpose:
            x = rearrange(x, "b d n -> b n d")
        x = vq.project_in(x)
        if is_multiheaded:
            ein_rhs_eq = (
                "h b n d" if vq.separate_codebook_per_head else "1 (b h) n d"
            )
            x = rearrange(x, f"b n (h d) -> {ein_rhs_eq}", h=heads)

        quantize, embed_ind, dist, embed = self._codebook_forward_semantic(
            x, text_emb
        )

        if self.training and text_emb is not None:
            self.update_semantic_centers(embed_ind, text_emb)

        if vq.training:
            quantize = x + (quantize - x).detach()

        loss = torch.tensor([0.0], device=device, requires_grad=vq.training)
        if vq.training and vq.commitment_weight > 0:
            detached_quantize = quantize.detach()
            if exists(mask):
                commit_loss = F.mse_loss(detached_quantize, x, reduction="none")
                if is_multiheaded:
                    from einops import repeat

                    mask = repeat(
                        mask,
                        "b n -> c (b h) n",
                        c=commit_loss.shape[0],
                        h=commit_loss.shape[1] // mask.shape[0],
                    )
                commit_loss = commit_loss[mask].mean()
            else:
                commit_loss = F.mse_loss(detached_quantize, x)
            loss = loss + commit_loss * vq.commitment_weight

        if is_multiheaded:
            if vq.separate_codebook_per_head:
                quantize = rearrange(quantize, "h b n d -> b n (h d)", h=heads)
                embed_ind = rearrange(embed_ind, "h b n -> b n h", h=heads)
            else:
                quantize = rearrange(quantize, "1 (b h) n d -> b n (h d)", h=heads)
                embed_ind = rearrange(embed_ind, "1 (b h) n -> b n h", h=heads)

        quantize = vq.project_out(quantize)
        if need_transpose:
            quantize = rearrange(quantize, "b n d -> b d n")
        if only_one:
            quantize = rearrange(quantize, "b 1 d -> b d")
            embed_ind = rearrange(embed_ind, "b 1 -> b")
            if dist.dim() >= 2 and dist.shape[1] == 1:
                dist = rearrange(dist, "h 1 c -> h c")

        return quantize, embed_ind, loss, dist, embed
