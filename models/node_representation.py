"""
模块3：节点表示离散化 — Top-K 文本 token + 1 个结构 token。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from config import Config, get_config
from models.structural_codebook import CodebookArtifacts, TFIDFStatistics
from models.subgraph_extraction import LocalSubgraph

logger = logging.getLogger(__name__)


@dataclass
class NodeTokenRepresentation:
    """单节点联合离散表示。"""

    node_id: int
    struct_token: str  # 如 "<S_15>"
    text_tokens: List[str]  # Top-K 文本 token 字符串
    struct_code_idx: Optional[int] = None
    text_token_ids: Optional[List[int]] = None


@dataclass
class SubgraphTokenizedView:
    """子图内所有节点的离散表示。"""

    center_node: int
    nodes: Dict[int, NodeTokenRepresentation]  # global node_id -> repr

    def to_dict_list(self) -> List[Dict[str, Any]]:
        return [
            {
                "node_id": r.node_id,
                "struct_token": r.struct_token,
                "text_tokens": r.text_tokens,
            }
            for r in self.nodes.values()
        ]


from text_tokenizers.text_tokenbook import TextTokenbook  # noqa: F401 — 模块3 使用


class NodeRepresentationTokenizer:
    """
    对子图中每个节点生成 text_tokens + struct_token。

    文本筛选::
        score[t] = text_sim[t] * (1 + λ_tfidf * TF-IDF_norm[c][t])
    未见结构码时退化为纯 text_sim。
    """

    def __init__(
        self,
        artifacts: CodebookArtifacts,
        tokenbook: TextTokenbook,
        tfidf: Optional[TFIDFStatistics] = None,
        cfg: Optional[Config] = None,
    ) -> None:
        self.cfg = cfg or get_config()
        self.artifacts = artifacts
        self.tokenbook = tokenbook
        self.tfidf = tfidf
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.encoder: Optional[nn.Module] = None

    def load_encoder(self) -> nn.Module:
        """加载模块1 训练好的 GNN 编码器。"""
        raise NotImplementedError

    def assign_struct_codes(
        self,
        structural_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        结构向量与码本余弦相似度，取 argmax 作为结构码索引。

        Returns
        -------
        indices [N_sub]
        """
        raise NotImplementedError

    def select_text_tokens(
        self,
        node_text_emb: torch.Tensor,
        struct_code_idx: int,
        top_k: Optional[int] = None,
    ) -> List[str]:
        """
        结构引导的 Top-K 文本 token 选取。

        Parameters
        ----------
        node_text_emb : [D] 单节点文本嵌入
        struct_code_idx : 结构码 c
        """
        raise NotImplementedError

    def _compute_text_similarity(
        self,
        node_text_emb: torch.Tensor,
    ) -> np.ndarray:
        """节点嵌入与 tokenbook 余弦相似度，归一化到 [0,1]，shape [V]。"""
        raise NotImplementedError

    def _fuse_scores(
        self,
        text_sim: np.ndarray,
        struct_code_idx: int,
    ) -> np.ndarray:
        """
        score[t] = text_sim[t] * (1 + λ * TF-IDF_norm[c][t])；
        异常码索引或空 TF-IDF 行时退化为 text_sim。
        """
        raise NotImplementedError

    def format_struct_token(self, code_idx: int) -> str:
        prefix = self.cfg.struct_token_prefix
        return f"{prefix}{code_idx}>"

    def tokenize_node(
        self,
        node_id: int,
        graph: Any,
        node_features: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> NodeTokenRepresentation:
        """对单节点（需全图上下文编码）生成表示。"""
        raise NotImplementedError

    def tokenize_subgraph(
        self,
        subgraph: LocalSubgraph,
        graph: Any,
        node_features: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> SubgraphTokenizedView:
        """
        子图内所有节点批量编码并离散化。

        典型流程：GNN 前向 -> 结构码 -> 文本 Top-K。
        """
        raise NotImplementedError
