"""
标准 GCN 节点分类基线（无 VQ / 无语义偏置），用于与结构码本、LLM 微调对比。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv


class GCNClassifier(nn.Module):
    """两层 GraphConv + 线性分类头（transductive 全图训练）。"""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.convs = nn.ModuleList()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1)
        for i in range(num_layers):
            out_ch = hidden_dim if i < num_layers - 1 else hidden_dim
            self.convs.append(GraphConv(dims[i], out_ch, activation=F.relu))
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, g, feats: torch.Tensor) -> torch.Tensor:
        h = feats
        for i, conv in enumerate(self.convs):
            h = conv(g, h)
            if i < self.num_layers - 1:
                h = self.dropout(h)
        h = self.dropout(h)
        return self.classifier(h)
