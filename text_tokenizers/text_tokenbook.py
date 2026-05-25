"""
文本 tokenbook 准备（离线构建词表与嵌入矩阵）。

供模块3 结构引导文本 token 筛选使用。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch

from config import Config, get_config

logger = logging.getLogger(__name__)


class TextTokenbook:
    """token 词表 + 嵌入矩阵 [V, D]。"""

    def __init__(self) -> None:
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.embeddings: Optional[torch.Tensor] = None

    def __len__(self) -> int:
        return len(self.token_to_id)

    def load(self, path: Path) -> "TextTokenbook":
        raise NotImplementedError

    def save(self, path: Path) -> None:
        raise NotImplementedError

    def get_embedding_matrix(self) -> torch.Tensor:
        if self.embeddings is None:
            raise RuntimeError("Tokenbook embeddings not loaded.")
        return self.embeddings


class TextTokenbookBuilder:
    """从语料 / 节点文本构建 tokenbook。"""

    def __init__(self, cfg: Optional[Config] = None) -> None:
        self.cfg = cfg or get_config()

    def build_from_corpus(
        self,
        texts: List[str],
        vocab_size: Optional[int] = None,
        embedding_model: Optional[str] = None,
    ) -> TextTokenbook:
        raise NotImplementedError

    def tokenize_text(self, text: str) -> List[str]:
        raise NotImplementedError
