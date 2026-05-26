"""
全局超参数配置。
所有模块通过 ``from config import Config`` 或 ``get_config()`` 读取，避免硬编码。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class Config:
    """面向大模型推理的离散图表征 — 统一配置。"""

    # ---------- 路径 ----------
    project_root: Path = field(default_factory=lambda: Path("."))
    data_root: Path = field(default_factory=lambda: Path("./data"))
    output_root: Path = field(default_factory=lambda: Path("./outputs"))
    node_text_csv: Optional[Path] = None  # 节点 ID -> 原始文本
    tokenbook_path: Optional[Path] = field(default_factory=lambda: Path("./codebook"))
    tokenbook_vocab_filename: str = "filtered_tokenbook.npy"
    tokenbook_embeddings_filename: str = "token_embeddings.npz"
    tokenbook_meta_filename: str = "tokenbook_meta.json"
    codebook_checkpoint_dir: Optional[Path] = None

    # ---------- 数据集 ----------
    dataset_name: str = "cora"
    sentence_bert_model: str = "all-MiniLM-L6-v2"

    # ---------- 模块1：结构码本 ----------
    codebook_size: int = 2048  # M
    codebook_dim: int = 256  # d
    gnn_type: str = "GCN"  # GCN | SAGE
    lambda_semantic: float = 0.1
    warmup_epochs: int = 20
    ema_beta: float = 0.99  # 语义中心 μ_k 的 EMA 衰减
    codebook_train_epochs: int = 100
    codebook_lr: float = 1e-3
    codebook_batch_size: int = -1  # -1 表示全图
    use_contrastive_loss: bool = False
    contrastive_weight: float = 0.1

    # ---------- 模块2：子图提取 ----------
    subgraph_k_hop: int = 1
    max_subgraph_nodes: Optional[int] = None  # None 表示不截断

    # ---------- 模块3：节点离散化 ----------
    text_vocab_size: int = 13648  # V（filtered_tokenbook.npy）
    top_k_text_tokens: int = 5  # K
    lambda_tfidf: float = 0.5
    struct_token_prefix: str = "<S_"

    # ---------- 模块4：序列化 ----------
    bias_alpha: float = 0.4
    bias_beta: float = 0.3
    bias_gamma: float = 0.3
    max_walk_steps: int = 50

    # ---------- 模块5：大模型微调 ----------
    base_model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    use_qlora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    finetune_lr: float = 2e-4
    finetune_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    qlora_epochs: int = 3
    lora_epochs: int = 5
    warmup_ratio: float = 0.03
    max_seq_length: int = 512
    finetune_seed: int = 42

    # ---------- 通用 ----------
    seed: int = 42
    device: str = "cuda"  # 运行时可由 get_device() 覆盖

    @property
    def bias_weights(self) -> Tuple[float, float, float]:
        return (self.bias_alpha, self.bias_beta, self.bias_gamma)

    def resolve_paths(self) -> "Config":
        """将相对路径解析为基于 project_root 的绝对路径。"""
        root = self.project_root.resolve()
        self.data_root = (root / self.data_root).resolve()
        self.output_root = (root / self.output_root).resolve()
        if self.node_text_csv is not None:
            self.node_text_csv = (root / self.node_text_csv).resolve()
        if self.tokenbook_path is not None:
            self.tokenbook_path = (root / self.tokenbook_path).resolve()
        if self.codebook_checkpoint_dir is not None:
            self.codebook_checkpoint_dir = (
                root / self.codebook_checkpoint_dir
            ).resolve()
        return self


_default_config: Optional[Config] = None


def get_config(overrides: Optional[dict] = None) -> Config:
    """获取全局配置单例；``overrides`` 用于一次性覆盖字段。"""
    global _default_config
    if _default_config is None:
        _default_config = Config()
    if overrides:
        for k, v in overrides.items():
            if hasattr(_default_config, k):
                setattr(_default_config, k, v)
            else:
                raise AttributeError(f"Unknown config key: {k}")
    return _default_config


def reset_config() -> None:
    """测试用：重置配置单例。"""
    global _default_config
    _default_config = None
