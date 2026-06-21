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
    text_dataset_subdir: str = "dataset"  # 相对 data_root，文本 DGL 数据目录名
    data_source: Optional[str] = None  # None=auto；text / cpf 强制来源
    sentence_bert_model: str = "all-MiniLM-L6-v2"

    # ---------- 模块1：结构码本 ----------
    codebook_size: int = 2048  # M
    codebook_dim: int = 256  # d
    gnn_type: str = "GCN"  # GCN | SAGE
    lambda_semantic: float = 0.1
    warmup_epochs: int = 20
    tfidf_stats_min_epoch: Optional[int] = None  # None → warmup_epochs+1；此前 epoch 不参与 best/TF-IDF
    ema_beta: float = 0.99  # 语义中心 μ_k 的 EMA 衰减
    codebook_train_epochs: int = 100
    codebook_lr: float = 1e-3
    codebook_batch_size: int = -1  # -1 表示全图
    use_contrastive_loss: bool = False
    contrastive_weight: float = 0.1
    # Token 预测辅助任务（模块1 训练 + 模块3 推理融合）
    enable_token_predictor: bool = True
    lambda_token: float = 0.05  # α：KL 损失权重
    token_pred_temperature: float = 1.0  # τ：预测分布温度
    token_target_temperature: float = 0.15  # τ'：目标分布温度
    token_kl_top_k: int = 0  # Top-K KL；0=全词表
    token_predictor_type: str = "linear"  # linear | factorized
    p_code_normalize: str = "max"  # none | max | minmax
    lambda_pred: float = 0.05  # 推理时 P_code 融合强度（E4 sweep 最佳）
    predictor_only_epochs: int = 15  # 冻结 VQ 时仅训 predictor 的 epoch 数

    # ---------- 模块2：子图提取 ----------
    subgraph_k_hop: int = 2  # 阶段 B：1→2 hop，丰富结构上下文
    max_subgraph_nodes: Optional[int] = None  # None 表示不截断

    # ---------- 模块3：节点离散化 ----------
    text_vocab_size: int = 13648  # V（filtered_tokenbook.npy）
    top_k_text_tokens: int = 8  # 阶段 B：5→8，增强类判别词覆盖
    lambda_tfidf: float = 0.5
    mmr_lambda: float = 0.5  # MMR 相关性权重，越大越偏向高分 token
    mmr_candidate_pool: int = 96  # 阶段 B：配合更大的 top_k
    filter_stopwords_at_selection: bool = True  # 选词时屏蔽停用词（词表不变）
    filter_noise_subwords_at_selection: bool = True  # 选词时屏蔽 PDF/子词噪声
    struct_token_prefix: str = "<S_"
    # 结构 token 展示：id | pcode_supplement | pcode_replace | struct_summary（首行摘要 + 行间 <S_k>）
    struct_token_mode: str = "id"
    p_code_struct_top_k: int = 3  # P_code top-k 可读词（pcode_* 模式）

    # ---------- 模块3b：可学习 TokenSelector ----------
    token_selector_hidden_dim: int = 128
    token_selector_lr: float = 1e-3
    token_selector_epochs: int = 50
    token_selector_batch_size: int = 32
    gumbel_tau_init: float = 1.0
    gumbel_tau_min: float = 0.5
    gumbel_tau_anneal_epochs: int = 30
    token_selector_candidate_pool: int = 128  # Gumbel / 推理仅在 s0 Top-K 候选池内
    token_selector_kl_weight: float = 0.3  # KL(w || softmax(s0))，防止偏离初始得分
    token_selector_entropy_weight: float = 0.01  # 最大化选择熵，缓解模式坍缩
    token_selector_vtext_dropout: float = 0.3  # 训练时对 v_text dropout，迫使分类头依赖文本
    token_selector_training_mode: str = "distill"  # distill | cls
    token_selector_distill_weight: float = 1.0  # KL(P_student || P_target_mmr)
    token_selector_cls_weight: float = 0.05  # 蒸馏模式下弱分类损失
    token_selector_student_temperature: float = 1.0  # softmax(s_selector) 温度
    token_selector_checkpoint: Optional[Path] = None

    # ---------- 模块4：序列化 ----------
    bias_alpha: float = 0.4
    bias_beta: float = 0.3
    bias_gamma: float = 0.3
    max_walk_steps: int = 50

    # ---------- 模块5：大模型微调 ----------
    base_model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    use_qlora: bool = True
    lora_r: int = 8  # 阶段 A：16→8，减轻过拟合
    lora_alpha: int = 16
    lora_dropout: float = 0.1  # 阶段 A：0.05→0.1
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    finetune_lr: float = 1e-4  # 阶段 A：2e-4→1e-4
    finetune_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    qlora_epochs: int = 2  # 阶段 A：默认 2 epoch（原 5 epoch 易过拟合）
    lora_epochs: int = 5
    warmup_ratio: float = 0.06  # 阶段 A：略增 warmup
    max_seq_length: int = 768  # 阶段 B：k=2 + top_k=8 序列更长
    finetune_seed: int = 42
    qlora_val_acc_threshold: float = 0.0  # QLoRA 最低 val 准确率；0 表示不阻断
    finetune_eval_batch_size: int = 4
    use_bf16: bool = True  # CUDA 可用时启用；CPU 自动关闭
    max_new_tokens: int = 32  # 推理生成长度（类名较短）

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
