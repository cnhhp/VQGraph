"""
Pipeline 通用工具（对应 prompt.md 中的 utils.py 约定）。
与根目录 ``utils.py``（VQGraph 原教师训练工具）并存，互不覆盖。
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

from config import Config, get_config

logger = logging.getLogger(__name__)

# 引文网络常用类别名（与 CPF .npz 中 class 顺序一致）
DATASET_CLASS_NAMES: Dict[str, List[str]] = {
    "cora": [
        "Case_Based",
        "Genetic_Algorithms",
        "Neural_Networks",
        "Probabilistic_Methods",
        "Reinforcement_Learning",
        "Rule_Learning",
        "Theory",
    ],
    "citeseer": [
        "Agents",
        "AI",
        "DB",
        "IR",
        "ML",
        "HCI",
        "RT",
    ],
    "pubmed": [
        "Diabetes_Mellitus_Experimental",
        "Diabetes_Mellitus_Type_1",
        "Diabetes_Mellitus_Type_2",
    ],
}

GraphDataBundle = Tuple[
    Any,  # DGL graph g
    torch.Tensor,  # feats [N, F]
    torch.Tensor,  # labels [N]
    torch.Tensor,  # idx_train
    torch.Tensor,  # idx_val
    torch.Tensor,  # idx_test
    Dict[int, str],  # node_id -> raw text
]


def resolve_class_name(dataset_name: str, label_idx: int) -> str:
    """将整数类别索引转为可读类名；未知数据集回退为 Class_{idx}。"""
    names = DATASET_CLASS_NAMES.get(dataset_name.lower())
    idx = int(label_idx)
    if names is not None and 0 <= idx < len(names):
        return names[idx]
    return f"Class_{idx}"


def get_device(cfg: Optional[Config] = None) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def normalize_array(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """将数组线性归一化到 [0, 1]。"""
    arr = np.asarray(arr, dtype=np.float64)
    lo, hi = arr.min(), arr.max()
    if hi - lo < eps:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo + eps)


def load_raw_text(node_text_path: Union[str, Path]) -> Dict[int, str]:
    """从 CSV 加载 node_id -> 文本 映射。列名支持 node_id/id 与 text/content。"""
    path = Path(node_text_path)
    df = pd.read_csv(path)
    id_col = None
    text_col = None
    for c in df.columns:
        cl = c.lower()
        if cl in ("node_id", "id", "nid"):
            id_col = c
        if cl in ("text", "content", "raw_text", "abstract"):
            text_col = c
    if id_col is None:
        id_col = df.columns[0]
    if text_col is None:
        text_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
    return {int(row[id_col]): str(row[text_col]) for _, row in df.iterrows()}


def _bow_pseudo_text(
    feats: torch.Tensor,
    attr_names: Optional[np.ndarray],
    top_k: int = 20,
) -> Dict[int, str]:
    """由 BoW 特征与 attr_names 构造伪文本，供无 CSV 时使用。"""
    n = feats.shape[0]
    text_dict: Dict[int, str] = {}
    dense = feats.cpu().numpy() if feats.is_sparse else feats.detach().cpu().numpy()
    if hasattr(dense, "toarray"):
        dense = dense.toarray()
    for i in range(n):
        row = dense[i].flatten()
        if attr_names is not None and len(attr_names) == row.shape[0]:
            idx = np.argsort(-row)[:top_k]
            words = [str(attr_names[j]) for j in idx if row[j] > 0]
            text_dict[i] = " ".join(words) if words else f"node_{i}"
        else:
            idx = np.argsort(-row)[:top_k]
            text_dict[i] = " ".join(f"w{j}" for j in idx if row[j] > 0) or f"node_{i}"
    return text_dict


def load_graph_data(
    dataset_name: str,
    root: Union[str, Path] = "./data",
    node_text_path: Optional[Union[str, Path]] = None,
    seed: int = 0,
    labelrate_train: int = 20,
    labelrate_val: int = 30,
    split_idx: int = 0,
) -> GraphDataBundle:
    """
    通过现有 DGL dataloader 加载图，并解析节点文本。

    Returns
    -------
    g, feats, labels, idx_train, idx_val, idx_test, text_dict
    """
    from dataloader import load_data

    g, labels, idx_train, idx_val, idx_test = load_data(
        dataset_name,
        str(root),
        split_idx=split_idx,
        seed=seed,
        labelrate_train=labelrate_train,
        labelrate_val=labelrate_val,
    )
    feats = g.ndata["feat"]

    text_dict: Dict[int, str] = {}
    if node_text_path is not None:
        text_dict = load_raw_text(node_text_path)
        logger.info("Loaded node text from %s (%d entries)", node_text_path, len(text_dict))
    else:
        meta = g.ndata.get("text") or g.ndata.get("raw_text")
        if meta is not None:
            for i in range(g.num_nodes()):
                t = meta[i]
                if isinstance(t, bytes):
                    t = t.decode("utf-8", errors="ignore")
                text_dict[i] = str(t)
        else:
            attr_names = getattr(g, "attr_names", None)
            if attr_names is None and "attr_names" in g.ndata:
                attr_names = g.ndata["attr_names"]
            logger.warning(
                "No node_text_csv; using BoW pseudo-text fallback for semantic embeddings."
            )
            text_dict = _bow_pseudo_text(feats, attr_names)

    n = g.num_nodes()
    for i in range(n):
        if i not in text_dict:
            text_dict[i] = f"node_{i}"

    return g, feats, labels, idx_train, idx_val, idx_test, text_dict


def extract_sentence_bert_embeddings(
    text_dict: Dict[int, str],
    model_name: str = "all-MiniLM-L6-v2",
    device: Optional[torch.device] = None,
    batch_size: int = 64,
) -> torch.Tensor:
    """
    为每个节点提取 Sentence-BERT 嵌入，行顺序与 node_id 0..N-1 对齐。

    Returns
    -------
    Tensor [N, D]
    """
    from sentence_transformers import SentenceTransformer

    if device is None:
        device = get_device()
    n = max(text_dict.keys()) + 1 if text_dict else 0
    texts = [text_dict.get(i, f"node_{i}") for i in range(n)]
    model = SentenceTransformer(model_name, device=str(device))
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 500,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )
    if not isinstance(emb, torch.Tensor):
        emb = torch.tensor(emb, dtype=torch.float32)
    return emb.float().to(device)


def compute_pagerank(
    g: Any,
    damping: float = 0.85,
    max_iter: int = 100,
) -> np.ndarray:
    """基于 DGL 图计算 PageRank，返回 shape [N]。"""
    import dgl

    pr = dgl.pagerank(g, alpha=damping, max_iter=max_iter)
    return pr.cpu().numpy().astype(np.float64)


def edge_index_from_dgl(g: Any) -> torch.Tensor:
    """从 DGL 图导出 edge_index [2, E]。"""
    src, dst = g.edges()
    return torch.stack([src, dst], dim=0).long()


def tokenize_for_tfidf(text: str) -> List[str]:
    """简单词级 tokenizer，供 TF-IDF 统计使用。"""
    return re.findall(r"[a-zA-Z]+", text.lower())


def build_tfidf_vocab(
    texts: List[str],
    max_vocab: int = 13648,
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """从文本列表构建词表 word -> id。"""
    from collections import Counter

    cnt: Counter = Counter()
    for t in texts:
        cnt.update(tokenize_for_tfidf(t))
    words = [w for w, _ in cnt.most_common(max_vocab)]
    token_to_id = {w: i for i, w in enumerate(words)}
    id_to_token = {i: w for w, i in token_to_id.items()}
    return token_to_id, id_to_token


def texts_to_token_ids(
    text_dict: Dict[int, str],
    token_to_id: Dict[str, int],
    num_nodes: int,
) -> List[List[int]]:
    """将节点文本转为 token id 列表。"""
    return texts_to_tokenbook_ids(text_dict, token_to_id, num_nodes)


def texts_to_tokenbook_ids(
    text_dict: Dict[int, str],
    token_to_id: Dict[str, int],
    num_nodes: int,
) -> List[List[int]]:
    """词级分词后，仅保留 tokenbook 词表中存在的 token id。"""
    out: List[List[int]] = []
    for i in range(num_nodes):
        toks = tokenize_for_tfidf(text_dict.get(i, ""))
        ids = [token_to_id[t] for t in toks if t in token_to_id]
        out.append(ids)
    return out


def set_seed(seed: int = 42) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(
    name: str = __name__,
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
) -> logging.Logger:
    log = logging.getLogger(name)
    if log.handlers:
        return log
    log.setLevel(level)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    log.addHandler(ch)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        log.addHandler(fh)
    return log
