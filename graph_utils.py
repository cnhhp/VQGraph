"""
Pipeline 通用工具（对应 prompt.md 中的 utils.py 约定）。
与根目录 ``utils.py``（VQGraph 原教师训练工具）并存，互不覆盖。
"""

from __future__ import annotations

import logging
import pickle
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


def format_category_display_name(raw: str) -> str:
    """将 'neural networks' 规范为 JSON 友好类名 'Neural_Networks'。"""
    parts = re.split(r"[\s_]+", str(raw).strip())
    return "_".join(p.capitalize() for p in parts if p)


def resolve_node_class_name(
    dataset_name: str,
    node_id: int,
    labels: torch.Tensor,
    graph: Any,
) -> str:
    """
    解析节点类别名：优先 g.ndata['category_name']（文本 DGL 数据集），
    否则回退 CPF 的 label 索引映射。
    """
    cat_names = getattr(graph, "category_names", None)
    if cat_names is not None:
        raw = cat_names[node_id]
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="ignore")
        return format_category_display_name(str(raw))
    if "category_name" in getattr(graph, "ndata", {}):
        raw = graph.ndata["category_name"][node_id]
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="ignore")
        return format_category_display_name(str(raw))
    label_idx = int(labels[node_id].item())
    return resolve_class_name(dataset_name, label_idx)


def text_dgl_dataset_dir(
    root: Union[str, Path],
    dataset_name: str,
    text_dataset_subdir: str = "dataset",
) -> Optional[Path]:
    """若存在 {root}/{subdir}/{name}/{name}_graph.pth 则返回该目录。"""
    d = Path(root) / text_dataset_subdir / dataset_name
    if (d / f"{dataset_name}_graph.pth").exists():
        return d
    return None


def _load_text_dgl_metadata(dataset_dir: Path, dataset_name: str) -> Dict[str, Any]:
    """加载 metadata，兼容 .pth / .pt。"""
    for suffix in (".pth", ".pt"):
        path = dataset_dir / f"{dataset_name}_metadata{suffix}"
        if path.exists():
            return torch.load(path, map_location="cpu")
    return {}


def load_text_dgl_dataset(
    dataset_name: str,
    dataset_dir: Path,
) -> GraphDataBundle:
    """
    加载 data/dataset/{name}/ 下的 DGL 图 + 文本 + 划分。

    期望文件：{name}_graph.pth、{name}_text.pkl（可选）、{name}_metadata.pth|.pt
    """
    dataset_dir = Path(dataset_dir)
    graph_path = dataset_dir / f"{dataset_name}_graph.pth"
    g = torch.load(graph_path, map_location="cpu")
    n = g.num_nodes()

    text_dict: Dict[int, str] = {}
    text_pkl = dataset_dir / f"{dataset_name}_text.pkl"
    if text_pkl.exists():
        with open(text_pkl, "rb") as f:
            texts = pickle.load(f)
        if isinstance(texts, dict):
            text_dict = {int(k): str(v) for k, v in texts.items()}
        elif isinstance(texts, (list, tuple)):
            text_dict = {i: str(texts[i]) for i in range(len(texts))}
        else:
            raise ValueError(f"Unsupported text format in {text_pkl}: {type(texts)}")
        logger.info("Loaded node text from %s (%d entries)", text_pkl, len(text_dict))
    else:
        logger.warning("No %s_text.pkl; node text will be empty unless CSV override.", dataset_name)

    meta = _load_text_dgl_metadata(dataset_dir, dataset_name)
    categories = meta.get("categories")
    if categories is not None:
        # 字符串类名无法存入 ndata tensor，挂到图对象上
        g.category_names = np.asarray(categories)

    feats = g.ndata["feat"]
    labels = g.ndata["label"].long()

    def _mask_to_idx(mask_key: str) -> torch.Tensor:
        if mask_key not in g.ndata:
            raise KeyError(f"Graph missing ndata['{mask_key}'] for text DGL split")
        mask = g.ndata[mask_key].bool()
        return torch.where(mask)[0].long()

    idx_train = _mask_to_idx("train_mask")
    idx_val = _mask_to_idx("val_mask")
    idx_test = _mask_to_idx("test_mask")

    for i in range(n):
        if i not in text_dict:
            text_dict[i] = f"node_{i}"

    logger.info(
        "Text DGL dataset %s: nodes=%d edges=%d feat_dim=%d train=%d val=%d test=%d",
        dataset_name,
        n,
        g.num_edges(),
        feats.shape[1],
        len(idx_train),
        len(idx_val),
        len(idx_test),
    )
    return g, feats, labels, idx_train, idx_val, idx_test, text_dict


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
    data_source: Optional[str] = None,
    text_dataset_subdir: str = "dataset",
) -> GraphDataBundle:
    """
    加载图数据与节点文本。

    data_source:
    - None / "auto": 若存在 data/dataset/{name}/{name}_graph.pth 则用文本 DGL 版，否则 CPF .npz
    - "text": 强制文本 DGL
    - "cpf": 强制 CPF dataloader

    Returns
    -------
    g, feats, labels, idx_train, idx_val, idx_test, text_dict
    """
    root = Path(root)
    source = (data_source or "auto").lower()
    text_dir = text_dgl_dataset_dir(root, dataset_name, text_dataset_subdir)

    use_text = False
    if source in ("text", "text_dgl", "dgl"):
        if text_dir is None:
            raise FileNotFoundError(
                f"data_source=text but no text DGL dataset at "
                f"{root / text_dataset_subdir / dataset_name}"
            )
        use_text = True
    elif source in ("cpf", "npz"):
        use_text = False
    elif source in ("auto", "none", ""):
        use_text = text_dir is not None
    else:
        raise ValueError(f"Unknown data_source: {data_source}")

    if use_text:
        assert text_dir is not None
        logger.info("Using text DGL dataset from %s", text_dir)
        g, feats, labels, idx_train, idx_val, idx_test, text_dict = load_text_dgl_dataset(
            dataset_name, text_dir
        )
    else:
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
        text_dict = {}

        if node_text_path is None:
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

    if node_text_path is not None:
        overrides = load_raw_text(node_text_path)
        text_dict.update(overrides)
        logger.info(
            "Applied node text overrides from %s (%d entries)",
            node_text_path,
            len(overrides),
        )

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


# 选词阶段过滤用（词表本身不改动）
ENGLISH_STOPWORDS: frozenset = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "if", "then", "else", "when",
        "at", "by", "for", "with", "about", "against", "between", "into",
        "through", "during", "before", "after", "above", "below", "to", "from",
        "up", "down", "in", "out", "on", "off", "over", "under", "again",
        "further", "once", "here", "there", "all", "any", "both", "each",
        "few", "more", "most", "other", "some", "such", "no", "nor", "not",
        "only", "own", "same", "so", "than", "too", "very", "can", "will",
        "just", "should", "now", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "doing", "would",
        "could", "ought", "i", "you", "he", "she", "it", "we", "they", "them",
        "their", "this", "that", "these", "those", "am", "as", "of", "which",
        "who", "whom", "what", "whose", "where", "why", "how", "because",
        "while", "although", "though", "until", "unless", "since", "via",
        # cora_text.pkl 模板字段
        "title", "abstract",
    }
)


def is_stopword(token: str) -> bool:
    """判断 token 是否在选词停用词表中（小写、纯字母）。"""
    w = str(token).lower().strip()
    if not w or not w.isalpha():
        return True
    if len(w) <= 1:
        return True
    return w in ENGLISH_STOPWORDS


_STOPWORD_SCORE_PENALTY = -1e12  # 勿用 -inf：argpartition(-scores) 会将其误选为最大


def mask_stopwords_in_scores(
    scores: np.ndarray,
    id_to_token: Dict[int, str],
) -> Tuple[np.ndarray, int]:
    """
    将停用词对应位置的 score 置为极低分，供 MMR/Top-K 选词使用（词表不变）。

    Returns
    -------
    masked_scores, num_masked
    """
    masked = np.asarray(scores, dtype=np.float64).copy()
    n_masked = 0
    for tid, tok in id_to_token.items():
        idx = int(tid)
        if idx < 0 or idx >= masked.shape[0]:
            continue
        if is_stopword(tok):
            masked[idx] = _STOPWORD_SCORE_PENALTY
            n_masked += 1
    return masked, n_masked


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
