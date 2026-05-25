"""
模块3：节点表示离散化 — Top-K 文本 token + 1 个结构 token。
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config, get_config
from models.codebook_trainer import (
    CodebookTrainer,
    TFIDFComputer,
    _wrap_semantic_vq,
)
from models.structural_codebook import CodebookArtifacts, TFIDFStatistics
from models.subgraph_extraction import LocalSubgraph, SubgraphExtractor
from models._root_bridge import load_root_models_module

Model = load_root_models_module().Model

from text_tokenizers.text_tokenbook import TextTokenbook  # noqa: E402

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
        model_conf: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.cfg = cfg or get_config()
        self.artifacts = artifacts
        self.tokenbook = tokenbook
        self.tfidf = tfidf
        self.model_conf = model_conf
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.encoder: Optional[nn.Module] = None
        self._struct_dist_cache: Optional[torch.Tensor] = None

    def _resolve_train_conf(self) -> Dict[str, Any]:
        if self.model_conf is not None:
            conf = dict(self.model_conf)
        else:
            if self.artifacts.save_dir is None:
                raise ValueError(
                    "model_conf is required when artifacts.save_dir is None"
                )
            conf_path = Path(self.artifacts.save_dir) / "train_conf.json"
            if not conf_path.exists():
                raise FileNotFoundError(
                    f"{conf_path} not found; re-run train_codebook.py or pass model_conf"
                )
            with open(conf_path, encoding="utf-8") as f:
                conf = json.load(f)
        device = conf.get("device", str(self.device))
        if isinstance(device, str):
            conf["device"] = torch.device(device)
        return conf

    def load_encoder(self) -> nn.Module:
        """加载模块1 训练好的 GNN 编码器（含 SemanticVectorQuantize）。"""
        if self.encoder is not None:
            return self.encoder

        conf = self._resolve_train_conf()
        device = conf["device"]
        text_dim = int(self.artifacts.semantic_centers.shape[1])

        model = Model(conf)
        _wrap_semantic_vq(model, text_dim, self.cfg)
        model.load_state_dict(self.artifacts.encoder_state_dict)
        model.to(device)
        model.eval()
        self.encoder = model
        self.device = device
        return model

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
        codebook = self.artifacts.codebook_embeddings.to(structural_emb.device)
        h = F.normalize(structural_emb.float(), dim=-1)
        cb = F.normalize(codebook.float(), dim=-1)
        sim = h @ cb.T
        return sim.argmax(dim=-1)

    def _infer_distances(
        self,
        graph: Any,
        node_features: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        全图推理，返回 VQ 余弦分配分数 dist [N, M]。

        GCN/SAGE 均需全图上下文，不可仅对子图前向。
        """
        if self._struct_dist_cache is not None:
            return self._struct_dist_cache

        import dgl

        model = self.load_encoder()
        device = self.device
        feats = node_features.to(device)
        text_emb = text_embeddings.to(device)

        model.eval()
        with torch.no_grad():
            if "SAGE" in model.model_name:
                graph.create_formats_()
                sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
                conf = self._resolve_train_conf()
                loader = dgl.dataloading.DataLoader(
                    graph,
                    torch.arange(graph.num_nodes()),
                    sampler,
                    batch_size=conf.get("batch_size", 512),
                    shuffle=False,
                    num_workers=0,
                )
                _, _, _, dist, _ = model.inference(
                    loader, feats, text_emb=text_emb
                )
            else:
                g = graph.to(device)
                _, _, _, dist, _ = model.inference(g, feats, text_emb=text_emb)

        if dist.dim() == 3:
            dist = dist.squeeze(0)
        self._struct_dist_cache = dist
        return dist

    def _compute_text_similarity(
        self,
        node_text_emb: torch.Tensor,
    ) -> np.ndarray:
        """节点嵌入与 tokenbook 余弦相似度，归一化到 [0,1]，shape [V]。"""
        book = self.tokenbook.get_embedding_matrix().to(self.device)
        node = node_text_emb.float().to(self.device)
        if node.dim() == 1:
            node = node.unsqueeze(0)
        book_norm = F.normalize(book, dim=1)
        node_norm = F.normalize(node, dim=1)
        cos = (node_norm @ book_norm.T).squeeze(0).detach().cpu().numpy()
        return (cos + 1.0) / 2.0

    def _align_tfidf_row(self, struct_code_idx: int) -> np.ndarray:
        """
        将 TF-IDF 先验行对齐到 tokenbook 词表长度 [V_tokenbook]。

        越界结构码 / 无 TF-IDF / 空行时返回全零（融合时退化为纯 text_sim）。
        """
        vocab_size = len(self.tokenbook.token_to_id)
        aligned = np.zeros(vocab_size, dtype=np.float64)

        if self.tfidf is None:
            return aligned

        num_codes = self.tfidf.num_codes
        if struct_code_idx < 0 or struct_code_idx >= num_codes:
            return aligned

        row = self.tfidf.get_prior_weights(struct_code_idx)
        if row.max() < 1e-8:
            return aligned

        tfidf_vocab = self.tfidf.token_to_id
        tokenbook_vocab = self.tokenbook.token_to_id

        if tfidf_vocab is None:
            n = min(len(row), vocab_size)
            aligned[:n] = row[:n]
            return aligned

        if tfidf_vocab == tokenbook_vocab and len(row) == vocab_size:
            return row.astype(np.float64)

        tfidf_id_to_token = {i: t for t, i in tfidf_vocab.items()}
        for tfidf_id, weight in enumerate(row):
            if weight <= 0:
                continue
            token = tfidf_id_to_token.get(tfidf_id)
            if token is None:
                continue
            book_id = tokenbook_vocab.get(token)
            if book_id is not None:
                aligned[book_id] = weight

        return aligned

    def _fuse_scores(
        self,
        text_sim: np.ndarray,
        struct_code_idx: int,
    ) -> np.ndarray:
        """
        score[t] = text_sim[t] * (1 + λ * TF-IDF_norm[c][t])；
        异常码索引或空 TF-IDF 行时退化为 text_sim。
        """
        prior = self._align_tfidf_row(struct_code_idx)
        if prior.max() < 1e-8:
            return text_sim
        lam = self.cfg.lambda_tfidf
        return text_sim * (1.0 + lam * prior)

    def select_text_tokens(
        self,
        node_text_emb: torch.Tensor,
        struct_code_idx: int,
        top_k: Optional[int] = None,
    ) -> Tuple[List[str], List[int]]:
        """
        结构引导的 Top-K 文本 token 选取。

        Returns
        -------
        text_tokens, text_token_ids
        """
        k = top_k or self.cfg.top_k_text_tokens
        vocab_size = len(self.tokenbook.token_to_id)
        if vocab_size == 0:
            return [], []

        text_sim = self._compute_text_similarity(node_text_emb)
        scores = self._fuse_scores(text_sim, struct_code_idx)
        k = min(k, vocab_size)

        top_ids = np.argpartition(-scores, k - 1)[:k]
        top_ids = top_ids[np.argsort(-scores[top_ids])]

        id_to_token = self.tokenbook.id_to_token
        tokens = [id_to_token[int(i)] for i in top_ids if int(i) in id_to_token]
        ids = [int(i) for i in top_ids]
        return tokens, ids

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
        dist = self._infer_distances(graph, node_features, text_embeddings)
        struct_code = int(dist[node_id].argmax().item())
        text_tokens, text_ids = self.select_text_tokens(
            text_embeddings[node_id], struct_code
        )
        return NodeTokenRepresentation(
            node_id=node_id,
            struct_token=self.format_struct_token(struct_code),
            text_tokens=text_tokens,
            struct_code_idx=struct_code,
            text_token_ids=text_ids,
        )

    def tokenize_subgraph(
        self,
        subgraph: LocalSubgraph,
        graph: Any,
        node_features: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> SubgraphTokenizedView:
        """
        子图内所有节点批量编码并离散化。

        典型流程：全图 GNN 前向 -> 结构码 -> 文本 Top-K。
        """
        dist = self._infer_distances(graph, node_features, text_embeddings)
        struct_codes = dist.argmax(dim=1)

        nodes: Dict[int, NodeTokenRepresentation] = {}
        for node_id in subgraph.node_ids:
            struct_code = int(struct_codes[node_id].item())
            text_tokens, text_ids = self.select_text_tokens(
                text_embeddings[node_id], struct_code
            )
            nodes[node_id] = NodeTokenRepresentation(
                node_id=node_id,
                struct_token=self.format_struct_token(struct_code),
                text_tokens=text_tokens,
                struct_code_idx=struct_code,
                text_token_ids=text_ids,
            )

        return SubgraphTokenizedView(
            center_node=subgraph.center_node,
            nodes=nodes,
        )


class _FixtureTokenbook(TextTokenbook):
    """冒烟测试用最小 tokenbook（不写入 text_tokenizers 模块）。"""

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        max_vocab: int = 512,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[torch.device] = None,
    ) -> "_FixtureTokenbook":
        from graph_utils import build_tfidf_vocab, extract_sentence_bert_embeddings

        token_to_id, id_to_token = build_tfidf_vocab(texts, max_vocab=max_vocab)
        tb = cls()
        tb.token_to_id = token_to_id
        tb.id_to_token = id_to_token

        words = [id_to_token[i] for i in range(len(id_to_token))]
        word_dict = {i: w for i, w in enumerate(words)}
        emb = extract_sentence_bert_embeddings(
            word_dict,
            model_name=model_name,
            device=device,
        )
        tb.embeddings = emb.float()
        return tb


def _run_smoke_test(args: argparse.Namespace) -> None:
    from graph_utils import (
        extract_sentence_bert_embeddings,
        load_graph_data,
        setup_logging,
    )

    setup_logging(__name__)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.device >= 0 else "cpu"
    )

    codebook_dir = Path(args.codebook_dir)
    artifacts = CodebookTrainer.load_artifacts(codebook_dir, device)

    tfidf = None
    tfidf_path = codebook_dir / "tfidf_stats.npz"
    if not args.no_tfidf and tfidf_path.exists():
        tfidf = TFIDFComputer.load(tfidf_path)

    g, feats, _, _, _, _, text_dict = load_graph_data(
        args.dataset,
        root=args.data_root,
        seed=args.seed,
    )
    texts = [text_dict[i] for i in range(g.num_nodes())]
    if args.use_fixture_tokenbook:
        tokenbook = _FixtureTokenbook.from_texts(
            texts,
            max_vocab=args.vocab_size,
            model_name=args.sentence_bert,
            device=device,
        )
        logger.info("Using fixture tokenbook (V=%d)", len(tokenbook))
    else:
        tokenbook = TextTokenbook.load(
            args.tokenbook_dir,
            model_name=args.sentence_bert,
            device=device,
        )
        logger.info("Using tokenbook from %s (V=%d)", args.tokenbook_dir, len(tokenbook))

    text_emb = extract_sentence_bert_embeddings(
        text_dict,
        model_name=args.sentence_bert,
        device=device,
    )

    extractor = SubgraphExtractor()
    subgraph = extractor.extract_from_dgl_graph(g, args.node, k=args.k)

    tokenizer = NodeRepresentationTokenizer(
        artifacts=artifacts,
        tokenbook=tokenbook,
        tfidf=tfidf,
    )
    view = tokenizer.tokenize_subgraph(subgraph, g, feats, text_emb)

    assert len(view.nodes) == subgraph.num_nodes, "node count mismatch"
    struct_pat = re.compile(
        re.escape(tokenizer.cfg.struct_token_prefix) + r"\d+>"
    )
    for node_id, repr_ in view.nodes.items():
        assert struct_pat.match(repr_.struct_token), repr_.struct_token
        assert len(repr_.text_tokens) <= tokenizer.cfg.top_k_text_tokens

    center = view.nodes[view.center_node]
    logger.info(
        "Smoke test passed: center=%d struct=%s text=%s",
        view.center_node,
        center.struct_token,
        center.text_tokens,
    )
    logger.info("Subgraph nodes tokenized: %d", len(view.nodes))

    if args.test_fuse_fallback:
        sim = np.random.rand(len(tokenbook)).astype(np.float64)
        fused = tokenizer._fuse_scores(sim, struct_code_idx=999999)
        np.testing.assert_array_equal(fused, sim)
        logger.info("TF-IDF fallback on invalid code: OK")


def _parse_main_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="模块3 节点离散化冒烟测试")
    p.add_argument("--codebook_dir", type=str, required=True)
    p.add_argument("--dataset", type=str, default="cora")
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--node", type=int, default=0)
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=int, default=0, help="-1 for CPU")
    p.add_argument("--no_tfidf", action="store_true")
    p.add_argument("--tokenbook_dir", type=str, default="./codebook")
    p.add_argument("--use_fixture_tokenbook", action="store_true")
    p.add_argument("--vocab_size", type=int, default=512)
    p.add_argument("--sentence_bert", type=str, default="all-MiniLM-L6-v2")
    p.add_argument("--test_fuse_fallback", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    _run_smoke_test(_parse_main_args())
