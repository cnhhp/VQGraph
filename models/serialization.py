"""
模块4：偏置欧拉回路游走 + 回溯压缩 + 树状文本序列化。
"""

from __future__ import annotations

import argparse
import logging
import re
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from config import Config, get_config
from graph_utils import normalize_array
from models.node_representation import SubgraphTokenizedView
from models.subgraph_extraction import LocalSubgraph

logger = logging.getLogger(__name__)

# 偏置函数类型: B(u, v) -> float，u 为当前节点，v 为候选邻居
BiasFunction = Callable[[int, int], float]


@dataclass
class TreeNode:
    """多叉树节点，子节点按偏置分数降序排列。"""

    node_id: int
    children: List["TreeNode"] = field(default_factory=list)
    bias_score: float = 0.0  # 相对父节点的边偏置分（用于排序）

    def add_child(self, child: "TreeNode") -> None:
        self.children.append(child)

    def sort_children_by_bias(self, descending: bool = True) -> None:
        self.children.sort(key=lambda c: c.bias_score, reverse=descending)
        for ch in self.children:
            ch.sort_children_by_bias(descending)


@dataclass
class SerializationResult:
    """序列化输出。"""

    raw_sequence: List[int]  # S_raw
    tree_root: TreeNode
    text: str  # 最终 LLM 可读多行字符串
    node_levels: Dict[int, int]  # node_id -> 层级（根为 0）
    all_edges_visited: bool = True
    stopped_by_max_steps: bool = False


class BiasedEulerSerializer:
    """
    偏置欧拉游走序列化器。

    步骤:
    1. 无向边拆为有向双向边
    2. biased_euler_tour 生成 S_raw（含死胡同 BFS 回退）
    3. compress_to_tree 消除回溯
    4. serialize_tree 输出层级文本
    """

    def __init__(
        self,
        cfg: Optional[Config] = None,
        bias_func: Optional[BiasFunction] = None,
    ) -> None:
        self.cfg = cfg or get_config()
        self.bias_func = bias_func
        self.alpha, self.beta, self.gamma = self.cfg.bias_weights
        self.max_steps: int = self.cfg.max_walk_steps

    @staticmethod
    def _undirected_edge_pairs(subgraph: LocalSubgraph) -> Set[Tuple[int, int]]:
        """从子图 edge_index 提取无向边对（全局 node_id）。"""
        node_set = set(subgraph.node_ids)
        pairs: Set[Tuple[int, int]] = set()
        if subgraph.edge_index.numel() == 0:
            return pairs
        src = subgraph.edge_index[0].tolist()
        dst = subgraph.edge_index[1].tolist()
        for u, v in zip(src, dst):
            if u == v or u not in node_set or v not in node_set:
                continue
            pairs.add((min(u, v), max(u, v)))
        return pairs

    @staticmethod
    def _text_sim_pair(
        u: int,
        v: int,
        text_embeddings: torch.Tensor,
    ) -> float:
        """节点 u、v 文本嵌入余弦相似度，归一化到 [0, 1]。"""
        eu = text_embeddings[u].float()
        ev = text_embeddings[v].float()
        cos = F.cosine_similarity(
            eu.unsqueeze(0),
            ev.unsqueeze(0),
        ).item()
        return float((cos + 1.0) / 2.0)

    def compute_subgraph_stats(
        self,
        subgraph: LocalSubgraph,
        damping: float = 0.85,
        max_iter: int = 100,
    ) -> Tuple[Dict[int, float], Dict[int, float]]:
        """
        在子图诱导边上计算度与 PageRank，并归一化到 [0, 1]。

        Returns
        -------
        norm_degree, norm_pagerank : node_id -> float
        """
        node_ids = subgraph.node_ids
        n = len(node_ids)
        if n == 0:
            return {}, {}

        undirected = self._undirected_edge_pairs(subgraph)
        idx = {nid: i for i, nid in enumerate(node_ids)}

        deg = np.zeros(n, dtype=np.float64)
        adj: List[List[int]] = [[] for _ in range(n)]
        for a, b in undirected:
            ia, ib = idx[a], idx[b]
            deg[ia] += 1
            deg[ib] += 1
            adj[ia].append(ib)
            adj[ib].append(ia)

        norm_degree_arr = normalize_array(deg)
        norm_degree = {
            node_ids[i]: float(norm_degree_arr[i]) for i in range(n)
        }

        if n == 1:
            norm_pagerank = {node_ids[0]: 1.0}
            return norm_degree, norm_pagerank

        pr = np.full(n, 1.0 / n, dtype=np.float64)
        out_deg = np.array([len(adj[i]) for i in range(n)], dtype=np.float64)

        for _ in range(max_iter):
            new_pr = np.full(n, (1.0 - damping) / n, dtype=np.float64)
            for i in range(n):
                if out_deg[i] == 0:
                    new_pr += damping * pr[i] / n
                else:
                    share = damping * pr[i] / out_deg[i]
                    for j in adj[i]:
                        new_pr[j] += share
            pr = new_pr

        norm_pr_arr = normalize_array(pr)
        norm_pagerank = {
            node_ids[i]: float(norm_pr_arr[i]) for i in range(n)
        }
        return norm_degree, norm_pagerank

    def build_default_bias_function(
        self,
        subgraph: LocalSubgraph,
        norm_degree: Dict[int, float],
        norm_pagerank: Dict[int, float],
        text_embeddings: torch.Tensor,
    ) -> BiasFunction:
        """
        B(u, v) = α·norm_degree(v) + β·norm_pagerank(v) + γ·text_sim(u, v)
        其中 norm_* 已归一化到 [0,1]。
        """

        def bias(u: int, v: int) -> float:
            return (
                self.alpha * norm_degree.get(v, 0.0)
                + self.beta * norm_pagerank.get(v, 0.0)
                + self.gamma
                * self._text_sim_pair(u, v, text_embeddings)
            )

        return bias

    def preprocess_directed_edges(
        self,
        subgraph: LocalSubgraph,
    ) -> Tuple[Dict[int, List[int]], List[Tuple[int, int]]]:
        """
        将子图每条无向边 (u,v) 拆为 (u→v) 与 (v→u)。

        Returns
        -------
        adj_out : 出边邻接表（全局 node_id）
        directed_edges : 所有有向边列表
        """
        node_ids = subgraph.node_ids
        adj_out: Dict[int, List[int]] = {nid: [] for nid in node_ids}
        directed_edges: List[Tuple[int, int]] = []

        for a, b in sorted(self._undirected_edge_pairs(subgraph)):
            directed_edges.append((a, b))
            directed_edges.append((b, a))
            adj_out[a].append(b)
            adj_out[b].append(a)

        for u in adj_out:
            adj_out[u] = sorted(set(adj_out[u]))

        return adj_out, directed_edges

    @staticmethod
    def _has_unvisited_out_edge(
        u: int,
        adj_out: Dict[int, List[int]],
        visited_edges: Set[Tuple[int, int]],
    ) -> bool:
        return any((u, v) not in visited_edges for v in adj_out.get(u, []))

    def _select_next_neighbor(
        self,
        u: int,
        candidates: List[int],
        bias_func: BiasFunction,
    ) -> int:
        """候选邻居中偏置分最高；同分取 node_id 最小。"""
        best_v = min(candidates)
        best_score = bias_func(u, best_v)
        for v in sorted(candidates):
            score = bias_func(u, v)
            if score > best_score:
                best_score = score
                best_v = v
        return best_v

    def _backtrack_via_bfs(
        self,
        u_current: int,
        u_target: int,
        adj_out: Dict[int, List[int]],
        visited_edges: Set[Tuple[int, int]],
    ) -> List[int]:
        """
        死胡同回退：BFS 求 u_current -> u_target 最短路径节点序列。
        路径上的边若未访问则标记为已访问（双向）。
        """
        if u_current == u_target:
            return [u_current]

        parent: Dict[int, Optional[int]] = {u_current: None}
        queue: deque[int] = deque([u_current])

        while queue:
            u = queue.popleft()
            if u == u_target:
                break
            for v in adj_out.get(u, []):
                if v not in parent:
                    parent[v] = u
                    queue.append(v)

        if u_target not in parent:
            logger.warning(
                "BFS backtrack failed %d -> %d; staying at current",
                u_current,
                u_target,
            )
            return [u_current]

        path: List[int] = []
        cur: Optional[int] = u_target
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        path.reverse()

        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            if (a, b) not in visited_edges:
                visited_edges.add((a, b))
            if (b, a) not in visited_edges:
                visited_edges.add((b, a))

        return path

    def biased_euler_tour(
        self,
        subgraph: LocalSubgraph,
        tokenized: SubgraphTokenizedView,
        bias_func: BiasFunction,
        start_node: Optional[int] = None,
        max_steps: Optional[int] = None,
    ) -> Tuple[List[int], bool, bool]:
        """
        偏置欧拉游走，生成原始序列 S_raw。

        关键逻辑:
        - 每步在未访问出边中选 B(u,v) 最大者，平局取 ID 最小
        - 无未访问出边时，从 S_raw 倒序找仍有出边的 u'，BFS 最短路径回退

        Returns
        -------
        S_raw, all_edges_visited, stopped_by_max_steps
        """
        adj_out, directed_edges = self.preprocess_directed_edges(subgraph)
        all_edges = set(directed_edges)
        start = start_node if start_node is not None else tokenized.center_node

        s_raw: List[int] = [start]
        u_current = start
        visited_edges: Set[Tuple[int, int]] = set()
        step_limit = max_steps if max_steps is not None else self.max_steps
        step_count = 0
        stopped_by_max_steps = False

        while step_count < step_limit:
            if len(visited_edges) >= len(all_edges):
                break

            candidates = [
                v
                for v in adj_out.get(u_current, [])
                if (u_current, v) not in visited_edges
            ]

            if candidates:
                # 有未访问出边：选偏置分最高的邻居
                v_star = self._select_next_neighbor(
                    u_current, candidates, bias_func
                )
                visited_edges.add((u_current, v_star))
                s_raw.append(v_star)
                u_current = v_star
            else:
                if len(visited_edges) >= len(all_edges):
                    break

                # 死胡同：从 S_raw 倒序找仍有未访问出边的节点 u'
                u_prime: Optional[int] = None
                for u in reversed(s_raw):
                    if self._has_unvisited_out_edge(u, adj_out, visited_edges):
                        u_prime = u
                        break

                if u_prime is None:
                    break

                path = self._backtrack_via_bfs(
                    u_current, u_prime, adj_out, visited_edges
                )
                for i, node in enumerate(path):
                    if i == 0 and s_raw and node == s_raw[-1]:
                        continue
                    s_raw.append(node)
                u_current = u_prime

            step_count += 1

        if step_count >= step_limit and len(visited_edges) < len(all_edges):
            stopped_by_max_steps = True

        all_edges_visited = len(visited_edges) >= len(all_edges)
        return s_raw, all_edges_visited, stopped_by_max_steps

    def compress_to_tree(
        self,
        raw_sequence: List[int],
        bias_func: BiasFunction,
    ) -> TreeNode:
        """
        回溯压缩：扫描 S_raw，栈维护当前路径；
        新节点入树，已见节点则弹栈至该节点（消除 A-B-A 回溯）。
        """
        if not raw_sequence:
            raise ValueError("raw_sequence must not be empty")

        root = TreeNode(node_id=raw_sequence[0])
        stack: List[TreeNode] = [root]
        stack_ids: Set[int] = {raw_sequence[0]}
        node_map: Dict[int, TreeNode] = {raw_sequence[0]: root}
        globally_seen: Set[int] = {raw_sequence[0]}

        for v in raw_sequence[1:]:
            if v in stack_ids:
                # 回溯：弹栈直至栈顶为 v
                while stack and stack[-1].node_id != v:
                    removed = stack.pop()
                    stack_ids.discard(removed.node_id)
            else:
                if v not in globally_seen:
                    parent = stack[-1]
                    score = bias_func(parent.node_id, v)
                    child = TreeNode(node_id=v, bias_score=score)
                    parent.add_child(child)
                    node_map[v] = child
                    globally_seen.add(v)
                stack.append(node_map[v])
                stack_ids.add(v)

        root.sort_children_by_bias(descending=True)
        return root

    def assign_levels(self, root: TreeNode) -> Dict[int, int]:
        """DFS 为树中节点分配层级，根为 0。"""
        levels: Dict[int, int] = {}

        def dfs(node: TreeNode, depth: int) -> None:
            levels[node.node_id] = depth
            for child in node.children:
                dfs(child, depth + 1)

        dfs(root, 0)
        return levels

    def serialize_tree(
        self,
        root: TreeNode,
        tokenized: SubgraphTokenizedView,
        node_levels: Dict[int, int],
    ) -> str:
        """
        先序 DFS 展平为缩进文本，每行::
            {level} {struct_token} [t1, t2, ...]
        """
        lines: List[str] = []

        def preorder(node: TreeNode) -> None:
            nid = node.node_id
            if nid not in tokenized.nodes:
                logger.warning("Node %d missing in tokenized view; skipped", nid)
                return
            if nid not in node_levels:
                logger.warning("Node %d missing level; skipped", nid)
                return
            repr_ = tokenized.nodes[nid]
            level = node_levels[nid]
            tokens_str = ", ".join(repr_.text_tokens)
            lines.append(f"{level} {repr_.struct_token} [{tokens_str}]")
            for child in node.children:
                preorder(child)

        preorder(root)
        return "\n".join(lines)

    def serialize(
        self,
        subgraph: LocalSubgraph,
        tokenized: SubgraphTokenizedView,
        text_embeddings: torch.Tensor,
        bias_func: Optional[BiasFunction] = None,
        degrees: Optional[np.ndarray] = None,
        pagerank: Optional[np.ndarray] = None,
    ) -> SerializationResult:
        """模块4 完整流水线。"""
        del degrees, pagerank  # 子图内统计，不使用全图数组

        norm_degree, norm_pagerank = self.compute_subgraph_stats(subgraph)
        bias = (
            bias_func
            or self.bias_func
            or self.build_default_bias_function(
                subgraph,
                norm_degree,
                norm_pagerank,
                text_embeddings,
            )
        )

        s_raw, all_visited, stopped = self.biased_euler_tour(
            subgraph, tokenized, bias
        )
        tree_root = self.compress_to_tree(s_raw, bias)
        node_levels = self.assign_levels(tree_root)
        text = self.serialize_tree(tree_root, tokenized, node_levels)

        return SerializationResult(
            raw_sequence=s_raw,
            tree_root=tree_root,
            text=text,
            node_levels=node_levels,
            all_edges_visited=all_visited,
            stopped_by_max_steps=stopped,
        )


def _run_smoke_test(args: argparse.Namespace) -> None:
    from graph_utils import (
        extract_sentence_bert_embeddings,
        load_graph_data,
        setup_logging,
    )
    from models.codebook_trainer import CodebookTrainer, TFIDFComputer
    from models.node_representation import NodeRepresentationTokenizer
    from models.subgraph_extraction import SubgraphExtractor

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

    from text_tokenizers.text_tokenbook import TextTokenbook

    g, feats, _, _, _, _, text_dict = load_graph_data(
        args.dataset,
        root=args.data_root,
        seed=args.seed,
    )

    tokenbook = TextTokenbook.load(
        args.tokenbook_dir,
        model_name=args.sentence_bert,
        device=device,
    )

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

    serializer = BiasedEulerSerializer()
    result = serializer.serialize(subgraph, view, text_emb)

    adj_out, directed_edges = serializer.preprocess_directed_edges(subgraph)
    logger.info(
        "Serialization: |S_raw|=%d |directed_edges|=%d visited_ratio=%.2f "
        "all_visited=%s stopped_by_max_steps=%s",
        len(result.raw_sequence),
        len(directed_edges),
        len(set(zip(result.raw_sequence[:-1], result.raw_sequence[1:])))
        / max(len(directed_edges), 1),
        result.all_edges_visited,
        result.stopped_by_max_steps,
    )
    logger.info("S_raw: %s", result.raw_sequence)

    line_pat = re.compile(r"^\d+ <S_\d+> \[.+\]$")
    for line in result.text.splitlines():
        assert line_pat.match(line), f"bad line format: {line!r}"

    def count_tree_nodes(node: TreeNode) -> int:
        return 1 + sum(count_tree_nodes(c) for c in node.children)

    tree_nodes = count_tree_nodes(result.tree_root)
    assert len(result.text.splitlines()) == tree_nodes, (
        "line count must equal tree nodes"
    )
    assert len(result.node_levels) == len(view.nodes), (
        "each subgraph node should appear exactly once"
    )

    center = view.nodes[view.center_node]
    first_line = result.text.splitlines()[0]
    assert first_line.startswith("0 "), "first line must be level 0"
    assert center.struct_token in first_line

    print("=== Serialized text ===")
    print(result.text)

    if args.verbose:
        print("\n=== S_raw ===")
        print(result.raw_sequence)
        print("\n=== node_levels ===")
        print(result.node_levels)

    logger.info("Smoke test passed: %d lines serialized", tree_nodes)


def _parse_main_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="模块4 偏置欧拉游走序列化冒烟测试")
    p.add_argument("--codebook_dir", type=str, required=True)
    p.add_argument("--tokenbook_dir", type=str, default="./codebook")
    p.add_argument("--dataset", type=str, default="cora")
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--node", type=int, default=0)
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=int, default=0, help="-1 for CPU")
    p.add_argument("--no_tfidf", action="store_true")
    p.add_argument("--sentence_bert", type=str, default="all-MiniLM-L6-v2")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    _run_smoke_test(_parse_main_args())
