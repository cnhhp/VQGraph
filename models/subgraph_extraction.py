"""
模块2：局部子图提取（k-hop ego graph）。
"""

from __future__ import annotations

import argparse
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

from config import Config, get_config

logger = logging.getLogger(__name__)


@dataclass
class LocalSubgraph:
    """
    以目标节点为中心的局部子图。

    Attributes
    ----------
    center_node : 目标节点 ID（原图中的全局 ID）
    node_ids : 子图包含的节点 ID 列表（全局 ID）
    edge_index : [2, E_sub] 子图内边（全局 ID 或局部 ID，由 ``use_local_ids`` 决定）
    local_to_global : 局部索引 -> 全局 ID
    global_to_local : 全局 ID -> 局部索引（仅子图内节点）
    adjacency : 邻接表 dict[local_id, List[local_id]]（可选，便于序列化）
    """

    center_node: int
    node_ids: List[int]
    edge_index: torch.Tensor
    local_to_global: Dict[int, int] = field(default_factory=dict)
    global_to_local: Dict[int, int] = field(default_factory=dict)
    adjacency: Dict[int, List[int]] = field(default_factory=dict)
    use_local_ids: bool = False

    @property
    def num_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def num_edges(self) -> int:
        return self.edge_index.shape[1] if self.edge_index.numel() > 0 else 0


class SubgraphExtractor:
    """从全图 G 中为每个目标节点提取 k-hop 子图。"""

    def __init__(self, cfg: Optional[Config] = None) -> None:
        self.cfg = cfg or get_config()
        self.k: int = self.cfg.subgraph_k_hop
        self.max_nodes: Optional[int] = self.cfg.max_subgraph_nodes
        self._cached_edge_index: Optional[torch.Tensor] = None
        self._cached_adj: Optional[Dict[int, List[int]]] = None
        self._cached_degrees: Optional[Dict[int, int]] = None

    def _ensure_global_adj(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> Tuple[Dict[int, List[int]], Dict[int, int]]:
        """构建或复用全图无向邻接表与度字典。"""
        if (
            self._cached_edge_index is not None
            and self._cached_adj is not None
            and self._cached_degrees is not None
            and self._cached_edge_index.shape == edge_index.shape
            and torch.equal(self._cached_edge_index, edge_index)
        ):
            return self._cached_adj, self._cached_degrees

        adj = self._build_global_adj(edge_index, num_nodes)
        degrees = {n: len(neighbors) for n, neighbors in adj.items()}
        self._cached_edge_index = edge_index.clone()
        self._cached_adj = adj
        self._cached_degrees = degrees
        return adj, degrees

    @staticmethod
    def _build_global_adj(
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> Dict[int, List[int]]:
        """
        从 edge_index 构建无向邻接表。

        每条边 (u, v) 双向加入，跳过自环。
        """
        adj: Dict[int, Set[int]] = {i: set() for i in range(num_nodes)}
        if edge_index.numel() == 0:
            return {i: [] for i in range(num_nodes)}

        src = edge_index[0].tolist()
        dst = edge_index[1].tolist()
        for u, v in zip(src, dst):
            if u == v:
                continue
            if 0 <= u < num_nodes and 0 <= v < num_nodes:
                adj[u].add(v)
                adj[v].add(u)

        return {n: sorted(neighbors) for n, neighbors in adj.items()}

    def extract_subgraph(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        target_node: int,
        k: Optional[int] = None,
    ) -> LocalSubgraph:
        """
        BFS 提取距离 <= k 的诱导子图。

        Parameters
        ----------
        edge_index : [2, E] 全图边
        num_nodes : 节点总数
        target_node : 中心节点 v_t
        k : 跳数，默认使用 config.subgraph_k_hop

        Returns
        -------
        LocalSubgraph
        """
        if k is None:
            k = self.k
        if not (0 <= target_node < num_nodes):
            raise ValueError(
                f"target_node={target_node} out of range [0, {num_nodes})"
            )

        edge_index = edge_index.long().cpu()
        adj, degrees = self._ensure_global_adj(edge_index, num_nodes)

        # BFS 收集 k-hop 节点
        node_set = self._bfs_collect(adj, target_node, k)
        node_set = self._maybe_truncate(
            node_set, target_node, degrees, self.max_nodes
        )

        # 中心节点排首位，其余升序
        node_ids = [target_node] + sorted(n for n in node_set if n != target_node)
        local_to_global = {i: gid for i, gid in enumerate(node_ids)}
        global_to_local = {gid: i for i, gid in enumerate(node_ids)}

        sub_edge_index = self._induced_edges(edge_index, set(node_ids), num_nodes)
        adjacency = self._build_adjacency(
            sub_edge_index,
            global_to_local,
            len(node_ids),
            directed=False,
            edges_are_local=False,
        )

        return LocalSubgraph(
            center_node=target_node,
            node_ids=node_ids,
            edge_index=sub_edge_index,
            local_to_global=local_to_global,
            global_to_local=global_to_local,
            adjacency=adjacency,
            use_local_ids=False,
        )

    def extract_batch(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        target_nodes: List[int],
        k: Optional[int] = None,
    ) -> List[LocalSubgraph]:
        """批量提取，带进度日志。"""
        total = len(target_nodes)
        if total == 0:
            return []

        log_interval = max(1, min(500, total // 10 or 1))
        results: List[LocalSubgraph] = []

        for i, node in enumerate(target_nodes):
            results.append(
                self.extract_subgraph(edge_index, num_nodes, node, k=k)
            )
            if (i + 1) % log_interval == 0 or (i + 1) == total:
                logger.info(
                    "Subgraph extraction progress: %d / %d (%.1f%%)",
                    i + 1,
                    total,
                    100.0 * (i + 1) / total,
                )

        return results

    def extract_from_pyg_data(
        self,
        data: Any,
        target_node: int,
        k: Optional[int] = None,
    ) -> LocalSubgraph:
        """基于 PyG Data 对象的便捷接口；无 PyG 时回退到 edge_index + BFS。"""
        if k is None:
            k = self.k

        try:
            from torch_geometric.utils import k_hop_subgraph
        except ImportError:
            logger.debug("torch_geometric not installed; using BFS fallback.")
            edge_index = data.edge_index
            num_nodes = data.num_nodes
            if num_nodes is None:
                num_nodes = int(data.x.shape[0]) if hasattr(data, "x") else (
                    int(edge_index.max().item()) + 1
                )
            return self.extract_subgraph(edge_index, num_nodes, target_node, k=k)

        num_nodes = data.num_nodes
        if num_nodes is None:
            num_nodes = int(data.x.shape[0]) if hasattr(data, "x") else (
                int(data.edge_index.max().item()) + 1
            )

        subset, sub_edge_index, mapping, _edge_mask = k_hop_subgraph(
            target_node,
            k,
            data.edge_index,
            num_nodes=num_nodes,
            relabel_nodes=True,
        )

        # PyG relabel_nodes=True 时 subset 为原图全局 ID
        node_ids_py = subset.tolist()
        center_global = target_node
        if center_global not in node_ids_py:
            node_ids_py = [center_global] + [
                n for n in sorted(node_ids_py) if n != center_global
            ]
        else:
            node_ids_py = [center_global] + sorted(
                n for n in node_ids_py if n != center_global
            )

        # sub_edge_index 已是局部 ID；映射回全局 ID 的 edge_index
        local_to_global = {i: gid for i, gid in enumerate(node_ids_py)}
        global_to_local = {gid: i for i, gid in enumerate(node_ids_py)}

        global_edges = sub_edge_index.clone()
        for row in range(2):
            for col in range(global_edges.shape[1]):
                global_edges[row, col] = local_to_global[
                    int(global_edges[row, col].item())
                ]

        adjacency = self._build_adjacency(
            sub_edge_index,
            global_to_local,
            len(node_ids_py),
            directed=False,
            edges_are_local=True,
        )

        return LocalSubgraph(
            center_node=center_global,
            node_ids=node_ids_py,
            edge_index=global_edges,
            local_to_global=local_to_global,
            global_to_local=global_to_local,
            adjacency=adjacency,
            use_local_ids=False,
        )

    def extract_from_dgl_graph(
        self,
        g: Any,
        target_node: int,
        k: Optional[int] = None,
    ) -> LocalSubgraph:
        """基于 DGL 图的便捷接口（对接现有 dataloader）。"""
        from graph_utils import edge_index_from_dgl

        edge_index = edge_index_from_dgl(g)
        return self.extract_subgraph(
            edge_index, g.num_nodes(), target_node, k=k
        )

    def _bfs_collect(
        self,
        adj: Dict[int, List[int]],
        source: int,
        k: int,
    ) -> Set[int]:
        """
        BFS 收集距离 <= k 的节点集合。

        从 source 出发，逐层扩展至第 k 跳；与 k_hop_subgraph 语义一致。
        """
        if k < 0:
            raise ValueError(f"k must be >= 0, got {k}")

        visited: Set[int] = {source}
        queue: deque[Tuple[int, int]] = deque([(source, 0)])

        while queue:
            u, dist = queue.popleft()
            if dist >= k:
                continue
            for v in adj.get(u, []):
                if v not in visited:
                    visited.add(v)
                    queue.append((v, dist + 1))

        return visited

    def _induced_edges(
        self,
        edge_index: torch.Tensor,
        node_set: Set[int],
        num_nodes: int,
    ) -> torch.Tensor:
        """保留端点均在 node_set 内的边。"""
        if edge_index.numel() == 0 or not node_set:
            return torch.empty((2, 0), dtype=torch.long)

        in_set = torch.zeros(num_nodes, dtype=torch.bool)
        for n in node_set:
            if 0 <= n < num_nodes:
                in_set[n] = True

        src = edge_index[0]
        dst = edge_index[1]
        mask = in_set[src] & in_set[dst]
        if not mask.any():
            return torch.empty((2, 0), dtype=torch.long)
        return edge_index[:, mask].long()

    def _build_adjacency(
        self,
        edge_index: torch.Tensor,
        global_to_local: Dict[int, int],
        num_local: int,
        directed: bool = False,
        edges_are_local: bool = False,
    ) -> Dict[int, List[int]]:
        """
        构建子图邻接表（局部 ID）。

        默认无向：每条边双向记录，邻居列表去重升序。
        """
        adj: Dict[int, Set[int]] = {i: set() for i in range(num_local)}

        if edge_index.numel() == 0:
            return {i: [] for i in range(num_local)}

        src_list = edge_index[0].tolist()
        dst_list = edge_index[1].tolist()

        for u, v in zip(src_list, dst_list):
            if edges_are_local:
                lu, lv = int(u), int(v)
            else:
                if u not in global_to_local or v not in global_to_local:
                    continue
                lu, lv = global_to_local[u], global_to_local[v]

            if lu == lv or not (0 <= lu < num_local and 0 <= lv < num_local):
                continue
            adj[lu].add(lv)
            if not directed:
                adj[lv].add(lu)

        return {i: sorted(neighbors) for i, neighbors in adj.items()}

    def _maybe_truncate(
        self,
        node_set: Set[int],
        center: int,
        degrees: Dict[int, int],
        max_nodes: Optional[int],
    ) -> Set[int]:
        """
        子图过大时按全图度降序截断，始终保留中心节点。

        max_nodes 为 None 时不截断。
        """
        if max_nodes is None or len(node_set) <= max_nodes:
            return node_set

        others = [n for n in node_set if n != center]
        others.sort(key=lambda n: (-degrees.get(n, 0), n))
        kept = {center, *others[: max_nodes - 1]}
        logger.warning(
            "Subgraph around node %d truncated from %d to %d nodes (max_subgraph_nodes=%d)",
            center,
            len(node_set),
            len(kept),
            max_nodes,
        )
        return kept


def _run_smoke_test(args: argparse.Namespace) -> None:
    """Cora 等数据集上的冒烟验证。"""
    from graph_utils import edge_index_from_dgl, load_graph_data, setup_logging

    setup_logging(__name__)
    g, _, _, _, _, _, _ = load_graph_data(
        args.dataset,
        root=args.data_root,
        seed=0,
    )
    edge_index = edge_index_from_dgl(g)
    num_nodes = g.num_nodes()
    extractor = SubgraphExtractor()

    sub = extractor.extract_subgraph(
        edge_index, num_nodes, args.node, k=args.k
    )
    assert sub.center_node == args.node, "center_node mismatch"
    assert args.node in sub.node_ids, "center not in node_ids"
    assert sub.node_ids[0] == args.node, "center should be first in node_ids"

    node_set = set(sub.node_ids)
    if sub.num_edges > 0:
        for u, v in zip(sub.edge_index[0].tolist(), sub.edge_index[1].tolist()):
            assert u in node_set and v in node_set, "induced edge out of node set"

    assert set(sub.adjacency.keys()) == set(range(sub.num_nodes)), (
        "adjacency keys should be 0..num_nodes-1"
    )
    for local_id, neighbors in sub.adjacency.items():
        assert sub.local_to_global[local_id] == sub.node_ids[local_id]
        for nb in neighbors:
            assert 0 <= nb < sub.num_nodes

    if args.k2_check:
        sub2 = extractor.extract_subgraph(
            edge_index, num_nodes, args.node, k=args.k + 1
        )
        assert sub2.num_nodes >= sub.num_nodes, "k+1 should have >= nodes than k"

    logger.info(
        "Smoke test passed: node=%d k=%d |V|=%d |E|=%d",
        args.node,
        args.k,
        sub.num_nodes,
        sub.num_edges,
    )
    logger.info("node_ids (first 10): %s", sub.node_ids[:10])


def _parse_main_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="模块2 子图提取冒烟测试")
    p.add_argument("--dataset", type=str, default="cora")
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--node", type=int, default=0, help="目标节点 ID")
    p.add_argument("--k", type=int, default=1, help="跳数")
    p.add_argument(
        "--k2-check",
        action="store_true",
        help="额外验证 k+1 子图节点数 >= k",
    )
    return p.parse_args()


if __name__ == "__main__":
    _run_smoke_test(_parse_main_args())
