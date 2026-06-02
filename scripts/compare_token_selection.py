"""
对比「无 P_code 融合」(lambda_pred=0) 与「有 P_code 融合」(默认 lambda_pred) 的文本 token 选取差异。

用法（服务器，需已重训含 token_predictor 的码本）::

    python scripts/compare_token_selection.py \\
      --codebook_dir ./outputs/codebook/cora/GCN/seed_0 \\
      --tokenbook_dir ./codebook \\
      --data_root ./data --data_source text \\
      --nodes 0,1,2,3,4,100,500 --device 0

演示模式（无数据，合成得分展示机制）::

    python scripts/compare_token_selection.py --demo
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

from config import get_config, reset_config
from graph_utils import extract_sentence_bert_embeddings, load_graph_data, setup_logging
from models.codebook_trainer import CodebookTrainer, TFIDFComputer
from models.node_representation import NodeRepresentationTokenizer, select_diverse_tokens
from text_tokenizers.text_tokenbook import TextTokenbook

logger = logging.getLogger(__name__)


def _parse_nodes(s: str, max_nodes: int) -> List[int]:
    if s.lower() == "train":
        return list(range(min(140, max_nodes)))
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return [n for n in out if 0 <= n < max_nodes]


def _tokens_from_jsonl_line(line: str, level: int = 0) -> Optional[List[str]]:
    """从 preprocess JSONL 的 input 字段解析 level-0 节点的 text tokens。"""
    obj = json.loads(line)
    inp = obj.get("input", "")
    pat = re.compile(
        rf"^{level}\s+<S_\d+>\s+\[([^\]]*)\]",
        re.MULTILINE,
    )
    m = pat.search(inp)
    if not m:
        return None
    inner = m.group(1).strip()
    if not inner:
        return []
    return [t.strip() for t in inner.split(",")]


def compare_one_node(
    tokenizer: NodeRepresentationTokenizer,
    node_id: int,
    node_text_emb: torch.Tensor,
    struct_code: int,
    lambda_pred_on: float,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """返回 (tokens_off, tokens_on, added, removed)。"""
    cfg = tokenizer.cfg
    orig = cfg.lambda_pred

    cfg.lambda_pred = 0.0
    off, _ = tokenizer.select_text_tokens(node_text_emb, struct_code)

    cfg.lambda_pred = lambda_pred_on
    on, _ = tokenizer.select_text_tokens(node_text_emb, struct_code)

    cfg.lambda_pred = orig

    set_off, set_on = set(off), set(on)
    added = [t for t in on if t not in set_off]
    removed = [t for t in off if t not in set_on]
    return off, on, added, removed


def run_demo() -> None:
    """合成得分演示：P_code 如何把结构码偏好的词推入 Top-K。"""
    print("=" * 60)
    print("合成演示（机制说明，非 Cora 实测）")
    print("=" * 60)

    tokens = [
        "learning", "networks", "neural", "algorithm", "graph",
        "classification", "theory", "bayesian", "optimization", "nodes",
        "tensorflow", "parallel", "robot", "genetic", "database",
    ]
    V = len(tokens)
    id_to_token = {i: t for i, t in enumerate(tokens)}

    # 模拟节点文本相似度（标题/摘要语义）
    text_sim = np.array(
        [0.92, 0.88, 0.85, 0.80, 0.55, 0.50, 0.45, 0.40, 0.38, 0.35,
         0.72, 0.48, 0.30, 0.28, 0.25],
        dtype=np.float64,
    )
    # TF-IDF 先验（结构码 c 下常见词）
    prior = np.zeros(V)
    for i, w in enumerate(["graph", "nodes", "algorithm", "networks", "learning"]):
        prior[tokens.index(w)] = 1.0 - i * 0.15
    prior = prior / (prior.max() + 1e-8)

    # P_code：码向量预测分布（结构模式绑定词，如 graph/walk/robot）
    p_code = np.zeros(V)
    for w, p in [("graph", 0.25), ("nodes", 0.20), ("robot", 0.18), ("walk", 0.0)]:
        if w in tokens:
            p_code[tokens.index(w)] = p
    if "walk" not in tokens:
        p_code[tokens.index("graph")] += 0.05
    p_code[tokens.index("robot")] = 0.18

    lam_tfidf, lam_pred = 0.5, 0.5
    k = 5

    def fuse(use_p_code: bool) -> np.ndarray:
        mult = 1.0 + lam_tfidf * prior
        if use_p_code:
            mult = mult + lam_pred * p_code
        return text_sim * mult

    scores_off = fuse(False)
    scores_on = fuse(True)

    # MMR 需要嵌入；用随机嵌入仅演示排序（演示模式跳过 MMR 多样性，直接 top-k）
    top_off = [tokens[i] for i in np.argsort(-scores_off)[:k]]
    top_on = [tokens[i] for i in np.argsort(-scores_on)[:k]]

    print(f"\n融合公式: score = text_sim × (1 + {lam_tfidf}×TF-IDF + {lam_pred}×P_code)")
    print(f"\n无 P_code (lambda_pred=0) Top-{k}: {top_off}")
    print(f"有 P_code (lambda_pred={lam_pred}) Top-{k}: {top_on}")
    print(f"新增: {[t for t in top_on if t not in top_off]}")
    print(f"移除: {[t for t in top_off if t not in top_on]}")
    print(
        "\n解读: P_code 抬高结构码偏好的词（如 graph/robot），"
        "在 text_sim 接近时改变 Top-K 组成，使 token 更贴近结构模式。"
    )


def run_full(args: argparse.Namespace) -> None:
    reset_config()
    cfg = get_config()
    if args.lambda_pred is not None:
        cfg.lambda_pred = args.lambda_pred

    device = torch.device(
        f"cuda:{args.device}" if torch.cuda.is_available() and args.device >= 0 else "cpu"
    )
    codebook_dir = Path(args.codebook_dir)

    artifacts = CodebookTrainer.load_artifacts(codebook_dir, device)
    state = artifacts.encoder_state_dict
    has_predictor = (
        "encoder.token_predictor.weight" in state
        or "encoder.token_predictor.proj.weight" in state
    )
    if args.p_code_normalize is not None:
        cfg.p_code_normalize = args.p_code_normalize
    print(f"Checkpoint 含 token_predictor: {has_predictor}")
    print(f"p_code_normalize: {cfg.p_code_normalize}")

    tfidf = None
    tfidf_path = codebook_dir / "tfidf_stats.npz"
    if tfidf_path.exists():
        tfidf = TFIDFComputer.load(tfidf_path)

    data_source = None if args.data_source == "auto" else args.data_source
    g, feats, labels, idx_train, _, _, text_dict = load_graph_data(
        args.dataset,
        root=args.data_root,
        seed=args.seed,
        data_source=data_source,
    )
    text_emb = extract_sentence_bert_embeddings(
        text_dict,
        model_name=cfg.sentence_bert_model,
        device=device,
    )

    tokenbook = TextTokenbook.load(
        args.tokenbook_dir,
        cfg=cfg,
        model_name=cfg.sentence_bert_model,
        device=device,
    )

    tokenizer = NodeRepresentationTokenizer(
        artifacts=artifacts,
        tokenbook=tokenbook,
        tfidf=tfidf,
        cfg=cfg,
    )
    dist = tokenizer._infer_distances(g, feats, text_emb)
    struct_codes = dist.argmax(dim=1).cpu().numpy()

    node_ids = _parse_nodes(args.nodes, g.num_nodes())
    lambda_on = cfg.lambda_pred

    stats = {"total": 0, "changed": 0, "unchanged": 0, "jaccard_sum": 0.0}
    rows = []

    for nid in node_ids:
        sc = int(struct_codes[nid])
        off, on, added, removed = compare_one_node(
            tokenizer, nid, text_emb[nid], sc, lambda_on
        )
        changed = off != on
        stats["total"] += 1
        if changed:
            stats["changed"] += 1
        else:
            stats["unchanged"] += 1
        inter = len(set(off) & set(on))
        union = len(set(off) | set(on)) or 1
        stats["jaccard_sum"] += inter / union

        label = int(labels[nid].item()) if labels is not None else -1
        rows.append(
            {
                "node": nid,
                "struct_code": sc,
                "label": label,
                "without_pcode": off,
                "with_pcode": on,
                "added": added,
                "removed": removed,
                "changed": changed,
            }
        )

    print("\n" + "=" * 60)
    print(f"对比: lambda_pred=0  vs  lambda_pred={lambda_on}")
    print("=" * 60)
    for r in rows[: args.max_print]:
        ch = " *" if r["changed"] else ""
        print(f"\n节点 {r['node']}  结构码 <S_{r['struct_code']}>  label={r['label']}{ch}")
        print(f"  无 P_code: {r['without_pcode']}")
        print(f"  有 P_code: {r['with_pcode']}")
        if r["changed"]:
            print(f"  + 新增: {r['added']}")
            print(f"  - 移除: {r['removed']}")

    if stats["total"]:
        avg_j = stats["jaccard_sum"] / stats["total"]
        print("\n" + "-" * 60)
        print(
            f"汇总 ({stats['total']} 节点): "
            f"变化 {stats['changed']} ({100*stats['changed']/stats['total']:.1f}%), "
            f"不变 {stats['unchanged']}, "
            f"平均 Jaccard={avg_j:.3f}"
        )
    if not has_predictor:
        print(
            "\n注意: 当前 checkpoint 无 token_predictor，"
            "有/无 P_code 结果应完全相同。请用 --compute_tfidf 重训码本后再对比。"
        )

    if args.jsonl_baseline:
        jsonl_path = Path(args.jsonl_baseline)
        if jsonl_path.exists():
            print(f"\n与 JSONL 基线 ({jsonl_path.name}) 对照（level-0 节点）:")
            with open(jsonl_path, encoding="utf-8") as f:
                lines = f.readlines()
            for i, r in enumerate(rows[: min(args.max_print, len(lines))]):
                old = _tokens_from_jsonl_line(lines[i])
                if old is not None:
                    print(f"  节点 {r['node']} JSONL(v2旧): {old}")
                    print(f"           无P_code:        {r['without_pcode']}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="对比有无 P_code 的文本 token 选取")
    p.add_argument("--demo", action="store_true", help="合成演示，无需数据")
    p.add_argument("--codebook_dir", type=str, default="./outputs/codebook/cora/GCN/seed_0")
    p.add_argument("--tokenbook_dir", type=str, default="./codebook")
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--dataset", type=str, default="cora")
    p.add_argument("--data_source", type=str, default="text")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--nodes", type=str, default="0,1,2,3,4,10,50,100,500,1000")
    p.add_argument("--lambda_pred", type=float, default=None)
    p.add_argument(
        "--p_code_normalize",
        type=str,
        default=None,
        choices=["none", "max", "minmax"],
    )
    p.add_argument("--max_print", type=int, default=15)
    p.add_argument(
        "--jsonl_baseline",
        type=str,
        default="./data/llm_finetune_v2/train.jsonl",
        help="可选：与已生成 JSONL 中 level-0 tokens 对照",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.demo:
        run_demo()
        return
    setup_logging(__name__)
    run_full(args)


if __name__ == "__main__":
    main()
