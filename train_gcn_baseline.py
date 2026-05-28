"""
正式 GCN 基线：在 Cora（文本 DGL，140/500/1000 划分）上训练标准 GCN 分类器。

与结构码本教师（VQ-VAE + GCN）不同，本脚本仅优化节点分类交叉熵，无码本重建损失。

用法（服务器）::

    cd ~/huanghp_2252895/VQGraph
    conda activate 2252895_vqgraph
    python train_gcn_baseline.py --dataset cora --data_root ./data --device 0
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from graph_utils import (
    DATASET_CLASS_NAMES,
    load_graph_data,
    set_seed,
    setup_logging,
)
from models.gcn_baseline import GCNClassifier
from utils import check_writable, get_evaluator, get_training_config

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train standard GCN node-classification baseline (no VQ)"
    )
    p.add_argument("--dataset", type=str, default="cora")
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/baseline_gcn",
        help="Root dir; run saved to {output_dir}/{dataset}/seed_{seed}/",
    )
    p.add_argument("--model_config_path", type=str, default="./train.conf.yaml")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=int, default=0, help="-1 for CPU")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=100)
    p.add_argument("--eval_interval", type=int, default=1)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--hidden_dim", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument(
        "--data_source",
        type=str,
        default=None,
        choices=["auto", "text", "cpf"],
        help="auto=优先 data/dataset/{name}/ 文本 DGL",
    )
    p.add_argument("--console_log", action="store_true")
    p.add_argument("--save_model", action="store_true", help="Save best model.pth")
    return p.parse_args()


def build_conf(args: argparse.Namespace, feat_dim: int, label_dim: int, device: torch.device) -> Dict[str, Any]:
    conf = get_training_config(args.model_config_path, "GCN", args.dataset)
    conf["model_name"] = "GCN"
    conf["feat_dim"] = feat_dim
    conf["label_dim"] = label_dim
    conf["device"] = device
    conf["num_layers"] = args.num_layers
    conf["max_epoch"] = args.epochs
    conf["patience"] = args.patience
    conf["eval_interval"] = args.eval_interval
    conf["seed"] = args.seed
    conf["dataset"] = args.dataset
    if args.hidden_dim is not None:
        conf["hidden_dim"] = args.hidden_dim
    if args.dropout is not None:
        conf["dropout_ratio"] = args.dropout
    if args.learning_rate is not None:
        conf["learning_rate"] = args.learning_rate
    if args.weight_decay is not None:
        conf["weight_decay"] = args.weight_decay
    conf.setdefault("hidden_dim", 64)
    conf.setdefault("dropout_ratio", 0.5)
    conf.setdefault("learning_rate", 0.01)
    conf.setdefault("weight_decay", 5e-4)
    return conf


def per_class_accuracy(
    pred: torch.Tensor,
    labels: torch.Tensor,
    idx: torch.Tensor,
    num_classes: int,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """按类统计准确率。"""
    pred = pred[idx].cpu()
    gold = labels[idx].cpu()
    results: Dict[str, Any] = {}
    for c in range(num_classes):
        mask = gold == c
        n = int(mask.sum().item())
        if n == 0:
            continue
        acc = float((pred[mask] == gold[mask]).float().mean().item())
        name = class_names[c] if class_names and c < len(class_names) else f"Class_{c}"
        results[name] = {"accuracy": acc, "count": n, "correct": int((pred[mask] == gold[mask]).sum().item())}
    return results


def train_epoch(
    model: GCNClassifier,
    g: Any,
    feats: torch.Tensor,
    labels: torch.Tensor,
    idx_train: torch.Tensor,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
) -> float:
    model.train()
    optimizer.zero_grad()
    logits = model(g, feats)
    loss = criterion(logits[idx_train], labels[idx_train])
    loss.backward()
    optimizer.step()
    return float(loss.item())


@torch.no_grad()
def evaluate(
    model: GCNClassifier,
    g: Any,
    feats: torch.Tensor,
    labels: torch.Tensor,
    idx: torch.Tensor,
    criterion: nn.Module,
    evaluator: Any,
) -> Tuple[float, float, torch.Tensor]:
    model.eval()
    logits = model(g, feats)
    loss = criterion(logits[idx], labels[idx]).item()
    out = logits.log_softmax(dim=1)
    acc = evaluator(out[idx], labels[idx])
    pred = out.argmax(dim=1)
    return loss, acc, pred


def run_training(
    g: Any,
    feats: torch.Tensor,
    labels: torch.Tensor,
    idx_train: torch.Tensor,
    idx_val: torch.Tensor,
    idx_test: torch.Tensor,
    conf: Dict[str, Any],
    output_dir: Path,
    save_model: bool = False,
) -> Dict[str, Any]:
    import dgl

    device = conf["device"]
    g = dgl.add_self_loop(g).to(device)
    feats = feats.to(device)
    labels = labels.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)

    model = GCNClassifier(
        in_dim=conf["feat_dim"],
        hidden_dim=conf["hidden_dim"],
        out_dim=conf["label_dim"],
        num_layers=conf["num_layers"],
        dropout=conf["dropout_ratio"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=conf["learning_rate"],
        weight_decay=conf["weight_decay"],
    )
    evaluator = get_evaluator(conf["dataset"])
    class_names = DATASET_CLASS_NAMES.get(conf["dataset"].lower())

    best_epoch, best_val, count = 0, 0.0, 0
    state = copy.deepcopy(model.state_dict())
    history: List[List[float]] = []

    for epoch in range(1, conf["max_epoch"] + 1):
        loss_train = train_epoch(model, g, feats, labels, idx_train, criterion, optimizer)
        if epoch % conf["eval_interval"] == 0:
            _, acc_train, _ = evaluate(model, g, feats, labels, idx_train, criterion, evaluator)
            _, acc_val, _ = evaluate(model, g, feats, labels, idx_val, criterion, evaluator)
            _, acc_test, _ = evaluate(model, g, feats, labels, idx_test, criterion, evaluator)
            history.append([epoch, loss_train, acc_train, acc_val, acc_test])
            logger.info(
                "Ep %3d | loss %.4f | train %.4f | val %.4f | test %.4f",
                epoch,
                loss_train,
                acc_train,
                acc_val,
                acc_test,
            )
            if acc_val >= best_val:
                best_epoch = epoch
                best_val = acc_val
                state = copy.deepcopy(model.state_dict())
                count = 0
            else:
                count += 1
        if count >= conf["patience"] or epoch == conf["max_epoch"]:
            break

    model.load_state_dict(state)
    _, acc_train, pred = evaluate(model, g, feats, labels, idx_train, criterion, evaluator)
    _, acc_val, _ = evaluate(model, g, feats, labels, idx_val, criterion, evaluator)
    _, acc_test, _ = evaluate(model, g, feats, labels, idx_test, criterion, evaluator)

    per_train = per_class_accuracy(pred, labels, idx_train, conf["label_dim"], class_names)
    per_val = per_class_accuracy(pred, labels, idx_val, conf["label_dim"], class_names)
    per_test = per_class_accuracy(pred, labels, idx_test, conf["label_dim"], class_names)

    metrics = {
        "model": "GCN_baseline",
        "dataset": conf["dataset"],
        "data_source": conf.get("data_source", "auto"),
        "split": {
            "train": int(len(idx_train)),
            "val": int(len(idx_val)),
            "test": int(len(idx_test)),
        },
        "feat_dim": conf["feat_dim"],
        "best_epoch": best_epoch,
        "accuracy": {
            "train": acc_train,
            "val": acc_val,
            "test": acc_test,
        },
        "per_class_val": per_val,
        "per_class_test": per_test,
        "per_class_train": per_train,
        "hyperparameters": {
            "num_layers": conf["num_layers"],
            "hidden_dim": conf["hidden_dim"],
            "dropout": conf["dropout_ratio"],
            "learning_rate": conf["learning_rate"],
            "weight_decay": conf["weight_decay"],
            "max_epoch": conf["max_epoch"],
            "patience": conf["patience"],
            "seed": conf["seed"],
        },
        "note": (
            "Standard GCN classifier (CE only). Compare with VQ codebook teacher "
            "(train_codebook.py) and LLM QLoRA (finetune_llm.py) on the same masks."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "baseline_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with open(output_dir / "train_conf.json", "w", encoding="utf-8") as f:
        json.dump(
            {k: (str(v) if isinstance(v, torch.device) else v) for k, v in conf.items()},
            f,
            indent=2,
        )
    if history:
        np.savez(output_dir / "loss_and_score.npz", history=np.array(history))
    if save_model:
        torch.save(model.state_dict(), output_dir / "model.pth")

    logger.info(
        "Best epoch %d | train=%.4f val=%.4f test=%.4f | saved %s",
        best_epoch,
        acc_train,
        acc_val,
        acc_test,
        metrics_path,
    )
    return metrics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device("cpu")

    output_dir = Path(args.output_dir) / args.dataset / f"seed_{args.seed}"
    check_writable(output_dir, overwrite=True)
    setup_logging(__name__, log_file=output_dir / "baseline.log")
    if args.console_log:
        logging.getLogger(__name__).setLevel(logging.INFO)

    data_source = None if (args.data_source is None or args.data_source == "auto") else args.data_source
    g, feats, labels, idx_train, idx_val, idx_test, _text = load_graph_data(
        args.dataset,
        root=args.data_root,
        seed=args.seed,
        data_source=data_source,
    )
    logger.info(
        "Loaded %s: nodes=%d feat_dim=%d train=%d val=%d test=%d device=%s",
        args.dataset,
        g.num_nodes(),
        feats.shape[1],
        len(idx_train),
        len(idx_val),
        len(idx_test),
        device,
    )

    conf = build_conf(args, feats.shape[1], int(labels.max().item()) + 1, device)
    conf["data_source"] = args.data_source or "auto"
    metrics = run_training(
        g,
        feats,
        labels,
        idx_train,
        idx_val,
        idx_test,
        conf,
        output_dir,
        save_model=args.save_model,
    )
    print(
        f"GCN baseline | val={metrics['accuracy']['val']:.4f} "
        f"test={metrics['accuracy']['test']:.4f} | {output_dir}"
    )


if __name__ == "__main__":
    main()
