"""
模块1：结构码本训练器 — 语义偏置 VQ、λ 预热、离线 TF-IDF。
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import dgl

from config import Config, get_config
from graph_utils import (
    build_tfidf_vocab,
    texts_to_token_ids,
)
from models._root_bridge import load_root_models_module
from models.semantic_vq import SemanticVectorQuantize
from utils import get_evaluator, get_training_config

Model = load_root_models_module().Model

logger = logging.getLogger(__name__)


@dataclass
class CodebookArtifacts:
    encoder_state_dict: Dict[str, Any]
    codebook_embeddings: torch.Tensor
    semantic_centers: torch.Tensor
    node_code_assignments: Optional[np.ndarray] = None
    save_dir: Optional[Path] = None


@dataclass
class TFIDFStatistics:
    count_matrix: np.ndarray
    df: np.ndarray
    tfidf_norm: np.ndarray
    vocab_size: int
    num_codes: int
    token_to_id: Optional[Dict[str, int]] = None

    def get_prior_weights(self, code_idx: int) -> np.ndarray:
        row = self.tfidf_norm[code_idx]
        if row.max() < 1e-8:
            return np.zeros_like(row)
        return row


def _wrap_semantic_vq(model: Model, text_dim: int, cfg: Config) -> SemanticVectorQuantize:
    enc = model.encoder
    device = next(enc.parameters()).device
    svq = SemanticVectorQuantize(
        enc.vq,
        text_dim=text_dim,
        lambda_semantic=cfg.lambda_semantic,
        ema_beta=cfg.ema_beta,
    ).to(device)
    enc.vq = svq
    return svq


def train_semantic_fixed(
    model: Model,
    data: Any,
    feats: torch.Tensor,
    labels: torch.Tensor,
    text_emb: torch.Tensor,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    idx_train: torch.Tensor,
    lamb: float = 1.0,
) -> float:
    model.train()
    optimizer.zero_grad()
    _, logits, loss, _, _ = model(data, feats, text_emb=text_emb)
    out = logits.log_softmax(dim=1)
    loss = loss + criterion(out[idx_train], labels[idx_train])
    loss_val = loss.item()
    (loss * lamb).backward()
    optimizer.step()
    return loss_val


def train_sage_semantic(
    model: Model,
    dataloader: Any,
    feats: torch.Tensor,
    labels: torch.Tensor,
    text_emb: torch.Tensor,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    lamb: float = 1.0,
) -> float:
    device = feats.device
    model.train()
    total_loss = 0.0
    n_batches = 0
    for _step, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        blocks = [blk.int().to(device) for blk in blocks]
        batch_feats = feats[input_nodes]
        batch_labels = labels[output_nodes]
        batch_text = text_emb[input_nodes]
        optimizer.zero_grad()
        _, logits, loss, _, _ = model(blocks, batch_feats, text_emb=batch_text)
        out = logits.log_softmax(dim=1)
        loss = loss + criterion(out, batch_labels)
        total_loss += loss.item()
        (loss * lamb).backward()
        optimizer.step()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def evaluate_semantic(
    model: Model,
    data: Any,
    feats: torch.Tensor,
    labels: torch.Tensor,
    text_emb: torch.Tensor,
    criterion: nn.Module,
    evaluator: Any,
    idx_eval: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, float, float, Any, torch.Tensor, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        _, logits, _, dist, codebook = model.inference(data, feats, text_emb=text_emb)
        out = logits.log_softmax(dim=1)
        if idx_eval is None:
            loss = criterion(out, labels).item()
            score = evaluator(out, labels)
        else:
            loss = criterion(out[idx_eval], labels[idx_eval]).item()
            score = evaluator(out[idx_eval], labels[idx_eval])
    return out, loss, score, None, dist, codebook


def assign_node_codes(
    model: Model,
    g: Any,
    feats: torch.Tensor,
    text_emb: torch.Tensor,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    """全图节点结构码 argmax。"""
    model.eval()
    n = g.num_nodes()
    codes = np.zeros(n, dtype=np.int64)
    if "SAGE" in model.model_name:
        g.create_formats_()
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        loader = dgl.dataloading.DataLoader(
            g,
            torch.arange(n),
            sampler,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        with torch.no_grad():
            _, _, _, dist_all, _ = model.inference(
                loader, feats.to(device), text_emb=text_emb.to(device)
            )
        codes = dist_all.argmax(dim=1).cpu().numpy()
    else:
        with torch.no_grad():
            _, _, _, dist, _ = model.inference(
                g.to(device), feats.to(device), text_emb=text_emb.to(device)
            )
        if dist.dim() == 3:
            dist = dist.squeeze(0)
        codes = dist.argmax(dim=1).cpu().numpy()
    return codes


class CodebookTrainer:
    """结构码本训练：DGL + GCN/SAGE + 语义偏置 VQ + λ 预热。"""

    def __init__(self, cfg: Optional[Config] = None) -> None:
        self.cfg = cfg or get_config()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model: Optional[Model] = None
        self.svq: Optional[SemanticVectorQuantize] = None
        self.text_embeddings: Optional[torch.Tensor] = None
        self.current_epoch: int = 0

    def build_model(self, conf: Dict[str, Any]) -> Model:
        model = Model(conf)
        self.model = model
        return model

    def wrap_semantic_vq(self, text_dim: int) -> SemanticVectorQuantize:
        assert self.model is not None
        self.svq = _wrap_semantic_vq(self.model, text_dim, self.cfg)
        return self.svq

    def fit(
        self,
        g: Any,
        feats: torch.Tensor,
        labels: torch.Tensor,
        text_embeddings: torch.Tensor,
        idx_train: torch.Tensor,
        idx_val: torch.Tensor,
        idx_test: torch.Tensor,
        conf: Dict[str, Any],
        output_dir: Path,
        logger_inst: Optional[logging.Logger] = None,
    ) -> CodebookArtifacts:
        log = logger_inst or logger
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        device = conf["device"]
        feats = feats.to(device)
        labels = labels.to(device)
        text_embeddings = text_embeddings.to(device)
        self.text_embeddings = text_embeddings

        model = self.build_model(conf)
        self.wrap_semantic_vq(text_embeddings.shape[1])

        criterion = nn.NLLLoss()
        evaluator = get_evaluator(conf["dataset"])
        lr = conf.get("learning_rate") or self.cfg.codebook_lr
        wd = conf.get("weight_decay")
        if wd is None:
            wd = 0.0005
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        max_epoch = conf.get("max_epoch", self.cfg.codebook_train_epochs)
        warmup_epochs = self.cfg.warmup_epochs
        patience = conf.get("patience", 50)
        eval_interval = conf.get("eval_interval", 1)

        if "SAGE" in model.model_name:
            g.create_formats_()
            sampler = dgl.dataloading.MultiLayerNeighborSampler(
                [eval(f) for f in conf["fan_out"].split(",")]
            )
            train_loader = dgl.dataloading.DataLoader(
                g,
                idx_train,
                sampler,
                batch_size=conf["batch_size"],
                shuffle=True,
                drop_last=False,
                num_workers=conf.get("num_workers", 0),
            )
            sampler_eval = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            eval_loader = dgl.dataloading.DataLoader(
                g,
                torch.arange(g.num_nodes()),
                sampler_eval,
                batch_size=conf["batch_size"],
                shuffle=False,
                num_workers=conf.get("num_workers", 0),
            )
            data_train, data_eval = train_loader, eval_loader
        else:
            g = g.to(device)
            data_train, data_eval = g, g

        best_epoch, best_score_val, count = 0, 0.0, 0
        state = copy.deepcopy(model.state_dict())

        for epoch in range(1, max_epoch + 1):
            self.current_epoch = epoch
            in_warmup = epoch <= warmup_epochs
            lam_eff = 0.0 if in_warmup else self.cfg.lambda_semantic
            assert self.svq is not None
            self.svq.set_lambda_effective(lam_eff)

            if "SAGE" in model.model_name:
                loss = train_sage_semantic(
                    model,
                    data_train,
                    feats,
                    labels,
                    text_embeddings,
                    criterion,
                    optimizer,
                )
            else:
                loss = train_semantic_fixed(
                    model,
                    data_train,
                    feats,
                    labels,
                    text_embeddings,
                    criterion,
                    optimizer,
                    idx_train,
                )

            if epoch % eval_interval == 0:
                out, loss_val_f, score_val, _, dist, codebook = evaluate_semantic(
                    model,
                    data_eval,
                    feats,
                    labels,
                    text_embeddings,
                    criterion,
                    evaluator,
                    idx_val,
                )
                score_test = evaluator(out[idx_test], labels[idx_test])
                log.info(
                    "Ep %3d | warmup=%s | lambda_eff=%.4f | loss=%.4f | "
                    "s_val=%.4f | s_test=%.4f",
                    epoch,
                    in_warmup,
                    lam_eff,
                    loss,
                    score_val,
                    score_test,
                )
                if score_val >= best_score_val:
                    best_epoch = epoch
                    best_score_val = score_val
                    state = copy.deepcopy(model.state_dict())
                    count = 0
                else:
                    count += 1

            if count >= patience or epoch == max_epoch:
                break

        model.load_state_dict(state)
        log.info("Best epoch %d, val acc %.4f", best_epoch, best_score_val)

        node_codes = assign_node_codes(
            model, g, feats, text_embeddings, device, conf.get("batch_size", 512)
        )
        assert self.svq is not None
        artifacts = CodebookArtifacts(
            encoder_state_dict=state,
            codebook_embeddings=self.svq.codebook.detach().cpu(),
            semantic_centers=self.svq.semantic_centers.detach().cpu(),
            node_code_assignments=node_codes,
            save_dir=output_dir,
        )
        self.save_artifacts(artifacts, output_dir, train_conf=conf)
        return artifacts

    @staticmethod
    def _jsonify_conf(conf: Dict[str, Any]) -> Dict[str, Any]:
        """将训练 conf 转为可 JSON 序列化的字典。"""
        out: Dict[str, Any] = {}
        for key, val in conf.items():
            if isinstance(val, torch.device):
                out[key] = str(val)
            elif isinstance(val, Path):
                out[key] = str(val)
            else:
                out[key] = val
        return out

    def save_artifacts(
        self,
        artifacts: CodebookArtifacts,
        save_dir: Path,
        train_conf: Optional[Dict[str, Any]] = None,
    ) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(artifacts.encoder_state_dict, save_dir / "model.pth")
        np.savez(
            save_dir / "codebook_embeddings.npz",
            artifacts.codebook_embeddings.numpy(),
        )
        np.savez(
            save_dir / "semantic_centers.npz",
            artifacts.semantic_centers.numpy(),
        )
        if artifacts.node_code_assignments is not None:
            np.savez(
                save_dir / "node_codes.npz",
                node_codes=artifacts.node_code_assignments,
            )
        cfg_snap = {
            "lambda_semantic": self.cfg.lambda_semantic,
            "warmup_epochs": self.cfg.warmup_epochs,
            "ema_beta": self.cfg.ema_beta,
            "codebook_size": self.cfg.codebook_size,
        }
        with open(save_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(cfg_snap, f, indent=2)
        if train_conf is not None:
            with open(save_dir / "train_conf.json", "w", encoding="utf-8") as f:
                json.dump(self._jsonify_conf(train_conf), f, indent=2)
        logger.info("Saved artifacts to %s", save_dir)

    @staticmethod
    def load_artifacts(load_dir: Path, device: torch.device) -> CodebookArtifacts:
        load_dir = Path(load_dir)
        state = torch.load(load_dir / "model.pth", map_location=device)
        cb = np.load(load_dir / "codebook_embeddings.npz")
        mu = np.load(load_dir / "semantic_centers.npz")
        codes = None
        nc_path = load_dir / "node_codes.npz"
        if nc_path.exists():
            nc = np.load(nc_path)
            codes = nc["node_codes"] if "node_codes" in nc else nc[nc.files[0]]
        arr_key = list(cb.keys())[0]
        return CodebookArtifacts(
            encoder_state_dict=state,
            codebook_embeddings=torch.tensor(cb[arr_key], dtype=torch.float32),
            semantic_centers=torch.tensor(mu[list(mu.keys())[0]], dtype=torch.float32),
            node_code_assignments=codes,
            save_dir=load_dir,
        )

    @staticmethod
    def build_conf_from_args(
        args: Any,
        feat_dim: int,
        label_dim: int,
        device: torch.device,
    ) -> Dict[str, Any]:
        conf = {}
        if getattr(args, "model_config_path", None):
            conf = get_training_config(
                args.model_config_path, args.teacher, args.dataset
            )
        args_dict = {k: v for k, v in vars(args).items() if v is not None}
        conf = dict(args_dict, **conf)
        conf["feat_dim"] = feat_dim
        conf["label_dim"] = label_dim
        conf["device"] = device
        conf["model_name"] = args.teacher
        conf["max_epoch"] = getattr(args, "epochs", None) or conf.get(
            "max_epoch", 100
        )
        conf["codebook_size"] = getattr(args, "codebook_size", None) or conf.get(
            "codebook_size", 2048
        )
        defaults = {
            "norm_type": "none",
            "dropout_ratio": 0.0,
            "num_layers": 2,
            "hidden_dim": 128,
            "learning_rate": 1e-4,
            "weight_decay": 5e-4,
            "batch_size": 512,
            "fan_out": "5,5",
            "eval_interval": 1,
            "num_workers": 0,
            "lamb_node": 0.001,
            "lamb_edge": 0.03,
            "patience": 50,
        }
        for k, v in defaults.items():
            conf.setdefault(k, v)
        return conf


class TFIDFComputer:
    """离线 TF-IDF：count[C][V] -> TF-IDF_norm。"""

    def __init__(self, cfg: Optional[Config] = None) -> None:
        self.cfg = cfg or get_config()

    def build_count_matrix(
        self,
        node_codes: np.ndarray,
        node_text_token_ids: List[List[int]],
        vocab_size: int,
        num_codes: int,
    ) -> np.ndarray:
        count = np.zeros((num_codes, vocab_size), dtype=np.float64)
        for code, tok_ids in zip(node_codes, node_text_token_ids):
            c = int(code)
            if c < 0 or c >= num_codes:
                continue
            for t in tok_ids:
                if 0 <= t < vocab_size:
                    count[c, t] += 1.0
        return count

    def compute_tfidf(self, count_matrix: np.ndarray) -> TFIDFStatistics:
        C, V = count_matrix.shape
        df = np.zeros(V, dtype=np.float64)
        for t in range(V):
            df[t] = float(np.sum(count_matrix[:, t] > 0))

        tfidf = np.zeros_like(count_matrix)
        for c in range(C):
            row_sum = count_matrix[c].sum()
            for t in range(V):
                if count_matrix[c, t] <= 0:
                    continue
                tf = count_matrix[c, t] / (row_sum + 1e-8)
                idf = np.log(C / (df[t] + 1.0))
                tfidf[c, t] = tf * idf

        tfidf_norm = np.zeros_like(tfidf)
        for c in range(C):
            row_max = tfidf[c].max()
            if row_max > 1e-8:
                tfidf_norm[c] = tfidf[c] / row_max

        return TFIDFStatistics(
            count_matrix=count_matrix,
            df=df,
            tfidf_norm=tfidf_norm,
            vocab_size=V,
            num_codes=C,
        )

    def run_offline(
        self,
        artifacts: CodebookArtifacts,
        text_dict: Dict[int, str],
        train_node_ids: np.ndarray,
        save_path: Path,
        max_vocab: Optional[int] = None,
    ) -> TFIDFStatistics:
        save_path = Path(save_path)
        max_vocab = max_vocab or self.cfg.text_vocab_size
        num_codes = artifacts.codebook_embeddings.shape[0]

        train_texts = [text_dict[int(i)] for i in train_node_ids]
        token_to_id, id_to_token = build_tfidf_vocab(train_texts, max_vocab=max_vocab)
        vocab_size = len(token_to_id)

        all_ids = texts_to_token_ids(
            text_dict, token_to_id, num_nodes=len(text_dict)
        )
        node_codes = artifacts.node_code_assignments
        if node_codes is None:
            raise ValueError("node_code_assignments required for TF-IDF")

        train_codes = node_codes[train_node_ids]
        train_token_ids = [all_ids[int(i)] for i in train_node_ids]

        count = self.build_count_matrix(
            train_codes, train_token_ids, vocab_size, num_codes
        )
        stats = self.compute_tfidf(count)
        stats.token_to_id = token_to_id

        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            save_path,
            count_matrix=stats.count_matrix,
            df=stats.df,
            tfidf_norm=stats.tfidf_norm,
            vocab_size=stats.vocab_size,
            num_codes=stats.num_codes,
        )
        vocab_path = save_path.with_suffix(".vocab.json")
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(token_to_id, f, ensure_ascii=False, indent=0)
        logger.info(
            "Saved TF-IDF to %s shape (%d, %d)",
            save_path,
            stats.num_codes,
            stats.vocab_size,
        )
        return stats

    @staticmethod
    def load(save_path: Path) -> TFIDFStatistics:
        save_path = Path(save_path)
        data = np.load(save_path)
        token_to_id = None
        vocab_path = save_path.with_suffix(".vocab.json")
        if vocab_path.exists():
            with open(vocab_path, encoding="utf-8") as f:
                token_to_id = json.load(f)
        return TFIDFStatistics(
            count_matrix=data["count_matrix"],
            df=data["df"],
            tfidf_norm=data["tfidf_norm"],
            vocab_size=int(data["vocab_size"]),
            num_codes=int(data["num_codes"]),
            token_to_id=token_to_id,
        )
