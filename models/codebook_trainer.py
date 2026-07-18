"""
模块1：结构码本训练器 — 语义偏置 VQ、λ 预热、离线 TF-IDF。
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import dgl

from config import Config, get_config
from graph_utils import texts_to_tokenbook_ids
from models._root_bridge import load_root_models_module
from models.semantic_vq import SemanticVectorQuantize
from utils import get_evaluator, get_training_config

Model = load_root_models_module().Model

if TYPE_CHECKING:
    from text_tokenizers.text_tokenbook import TextTokenbook

logger = logging.getLogger(__name__)


@dataclass
class CodebookArtifacts:
    encoder_state_dict: Dict[str, Any]
    codebook_embeddings: torch.Tensor
    semantic_centers: torch.Tensor
    node_code_assignments: Optional[np.ndarray] = None
    save_dir: Optional[Path] = None
    # 层次粗/细码（可选）
    codebook_coarse: Optional[torch.Tensor] = None
    codebook_fine: Optional[torch.Tensor] = None
    node_codes_coarse: Optional[np.ndarray] = None
    node_codes_fine: Optional[np.ndarray] = None
    hierarchical_vq: bool = False


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
    # 层次模式：粗码 SemanticVQ 已在 HierarchicalGCN 内
    is_hier = getattr(model, "hierarchical_vq", False) or (
        getattr(enc, "vq_co", None) is not None
        and enc.__class__.__name__ == "HierarchicalGCN"
    )
    if is_hier:
        svq = enc.vq_co
        # 若构建时 text_dim 与 Sentence-BERT 不一致，重建 semantic_centers buffer
        if svq.text_dim != text_dim:
            logger.warning(
                "HierarchicalGCN vq_co text_dim=%d != %d; resizing semantic_centers",
                svq.text_dim,
                text_dim,
            )
            M = svq.codebook_size
            svq.text_dim = text_dim
            svq.register_buffer(
                "semantic_centers",
                torch.zeros(M, text_dim, device=next(enc.parameters()).device),
            )
        return svq
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
) -> Tuple[float, Optional[float]]:
    model.train()
    optimizer.zero_grad()
    if getattr(model, "hierarchical_vq", False):
        # 层次模式按原定方案：不加主分类 CE（分类留给 LLM）；仅码本自监督 + 可选弱 L_L
        _, logits, loss, _, _, token_loss_raw = model(
            data, feats, text_emb=text_emb, labels=labels
        )
    else:
        _, logits, loss, _, _, token_loss_raw = model(data, feats, text_emb=text_emb)
        out = logits.log_softmax(dim=1)
        loss = loss + criterion(out[idx_train], labels[idx_train])
    loss_val = loss.item()
    token_loss_val = float(token_loss_raw.item()) if token_loss_raw is not None else None
    (loss * lamb).backward()
    optimizer.step()
    return loss_val, token_loss_val


def train_sage_semantic(
    model: Model,
    dataloader: Any,
    feats: torch.Tensor,
    labels: torch.Tensor,
    text_emb: torch.Tensor,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    lamb: float = 1.0,
) -> Tuple[float, Optional[float]]:
    device = feats.device
    model.train()
    total_loss = 0.0
    total_token_loss = 0.0
    token_loss_batches = 0
    n_batches = 0
    for _step, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        blocks = [blk.int().to(device) for blk in blocks]
        batch_feats = feats[input_nodes]
        batch_labels = labels[output_nodes]
        batch_text = text_emb[input_nodes]
        optimizer.zero_grad()
        _, logits, loss, _, _, token_loss_raw = model(
            blocks, batch_feats, text_emb=batch_text
        )
        out = logits.log_softmax(dim=1)
        loss = loss + criterion(out, batch_labels)
        total_loss += loss.item()
        if token_loss_raw is not None:
            total_token_loss += float(token_loss_raw.item())
            token_loss_batches += 1
        (loss * lamb).backward()
        optimizer.step()
        n_batches += 1
    avg_token = (
        total_token_loss / token_loss_batches if token_loss_batches > 0 else None
    )
    return total_loss / max(n_batches, 1), avg_token


def train_predictor_code_level(
    model: Model,
    codebook: torch.Tensor,
    semantic_centers: torch.Tensor,
    tokenbook_emb: torch.Tensor,
    optimizer: optim.Optimizer,
    tau: float = 1.0,
    tau_prime: float = 0.03,
    top_k: int = 64,
    lambda_token: float = 0.5,
) -> Tuple[float, Optional[float]]:
    """码本级 KL：z=codebook[c]，目标=semantic_centers[c]→token 分布（与推理对齐）。"""
    from models.token_predictor import compute_token_kl_loss

    predictor = model.encoder.token_predictor
    predictor.train()
    optimizer.zero_grad()
    token_loss = compute_token_kl_loss(
        codebook,
        semantic_centers,
        tokenbook_emb,
        predictor,
        tau=tau,
        tau_prime=tau_prime,
        top_k=top_k,
    )
    (lambda_token * token_loss).backward()
    optimizer.step()
    return float(token_loss.item()), float(token_loss.item())


def train_predictor_only_fixed(
    model: Model,
    data: Any,
    feats: torch.Tensor,
    text_emb: torch.Tensor,
    optimizer: optim.Optimizer,
) -> Tuple[float, Optional[float]]:
    """冻结 VQ/encoder，仅对 token_predictor 反传 L_token。"""
    model.eval()
    encoder = model.encoder
    encoder.token_predictor.train()
    optimizer.zero_grad()
    with torch.no_grad():
        h = feats
        g = data
        h = encoder.graph_layer_1(g, h)
        h = encoder.dropout(h)
        if text_emb is not None:
            quantized, _, _, _, _ = encoder.vq(h, text_emb=text_emb)
        else:
            quantized, _, _, _, _ = encoder.vq(h)
    from models.token_predictor import maybe_add_token_loss

    token_loss_weighted, token_loss_raw = maybe_add_token_loss(
        encoder, quantized, text_emb
    )
    if token_loss_raw is None:
        return 0.0, None
    token_loss_weighted.backward()
    optimizer.step()
    return float(token_loss_weighted.item()), float(token_loss_raw.item())


def _freeze_all_but_predictor(model: Model) -> None:
    for param in model.parameters():
        param.requires_grad = False
    predictor = getattr(model.encoder, "token_predictor", None)
    if predictor is not None:
        for param in predictor.parameters():
            param.requires_grad = True


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
        if getattr(model, "hierarchical_vq", False):
            # 无主 CE：用弱 L_L 头 val acc 选模（-loss 会偏好细码塌缩）
            _, logits, loss_t, dist, codebook, _ = model(
                data, feats, text_emb=text_emb, labels=labels
            )
            out = logits.log_softmax(dim=1)
            loss = float(loss_t.item())
            assign = getattr(model.encoder, "_last_assign", {}) or {}
            logits_l = assign.get("logits_l")
            idx_fi = assign.get("idx_fi")
            idx_co = assign.get("idx_co")
            if logits_l is not None:
                out_l = logits_l.log_softmax(dim=1)
                if idx_eval is None:
                    score_l = evaluator(out_l, labels)
                else:
                    score_l = evaluator(out_l[idx_eval], labels[idx_eval])
            else:
                score_l = 0.0
            if idx_eval is None:
                score_probe = evaluator(out, labels)
            else:
                score_probe = evaluator(out[idx_eval], labels[idx_eval])
            uniq_fi = (
                int(torch.unique(idx_fi).numel()) if idx_fi is not None else 0
            )
            uniq_co = (
                int(torch.unique(idx_co).numel()) if idx_co is not None else 0
            )
            model.encoder._last_loss_aux = dict(
                getattr(model.encoder, "_last_loss_aux", {}) or {}
            )
            model.encoder._last_loss_aux["s_L_val"] = float(score_l)
            model.encoder._last_loss_aux["s_probe"] = float(score_probe)
            model.encoder._last_loss_aux["unique_fi_eval"] = float(uniq_fi)
            model.encoder._last_loss_aux["unique_co_eval"] = float(uniq_co)
            model.encoder._last_loss_aux["codebook_loss"] = loss
            # 无主 CE：s_L 先达门槛，再最大化 eval 细码占用
            min_sl = float(getattr(model.encoder, "select_min_s_L", 0.75))
            if float(score_l) < min_sl:
                score = float(score_l)
            else:
                score = 10.0 + float(uniq_fi) + 0.1 * float(score_l)
        else:
            infer_out = model.inference(data, feats, text_emb=text_emb)
            _, logits, _, dist, codebook = infer_out[:5]
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
    labels: Optional[torch.Tensor] = None,
) -> np.ndarray:
    """全图节点结构码 argmax。层次模式返回细码，并写入 encoder._last_assign。"""
    model.eval()
    n = g.num_nodes()
    codes = np.zeros(n, dtype=np.int64)
    if getattr(model, "hierarchical_vq", False):
        if labels is None:
            raise ValueError("assign_node_codes hierarchical requires labels")
        with torch.no_grad():
            infer_out = model.inference(
                g.to(device),
                feats.to(device),
                text_emb=text_emb.to(device),
                labels=labels.to(device),
            )
            dist = infer_out[3]
        if dist.dim() == 3:
            dist = dist.squeeze(0)
        codes = dist.argmax(dim=1).cpu().numpy()
        return codes
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
            infer_out = model.inference(
                loader, feats.to(device), text_emb=text_emb.to(device)
            )
            dist_all = infer_out[3]
        codes = dist_all.argmax(dim=1).cpu().numpy()
    else:
        with torch.no_grad():
            infer_out = model.inference(
                g.to(device), feats.to(device), text_emb=text_emb.to(device)
            )
            dist = infer_out[3]
        if dist.dim() == 3:
            dist = dist.squeeze(0)
        codes = dist.argmax(dim=1).cpu().numpy()
    return codes


def assign_hierarchical_codes(
    model: Model,
    g: Any,
    feats: torch.Tensor,
    text_emb: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """返回 (codes_coarse, codes_fine)。"""
    model.eval()
    with torch.no_grad():
        model.inference(
            g.to(device),
            feats.to(device),
            text_emb=text_emb.to(device),
            labels=labels.to(device),
        )
    last = getattr(model.encoder, "_last_assign", {})
    idx_co = last.get("idx_co")
    idx_fi = last.get("idx_fi")
    if idx_co is None or idx_fi is None:
        raise RuntimeError("Hierarchical assign missing _last_assign indices")
    return idx_co.cpu().numpy(), idx_fi.cpu().numpy()


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

        # 层次模式：写入 conf，供 Model / HierarchicalGCN 使用
        hierarchical = bool(
            conf.get("hierarchical_vq", getattr(self.cfg, "hierarchical_vq", False))
        )
        conf["hierarchical_vq"] = hierarchical
        conf["text_dim"] = int(text_embeddings.shape[1])
        conf["lambda_semantic"] = self.cfg.lambda_semantic
        conf["ema_beta"] = self.cfg.ema_beta
        if hierarchical:
            conf["codebook_size_coarse"] = int(
                conf.get(
                    "codebook_size_coarse",
                    getattr(self.cfg, "codebook_size_coarse", 16),
                )
            )
            conf["codebook_size_fine"] = int(
                conf.get(
                    "codebook_size_fine",
                    getattr(self.cfg, "codebook_size_fine", 256),
                )
            )
            conf["codebook_size"] = conf["codebook_size_fine"]
            conf.setdefault("lambda_H", getattr(self.cfg, "lambda_H", 0.5))
            conf.setdefault("lambda_D", getattr(self.cfg, "lambda_D", 0.05))
            conf.setdefault("lambda_L", getattr(self.cfg, "lambda_L", 0.1))
            conf.setdefault("lambda_div", getattr(self.cfg, "lambda_div", 0.05))
            conf.setdefault("lambda_div_fi", getattr(self.cfg, "lambda_div_fi", 0.5))
            conf.setdefault("lambda_ico", getattr(self.cfg, "lambda_ico", 0.2))
            conf.setdefault(
                "select_min_s_L", getattr(self.cfg, "select_min_s_L", 0.75)
            )
            conf.setdefault("fine_noise", getattr(self.cfg, "fine_noise", 0.0))
            conf.setdefault("text_fuse", getattr(self.cfg, "text_fuse", 0.5))
            self.cfg.hierarchical_vq = True
            self.cfg.codebook_size_coarse = conf["codebook_size_coarse"]
            self.cfg.codebook_size_fine = conf["codebook_size_fine"]
            self.cfg.codebook_size = conf["codebook_size_fine"]
            self.cfg.lambda_H = float(conf["lambda_H"])
            self.cfg.lambda_D = float(conf["lambda_D"])
            self.cfg.lambda_L = float(conf["lambda_L"])
            self.cfg.lambda_div = float(conf["lambda_div"])
            self.cfg.lambda_div_fi = float(conf["lambda_div_fi"])
            self.cfg.lambda_ico = float(conf["lambda_ico"])
            self.cfg.select_min_s_L = float(conf["select_min_s_L"])
            self.cfg.fine_noise = float(conf["fine_noise"])
            self.cfg.text_fuse = float(conf["text_fuse"])
            log.info(
                "Hierarchical VQ: M_co=%d M_fi=%d lambda_H=%.3f lambda_D=%.3f "
                "lambda_L=%.3f lambda_div=%.3f lambda_div_fi=%.3f lambda_ico=%.3f "
                "text_fuse=%.3f select_min_s_L=%.2f (no main CE)",
                conf["codebook_size_coarse"],
                conf["codebook_size_fine"],
                conf["lambda_H"],
                conf["lambda_D"],
                conf["lambda_L"],
                conf["lambda_div"],
                conf["lambda_div_fi"],
                conf["lambda_ico"],
                conf["text_fuse"],
                conf["select_min_s_L"],
            )

        model = self.build_model(conf)
        self.wrap_semantic_vq(text_embeddings.shape[1])

        use_token_pred = (
            self.cfg.enable_token_predictor and self.cfg.lambda_token > 0
        )
        if use_token_pred:
            from text_tokenizers.text_tokenbook import TextTokenbook

            tokenbook_dir = Path(
                conf.get("tokenbook_dir")
                or self.cfg.tokenbook_path
                or "./codebook"
            )
            vocab_path = tokenbook_dir / self.cfg.tokenbook_vocab_filename
            if not vocab_path.exists():
                raise FileNotFoundError(
                    f"Tokenbook vocabulary required for token predictor: {vocab_path}"
                )
            tokenbook = TextTokenbook.load(
                tokenbook_dir,
                cfg=self.cfg,
                model_name=self.cfg.sentence_bert_model,
                device=device,
                build_embeddings=True,
            )
            model.attach_token_predictor(
                len(tokenbook),
                tokenbook.get_embedding_matrix(),
                self.cfg,
            )
            log.info(
                "Token predictor attached: V=%d, type=%s, top_k=%d, lambda_token=%.4f, "
                "tau=%.2f, tau_prime=%.2f",
                len(tokenbook),
                getattr(self.cfg, "token_predictor_type", "linear"),
                getattr(self.cfg, "token_kl_top_k", 0),
                self.cfg.lambda_token,
                self.cfg.token_pred_temperature,
                self.cfg.token_target_temperature,
            )

        load_ckpt = conf.get("load_checkpoint")
        if load_ckpt:
            ckpt_path = Path(load_ckpt)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"load_checkpoint not found: {ckpt_path}")
            state = torch.load(ckpt_path, map_location=device, weights_only=False)
            missing, unexpected = model.load_state_dict(state, strict=False)
            log.info(
                "Loaded checkpoint %s (missing=%d, unexpected=%d)",
                ckpt_path,
                len(missing),
                len(unexpected),
            )

        predictor_only = bool(conf.get("predictor_only", False))
        if predictor_only:
            return self._fit_predictor_only(
                model=model,
                g=g,
                feats=feats,
                labels=labels,
                text_embeddings=text_embeddings,
                idx_train=idx_train,
                idx_val=idx_val,
                idx_test=idx_test,
                conf=conf,
                output_dir=output_dir,
                log=log,
            )

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

        stable_min_epoch = conf.get("tfidf_stats_min_epoch")
        if stable_min_epoch is None:
            stable_min_epoch = getattr(self.cfg, "tfidf_stats_min_epoch", None)
        if stable_min_epoch is None:
            stable_min_epoch = warmup_epochs + 1
        stable_min_epoch = int(stable_min_epoch)
        self.cfg.tfidf_stats_min_epoch = stable_min_epoch
        log.info(
            "Best checkpoint & TF-IDF stats use epochs >= %d (warmup_epochs=%d)",
            stable_min_epoch,
            warmup_epochs,
        )

        best_epoch, best_score_val, count = 0, 0.0, 0
        state = copy.deepcopy(model.state_dict())
        latest_state = state
        last_epoch = 0

        for epoch in range(1, max_epoch + 1):
            last_epoch = epoch
            self.current_epoch = epoch
            in_warmup = epoch <= warmup_epochs
            lam_eff = 0.0 if in_warmup else self.cfg.lambda_semantic
            assert self.svq is not None
            self.svq.set_lambda_effective(lam_eff)
            # L_H warmup：与语义 λ 同步线性升到目标
            if hierarchical and hasattr(model.encoder, "set_lambda_H_effective"):
                target_h = float(conf.get("lambda_H", self.cfg.lambda_H))
                if in_warmup and warmup_epochs > 0:
                    h_eff = target_h * (epoch / float(warmup_epochs))
                else:
                    h_eff = target_h
                model.encoder.set_lambda_H_effective(h_eff)

            if "SAGE" in model.model_name:
                loss, token_loss = train_sage_semantic(
                    model,
                    data_train,
                    feats,
                    labels,
                    text_embeddings,
                    criterion,
                    optimizer,
                )
            else:
                loss, token_loss = train_semantic_fixed(
                    model,
                    data_train,
                    feats,
                    labels,
                    text_embeddings,
                    criterion,
                    optimizer,
                    idx_train,
                )
            train_aux = dict(getattr(model.encoder, "_last_loss_aux", {}) or {})

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
                aux = train_aux
                aux_str = ""
                eval_aux = dict(getattr(model.encoder, "_last_loss_aux", {}) or {})
                if hierarchical and aux:
                    aux_str = (
                        " | L_co=%.4f L_fi=%.4f edge=%.4f L_H=%.4f L_ico=%.4f "
                        "uniq_co=%d/%d uniq_fi=%d/%d | s_L=%.4f probe=%.4f"
                        % (
                            aux.get("commit_co", 0.0),
                            aux.get("commit_fi", 0.0),
                            aux.get("edge_rec", 0.0),
                            aux.get("L_H", 0.0),
                            aux.get("L_ico", 0.0),
                            int(aux.get("unique_co", 0)),
                            int(eval_aux.get("unique_co_eval", 0)),
                            int(aux.get("unique_fi", 0)),
                            int(eval_aux.get("unique_fi_eval", 0)),
                            float(eval_aux.get("s_L_val", 0.0)),
                            float(eval_aux.get("s_probe", 0.0)),
                        )
                    )
                if token_loss is not None:
                    log.info(
                        "Ep %3d | warmup=%s | lambda_eff=%.4f | loss=%.4f | "
                        "L_token=%.4f | s_val=%.4f | s_test=%.4f%s",
                        epoch,
                        in_warmup,
                        lam_eff,
                        loss,
                        token_loss,
                        score_val if not hierarchical else float(eval_aux.get("s_L_val", 0.0)),
                        score_test if not hierarchical else float(eval_aux.get("s_probe", 0.0)),
                        aux_str,
                    )
                else:
                    log.info(
                        "Ep %3d | warmup=%s | lambda_eff=%.4f | loss=%.4f | "
                        "s_val=%.4f | s_test=%.4f%s",
                        epoch,
                        in_warmup,
                        lam_eff,
                        loss,
                        score_val if not hierarchical else float(eval_aux.get("s_L_val", 0.0)),
                        score_test if not hierarchical else float(eval_aux.get("s_probe", 0.0)),
                        aux_str,
                    )
                latest_state = copy.deepcopy(model.state_dict())
                if epoch >= stable_min_epoch:
                    if best_epoch < stable_min_epoch or score_val >= best_score_val:
                        best_epoch = epoch
                        best_score_val = score_val
                        state = copy.deepcopy(latest_state)
                        count = 0
                    else:
                        count += 1
                else:
                    log.debug(
                        "Ep %d before stable_min_epoch=%d; skip best/patience",
                        epoch,
                        stable_min_epoch,
                    )

            if epoch >= stable_min_epoch and count >= patience:
                break
            if epoch == max_epoch:
                break

        if best_epoch >= stable_min_epoch:
            model.load_state_dict(state)
            if hierarchical:
                log.info(
                    "Best epoch %d (>= stable min %d), select_score=%.4f "
                    "(no main CE; gate s_L then uniq_fi)",
                    best_epoch,
                    stable_min_epoch,
                    best_score_val,
                )
            else:
                log.info(
                    "Best epoch %d (>= stable min %d), val acc %.4f",
                    best_epoch,
                    stable_min_epoch,
                    best_score_val,
                )
        else:
            model.load_state_dict(latest_state)
            best_epoch = last_epoch
            log.warning(
                "No eval after stable_min_epoch=%d; using final epoch %d weights for "
                "model + TF-IDF node codes",
                stable_min_epoch,
                last_epoch,
            )
            state = copy.deepcopy(latest_state)

        if hierarchical:
            codes_co, codes_fi = assign_hierarchical_codes(
                model, g, feats, text_embeddings, labels, device
            )
            node_codes = codes_fi
            assert self.svq is not None
            cb_co = model.encoder.vq_co.codebook.detach().cpu()
            cb_fi = model.encoder.vq_fi.codebook.detach().cpu()
            artifacts = CodebookArtifacts(
                encoder_state_dict=state,
                codebook_embeddings=cb_fi,
                semantic_centers=self.svq.semantic_centers.detach().cpu(),
                node_code_assignments=node_codes,
                save_dir=output_dir,
                codebook_coarse=cb_co,
                codebook_fine=cb_fi,
                node_codes_coarse=codes_co,
                node_codes_fine=codes_fi,
                hierarchical_vq=True,
            )
            n_co = len(np.unique(codes_co))
            n_fi = len(np.unique(codes_fi))
            log.info(
                "Hierarchical assign: unique coarse=%d/%d fine=%d/%d",
                n_co,
                conf["codebook_size_coarse"],
                n_fi,
                conf["codebook_size_fine"],
            )
        else:
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

    def _fit_predictor_only(
        self,
        model: Model,
        g: Any,
        feats: torch.Tensor,
        labels: torch.Tensor,
        text_embeddings: torch.Tensor,
        idx_train: torch.Tensor,
        idx_val: torch.Tensor,
        idx_test: torch.Tensor,
        conf: Dict[str, Any],
        output_dir: Path,
        log: logging.Logger,
    ) -> CodebookArtifacts:
        """E5：冻结 VQ/encoder，仅训练 token_predictor；复用 init_dir 的结构码与码本。"""
        device = conf["device"]
        init_dir = Path(conf.get("init_from_dir") or "")
        if not (init_dir / "codebook_embeddings.npz").exists():
            load_path = conf.get("load_checkpoint")
            if not load_path:
                raise ValueError("predictor_only needs init_from_dir or load_checkpoint")
            init_dir = Path(load_path).parent
        old = self.load_artifacts(init_dir, device)
        log.info("Predictor-only mode: frozen weights from %s", init_dir)

        _freeze_all_but_predictor(model)
        pred_lr = conf.get("predictor_lr", 1e-3)
        optimizer = optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=pred_lr,
            weight_decay=0.0,
        )
        max_epoch = int(
            conf.get("predictor_only_epochs") or self.cfg.predictor_only_epochs
        )
        use_code_level = bool(conf.get("predictor_code_level", True))
        tokenbook_emb = getattr(model.encoder, "tokenbook_embeddings", None)
        codebook = self.svq.codebook.detach()
        sem_centers = self.svq.semantic_centers.detach()
        g = g.to(device)
        best_token_loss = float("inf")
        best_state = copy.deepcopy(model.state_dict())

        for epoch in range(1, max_epoch + 1):
            self.svq.set_lambda_effective(self.cfg.lambda_semantic)
            if use_code_level and tokenbook_emb is not None:
                loss, token_loss = train_predictor_code_level(
                    model,
                    codebook,
                    sem_centers,
                    tokenbook_emb,
                    optimizer,
                    tau=self.cfg.token_pred_temperature,
                    tau_prime=self.cfg.token_target_temperature,
                    top_k=getattr(self.cfg, "token_kl_top_k", 64),
                    lambda_token=self.cfg.lambda_token,
                )
            else:
                loss, token_loss = train_predictor_only_fixed(
                    model, g, feats, text_embeddings, optimizer
                )
            if token_loss is not None and token_loss < best_token_loss:
                best_token_loss = token_loss
                best_state = copy.deepcopy(model.state_dict())
            if epoch % max(1, max_epoch // 5) == 0 or epoch == max_epoch:
                log.info(
                    "Predictor-only Ep %3d/%d | L_token=%.4f | best=%.4f",
                    epoch,
                    max_epoch,
                    token_loss or 0.0,
                    best_token_loss,
                )

        model.load_state_dict(best_state)
        artifacts = CodebookArtifacts(
            encoder_state_dict=best_state,
            codebook_embeddings=old.codebook_embeddings,
            semantic_centers=old.semantic_centers,
            node_code_assignments=old.node_code_assignments,
            save_dir=output_dir,
        )
        self.save_artifacts(artifacts, output_dir, train_conf=conf)
        log.info(
            "Predictor-only done. best L_token=%.4f. Artifacts: %s",
            best_token_loss,
            output_dir,
        )
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
        if artifacts.hierarchical_vq:
            if artifacts.codebook_coarse is not None:
                np.savez(
                    save_dir / "codebook_coarse.npz",
                    artifacts.codebook_coarse.numpy(),
                )
            if artifacts.codebook_fine is not None:
                np.savez(
                    save_dir / "codebook_fine.npz",
                    artifacts.codebook_fine.numpy(),
                )
            if artifacts.node_codes_coarse is not None:
                np.savez(
                    save_dir / "node_codes_coarse.npz",
                    node_codes=artifacts.node_codes_coarse,
                )
            if artifacts.node_codes_fine is not None:
                np.savez(
                    save_dir / "node_codes_fine.npz",
                    node_codes=artifacts.node_codes_fine,
                )
        cfg_snap = {
            "lambda_semantic": self.cfg.lambda_semantic,
            "warmup_epochs": self.cfg.warmup_epochs,
            "tfidf_stats_min_epoch": getattr(self.cfg, "tfidf_stats_min_epoch", None),
            "ema_beta": self.cfg.ema_beta,
            "codebook_size": self.cfg.codebook_size,
            "enable_token_predictor": self.cfg.enable_token_predictor,
            "lambda_token": self.cfg.lambda_token,
            "token_pred_temperature": self.cfg.token_pred_temperature,
            "token_target_temperature": self.cfg.token_target_temperature,
            "token_kl_top_k": getattr(self.cfg, "token_kl_top_k", 0),
            "token_predictor_type": getattr(self.cfg, "token_predictor_type", "linear"),
            "p_code_normalize": getattr(self.cfg, "p_code_normalize", "none"),
            "lambda_pred": self.cfg.lambda_pred,
            "text_vocab_size": self.cfg.text_vocab_size,
            "hierarchical_vq": bool(artifacts.hierarchical_vq),
            "codebook_size_coarse": getattr(self.cfg, "codebook_size_coarse", 16),
            "codebook_size_fine": getattr(self.cfg, "codebook_size_fine", 256),
            "lambda_H": getattr(self.cfg, "lambda_H", 0.5),
            "lambda_D": getattr(self.cfg, "lambda_D", 0.05),
            "lambda_L": getattr(self.cfg, "lambda_L", 0.05),
        }
        if artifacts.hierarchical_vq and artifacts.codebook_fine is not None:
            cfg_snap["codebook_size"] = int(artifacts.codebook_fine.shape[0])
            if artifacts.codebook_coarse is not None:
                cfg_snap["codebook_size_coarse"] = int(
                    artifacts.codebook_coarse.shape[0]
                )
            cfg_snap["codebook_size_fine"] = int(artifacts.codebook_fine.shape[0])
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
        hierarchical = bool(getattr(args, "hierarchical_vq", False))
        conf["hierarchical_vq"] = hierarchical
        if hierarchical:
            conf["codebook_size_coarse"] = int(
                getattr(args, "codebook_size_coarse", None)
                or conf.get("codebook_size_coarse", 16)
            )
            conf["codebook_size_fine"] = int(
                getattr(args, "codebook_size_fine", None)
                or conf.get("codebook_size_fine", 128)
            )
            conf["codebook_size"] = conf["codebook_size_fine"]
            if getattr(args, "lambda_H", None) is not None:
                conf["lambda_H"] = float(args.lambda_H)
            if getattr(args, "lambda_D", None) is not None:
                conf["lambda_D"] = float(args.lambda_D)
            if getattr(args, "lambda_L", None) is not None:
                conf["lambda_L"] = float(args.lambda_L)
            if getattr(args, "lambda_div", None) is not None:
                conf["lambda_div"] = float(args.lambda_div)
            if getattr(args, "lambda_div_fi", None) is not None:
                conf["lambda_div_fi"] = float(args.lambda_div_fi)
            if getattr(args, "lambda_ico", None) is not None:
                conf["lambda_ico"] = float(args.lambda_ico)
            if getattr(args, "select_min_s_L", None) is not None:
                conf["select_min_s_L"] = float(args.select_min_s_L)
            if getattr(args, "text_fuse", None) is not None:
                conf["text_fuse"] = float(args.text_fuse)
            # 层次模式默认加强边监督
            if getattr(args, "lamb_edge", None) is not None:
                conf["lamb_edge"] = float(args.lamb_edge)
            else:
                conf.setdefault("lamb_edge", 0.15)
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
            "lambda_H": 1.0,
            "lambda_D": 0.05,
            "lambda_L": 0.1,
            "lambda_div": 0.05,
            "lambda_div_fi": 0.5,
            "lambda_ico": 0.2,
            "select_min_s_L": 0.75,
            "fine_noise": 0.0,
            "text_fuse": 0.5,
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
        tokenbook: TextTokenbook,
    ) -> TFIDFStatistics:
        save_path = Path(save_path)
        num_codes = artifacts.codebook_embeddings.shape[0]
        token_to_id = tokenbook.token_to_id
        vocab_size = len(token_to_id)
        if vocab_size == 0:
            raise ValueError("Tokenbook vocabulary is empty.")

        all_ids = texts_to_tokenbook_ids(
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
            "Saved TF-IDF (tokenbook vocab) to %s shape (%d, %d)",
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
