"""
结构码本 PCA 投影器：256-d codebook -> k-d 数值向量，供 LLM 序列化使用。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


def minmax_per_dim(values: np.ndarray) -> np.ndarray:
    """对 [M, k] 矩阵按列 min-max 归一化到 [0, 1]。"""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D array, got shape {arr.shape}")
    vmin = arr.min(axis=0, keepdims=True)
    vmax = arr.max(axis=0, keepdims=True)
    span = vmax - vmin
    span = np.where(span < 1e-12, 1.0, span)
    return (arr - vmin) / span


def format_vector_values(values: np.ndarray, decimals: int = 4) -> str:
    """将 k 维向量格式化为 '[0.8234, 0.6123, ...]'。"""
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    parts = [f"{float(v):.{decimals}f}" for v in arr]
    return "[" + ", ".join(parts) + "]"


@dataclass
class StructuralProjector:
    """PCA 结构投影器：codebook[c] -> k 维归一化向量。"""

    projections: np.ndarray  # [M, k]，已 min-max 归一化
    components: np.ndarray  # [k, d_in]
    mean: np.ndarray  # [d_in]
    k: int
    d_in: int
    method: str = "pca"
    normalize: str = "minmax_per_dim"
    decimals: int = 4
    source_codebook_dir: Optional[str] = None
    explained_variance_ratio: Optional[np.ndarray] = None

    @classmethod
    def fit_pca(
        cls,
        codebook: np.ndarray,
        k: int,
        *,
        source_codebook_dir: Optional[Union[str, Path]] = None,
        decimals: int = 4,
    ) -> "StructuralProjector":
        """
        对 codebook [M, d] 拟合 PCA 并生成 [M, k] 投影表。

        Parameters
        ----------
        codebook : np.ndarray
            结构码本嵌入，shape [M, d_in]。
        k : int
            目标维度（通常等于 top_k_text_tokens）。
        """
        cb = np.asarray(codebook, dtype=np.float64)
        if cb.ndim != 2:
            raise ValueError(f"codebook must be 2-D, got shape {cb.shape}")
        m, d_in = cb.shape
        k = int(k)
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if k > min(m, d_in):
            raise ValueError(f"k={k} exceeds min(M={m}, d_in={d_in})")

        pca = PCA(n_components=k, random_state=42)
        raw = pca.fit_transform(cb)
        projections = minmax_per_dim(raw)

        src = str(source_codebook_dir) if source_codebook_dir is not None else None
        return cls(
            projections=projections.astype(np.float32),
            components=pca.components_.astype(np.float32),
            mean=pca.mean_.astype(np.float32),
            k=k,
            d_in=d_in,
            decimals=decimals,
            source_codebook_dir=src,
            explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float64),
        )

    def project(self, code_idx: int) -> np.ndarray:
        """返回 code_idx 对应的 k 维向量（已归一化）。"""
        idx = int(code_idx)
        if idx < 0 or idx >= self.projections.shape[0]:
            raise IndexError(
                f"code_idx {idx} out of range [0, {self.projections.shape[0]})"
            )
        return self.projections[idx].astype(np.float64)

    def format_vector(self, code_idx: int) -> str:
        """格式化为 LLM 可读字符串 '[0.8234, ...]'。"""
        return format_vector_values(self.project(code_idx), decimals=self.decimals)

    def cumulative_explained_variance(self) -> float:
        if self.explained_variance_ratio is None:
            return 0.0
        return float(self.explained_variance_ratio.sum())

    def save(self, output_dir: Union[str, Path]) -> Path:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        np.savez(
            out / "projected_codebook.npz",
            projections=self.projections,
        )
        np.savez(
            out / "pca_model.npz",
            components=self.components,
            mean=self.mean,
        )

        manifest = {
            "method": self.method,
            "k": self.k,
            "d_in": self.d_in,
            "M": int(self.projections.shape[0]),
            "normalize": self.normalize,
            "decimals": self.decimals,
            "source_codebook_dir": self.source_codebook_dir,
            "explained_variance_ratio": (
                self.explained_variance_ratio.tolist()
                if self.explained_variance_ratio is not None
                else None
            ),
            "cumulative_explained_variance": self.cumulative_explained_variance(),
        }
        with open(out / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        logger.info(
            "Saved projected codebook to %s (M=%d, k=%d, cum_var=%.4f)",
            out,
            self.projections.shape[0],
            self.k,
            self.cumulative_explained_variance(),
        )
        return out

    @classmethod
    def load(cls, load_dir: Union[str, Path]) -> "StructuralProjector":
        root = Path(load_dir)
        proj = np.load(root / "projected_codebook.npz")
        pca = np.load(root / "pca_model.npz")
        manifest_path = root / "manifest.json"

        decimals = 4
        source = None
        method = "pca"
        normalize = "minmax_per_dim"
        explained = None
        if manifest_path.exists():
            with open(manifest_path, encoding="utf-8") as f:
                meta = json.load(f)
            decimals = int(meta.get("decimals", 4))
            source = meta.get("source_codebook_dir")
            method = meta.get("method", "pca")
            normalize = meta.get("normalize", "minmax_per_dim")
            evr = meta.get("explained_variance_ratio")
            if evr is not None:
                explained = np.asarray(evr, dtype=np.float64)

        projections = proj["projections"]
        components = pca["components"]
        mean = pca["mean"]
        k = int(projections.shape[1])
        d_in = int(components.shape[1])

        return cls(
            projections=projections,
            components=components,
            mean=mean,
            k=k,
            d_in=d_in,
            method=method,
            normalize=normalize,
            decimals=decimals,
            source_codebook_dir=source,
            explained_variance_ratio=explained,
        )
