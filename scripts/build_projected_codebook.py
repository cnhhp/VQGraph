"""
离线构建 PCA 结构投影码本。

用法::
  python scripts/build_projected_codebook.py \\
    --codebook_dir ./outputs/experiments/e5b_no_ltoken/cora/GCN/seed_1 \\
    --output_dir ./outputs/experiments/e5b_no_ltoken/cora/GCN/seed_1/projected_k8 \\
    --k 8
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from config import get_config, reset_config
from graph_utils import setup_logging
from models.codebook_trainer import CodebookTrainer
from models.struct_projector import StructuralProjector

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="构建 PCA 结构投影码本")
    p.add_argument(
        "--codebook_dir",
        type=str,
        required=True,
        help="E1/E5b 码本产物目录（含 codebook_embeddings.npz）",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="投影码本输出目录",
    )
    p.add_argument(
        "--k",
        type=int,
        default=None,
        help="投影维度，默认 config.top_k_text_tokens",
    )
    p.add_argument(
        "--decimals",
        type=int,
        default=None,
        help="向量小数位数，默认 config.struct_vector_decimals",
    )
    p.add_argument("--device", type=int, default=-1, help="-1 表示 CPU")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    reset_config()
    cfg = get_config()

    k = args.k if args.k is not None else cfg.top_k_text_tokens
    decimals = (
        args.decimals
        if args.decimals is not None
        else getattr(cfg, "struct_vector_decimals", 4)
    )

    codebook_dir = Path(args.codebook_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(__name__, log_file=output_dir / "build_projected_codebook.log")

    device = torch.device(
        "cpu"
        if args.device < 0
        else (f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    )

    artifacts = CodebookTrainer.load_artifacts(codebook_dir, device)
    codebook = artifacts.codebook_embeddings.cpu().numpy()
    logger.info(
        "Loaded codebook from %s: shape=%s",
        codebook_dir,
        codebook.shape,
    )

    projector = StructuralProjector.fit_pca(
        codebook,
        k=k,
        source_codebook_dir=str(codebook_dir.resolve()),
        decimals=decimals,
    )
    projector.save(output_dir)

    if projector.explained_variance_ratio is not None:
        per_dim = projector.explained_variance_ratio
        logger.info(
            "PCA explained variance (per dim): %s",
            ", ".join(f"{v:.4f}" for v in per_dim),
        )
        logger.info(
            "Cumulative explained variance (k=%d): %.4f",
            k,
            projector.cumulative_explained_variance(),
        )

    print(f"Saved projected codebook to {output_dir}")
    print(f"  M={projector.projections.shape[0]}, k={k}, d_in={projector.d_in}")
    print(f"  cumulative explained variance: {projector.cumulative_explained_variance():.4f}")


if __name__ == "__main__":
    main()
