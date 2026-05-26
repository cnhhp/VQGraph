"""
文本 tokenbook 加载与嵌入缓存。

从预置 ``codebook/filtered_tokenbook.npy`` 加载词表，并用 Sentence-BERT 编码 / 缓存嵌入矩阵。
供模块3 结构引导文本 token 筛选使用。
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from config import Config, get_config
from graph_utils import extract_sentence_bert_embeddings, tokenize_for_tfidf

logger = logging.getLogger(__name__)


def _resolve_tokenbook_dir(
    path: Union[str, Path],
    cfg: Config,
) -> tuple[Path, Path]:
    """解析 tokenbook 目录与 vocab 文件路径。"""
    path = Path(path)
    if path.suffix == ".npy":
        tokenbook_dir = path.parent
        vocab_path = path
    else:
        tokenbook_dir = path
        vocab_path = tokenbook_dir / cfg.tokenbook_vocab_filename
    return tokenbook_dir, vocab_path


class TextTokenbook:
    """token 词表 + 嵌入矩阵 [V, D]。"""

    def __init__(self) -> None:
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.embeddings: Optional[torch.Tensor] = None
        self.model_name: Optional[str] = None
        self.vocab_path: Optional[Path] = None
        self.tokenbook_dir: Optional[Path] = None

    def __len__(self) -> int:
        return len(self.token_to_id)

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        cfg: Optional[Config] = None,
        model_name: Optional[str] = None,
        device: Optional[torch.device] = None,
        build_embeddings: bool = True,
    ) -> "TextTokenbook":
        """
        从目录或 ``.npy`` 词表文件加载 tokenbook。

        目录内默认文件：
        - ``filtered_tokenbook.npy``：词表
        - ``token_embeddings.npz``：可选 SBERT 嵌入缓存
        - ``tokenbook_meta.json``：缓存元数据
        """
        cfg = cfg or get_config()
        model_name = model_name or cfg.sentence_bert_model
        tokenbook_dir, vocab_path = _resolve_tokenbook_dir(path, cfg)

        if not vocab_path.exists():
            raise FileNotFoundError(f"Tokenbook vocabulary not found: {vocab_path}")

        tokens = np.load(vocab_path, allow_pickle=True)
        token_list = [str(t) for t in tokens.tolist()]
        tb = cls._from_token_list(token_list)
        tb.vocab_path = vocab_path
        tb.tokenbook_dir = tokenbook_dir
        tb.model_name = model_name

        if len(tb) != cfg.text_vocab_size:
            logger.warning(
                "Tokenbook size %d != config.text_vocab_size %d",
                len(tb),
                cfg.text_vocab_size,
            )

        if build_embeddings:
            TextTokenbookBuilder(cfg).ensure_embeddings(
                tb,
                model_name=model_name,
                device=device,
                tokenbook_dir=tokenbook_dir,
            )
        return tb

    @classmethod
    def _from_token_list(cls, token_list: List[str]) -> "TextTokenbook":
        tb = cls()
        tb.token_to_id = {t: i for i, t in enumerate(token_list)}
        tb.id_to_token = {i: t for i, t in enumerate(token_list)}
        return tb

    def save(self, path: Union[str, Path]) -> None:
        """保存词表、嵌入缓存与元数据到目录。"""
        if self.embeddings is None:
            raise RuntimeError("Cannot save tokenbook without embeddings.")
        if self.model_name is None:
            raise RuntimeError("Cannot save tokenbook without model_name.")

        save_dir = Path(path)
        if save_dir.suffix == ".npy":
            save_dir = save_dir.parent
        save_dir.mkdir(parents=True, exist_ok=True)

        vocab_path = save_dir / get_config().tokenbook_vocab_filename
        tokens = np.array([self.id_to_token[i] for i in range(len(self))], dtype=object)
        np.save(vocab_path, tokens, allow_pickle=True)

        emb_path = save_dir / get_config().tokenbook_embeddings_filename
        np.savez(
            emb_path,
            embeddings=self.embeddings.detach().cpu().numpy(),
        )

        meta_path = save_dir / get_config().tokenbook_meta_filename
        meta = {
            "model_name": self.model_name,
            "vocab_size": len(self),
            "embedding_dim": int(self.embeddings.shape[1]),
            "vocab_filename": get_config().tokenbook_vocab_filename,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        logger.info("Saved tokenbook to %s (V=%d, D=%d)", save_dir, len(self), meta["embedding_dim"])

    def get_embedding_matrix(self) -> torch.Tensor:
        if self.embeddings is None:
            raise RuntimeError("Tokenbook embeddings not loaded.")
        return self.embeddings


class TextTokenbookBuilder:
    """嵌入缓存 / 重建工具（词表来自预置 ``codebook/``）。"""

    def __init__(self, cfg: Optional[Config] = None) -> None:
        self.cfg = cfg or get_config()

    def build_from_corpus(
        self,
        texts: List[str],
        vocab_size: Optional[int] = None,
        embedding_model: Optional[str] = None,
    ) -> TextTokenbook:
        raise NotImplementedError(
            "TextTokenbook 使用预置 codebook/filtered_tokenbook.npy；"
            "请调用 TextTokenbook.load('./codebook') 而非从语料构建词表。"
        )

    def tokenize_text(self, text: str, tokenbook: TextTokenbook) -> List[str]:
        """词级分词后，仅保留 tokenbook 词表中存在的 token。"""
        return [t for t in tokenize_for_tfidf(text) if t in tokenbook.token_to_id]

    def ensure_embeddings(
        self,
        tokenbook: TextTokenbook,
        model_name: Optional[str] = None,
        device: Optional[torch.device] = None,
        tokenbook_dir: Optional[Path] = None,
        force_rebuild: bool = False,
    ) -> None:
        """加载或构建 tokenbook 嵌入矩阵并写入本地 cache。"""
        model_name = model_name or tokenbook.model_name or self.cfg.sentence_bert_model
        tokenbook.model_name = model_name

        if tokenbook_dir is None:
            tokenbook_dir = tokenbook.tokenbook_dir
        if tokenbook_dir is None:
            raise ValueError("tokenbook_dir is required to load or save embeddings cache.")

        emb_path = tokenbook_dir / self.cfg.tokenbook_embeddings_filename
        meta_path = tokenbook_dir / self.cfg.tokenbook_meta_filename
        vocab_size = len(tokenbook)

        if not force_rebuild and emb_path.exists():
            data = np.load(emb_path)
            emb = torch.tensor(data["embeddings"], dtype=torch.float32)
            if emb.shape[0] == vocab_size:
                meta_ok = True
                if meta_path.exists():
                    with open(meta_path, encoding="utf-8") as f:
                        meta = json.load(f)
                    meta_ok = (
                        meta.get("model_name") == model_name
                        and meta.get("vocab_size") == vocab_size
                        and meta.get("embedding_dim") == emb.shape[1]
                        and meta.get("vocab_filename")
                        == self.cfg.tokenbook_vocab_filename
                    )
                    if not meta_ok:
                        logger.warning(
                            "Tokenbook embedding cache meta mismatch; rebuilding embeddings."
                        )
                if meta_ok:
                    tokenbook.embeddings = emb
                    logger.info(
                        "Loaded tokenbook embeddings from cache: %s shape %s",
                        emb_path,
                        tuple(emb.shape),
                    )
                    return

        logger.info(
            "Building tokenbook embeddings with %s for %d tokens...",
            model_name,
            vocab_size,
        )
        word_dict = {i: tokenbook.id_to_token[i] for i in range(vocab_size)}
        tokenbook.embeddings = extract_sentence_bert_embeddings(
            word_dict,
            model_name=model_name,
            device=device,
        ).float()

        tokenbook.tokenbook_dir = tokenbook_dir
        tokenbook.save(tokenbook_dir)


def load_tokenbook(
    path: Union[str, Path],
    cfg: Optional[Config] = None,
    model_name: Optional[str] = None,
    device: Optional[torch.device] = None,
    build_embeddings: bool = True,
) -> TextTokenbook:
    """便捷函数：加载预置 tokenbook。"""
    return TextTokenbook.load(
        path,
        cfg=cfg,
        model_name=model_name,
        device=device,
        build_embeddings=build_embeddings,
    )


def _run_smoke_test(args: argparse.Namespace) -> None:
    from graph_utils import setup_logging

    setup_logging(__name__)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.device >= 0 else "cpu"
    )

    tb = TextTokenbook.load(
        args.tokenbook_dir,
        model_name=args.sentence_bert,
        device=device,
        build_embeddings=not args.no_build_embeddings,
    )
    emb = tb.get_embedding_matrix()
    logger.info("Tokenbook loaded: V=%d, embeddings=%s, model=%s", len(tb), tuple(emb.shape), tb.model_name)

    for i in range(min(3, len(tb))):
        token = tb.id_to_token[i]
        norm = float(torch.norm(emb[i]).item())
        logger.info("  token[%d]=%r norm=%.4f", i, token, norm)

    if args.check_cache_reload and not args.no_build_embeddings:
        tb2 = TextTokenbook.load(
            args.tokenbook_dir,
            model_name=args.sentence_bert,
            device=device,
            build_embeddings=True,
        )
        assert torch.allclose(tb2.embeddings, tb.embeddings), "cache reload mismatch"
        logger.info("Cache reload check: OK")


def _parse_main_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TextTokenbook 加载冒烟测试")
    p.add_argument("--tokenbook_dir", type=str, default="./codebook")
    p.add_argument("--sentence_bert", type=str, default="all-MiniLM-L6-v2")
    p.add_argument("--device", type=int, default=0, help="-1 for CPU")
    p.add_argument("--no_build_embeddings", action="store_true")
    p.add_argument("--check_cache_reload", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    _run_smoke_test(_parse_main_args())
