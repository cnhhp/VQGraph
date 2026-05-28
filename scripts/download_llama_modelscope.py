"""
魔搭下载 Meta-Llama-3-8B-Instruct（解决 .lock 冲突 + 跳过无用大文件）。

用法:
  python scripts/download_llama_modelscope.py
  python scripts/download_llama_modelscope.py --local_dir D:/models/Meta-Llama-3-8B-Instruct
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

MODEL_ID = "LLM-Research/Meta-Llama-3-8B-Instruct"
DEFAULT_LOCAL_DIR = Path(r"D:\models\Meta-Llama-3-8B-Instruct")


def clear_modelscope_locks(model_id: str) -> None:
    """删除魔搭 hub 锁目录，避免多进程/中断后一直 Waiting to acquire lock。"""
    safe_name = model_id.replace("/", "___")
    cache_root = Path.home() / ".cache" / "modelscope" / "hub"
    lock_dir = cache_root / ".lock"
    lock_file = lock_dir / safe_name

    if lock_dir.exists():
        try:
            if lock_file.exists():
                lock_file.unlink(missing_ok=True)
                logger.info("Removed lock file: %s", lock_file)
            # 若 .lock 为空可保留目录
        except OSError as exc:
            logger.warning("Could not remove lock %s: %s", lock_file, exc)
            logger.warning(
                "仍有 Python 进程占用魔搭下载，请先结束: "
                "Get-Process python | Stop-Process -Force"
            )
            raise


def count_shards(local_dir: Path) -> int:
    return len(list(local_dir.glob("model-*-of-00004.safetensors")))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Llama-3-8B via ModelScope")
    parser.add_argument("--local_dir", type=str, default=str(DEFAULT_LOCAL_DIR))
    parser.add_argument(
        "--enable_file_lock",
        action="store_true",
        help="启用魔搭文件锁（默认关闭，避免锁冲突）",
    )
    parser.add_argument(
        "--kill_python",
        action="store_true",
        help="下载前尝试结束本机所有 python.exe（慎用）",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    if args.kill_python:
        logger.warning("--kill_python 需手动结束占用锁的进程，本脚本不自动杀进程")

    clear_modelscope_locks(MODEL_ID)

    from modelscope import snapshot_download

    logger.info("Downloading %s -> %s", MODEL_ID, local_dir)
    logger.info("enable_file_lock=%s", args.enable_file_lock)

    snapshot_download(
        MODEL_ID,
        local_dir=str(local_dir),
        enable_file_lock=args.enable_file_lock,
        # 跳过 15GB 的原始 .pth，transformers 只需 safetensors 分片
        ignore_patterns=["original/*.pth", "original/**"],
    )

    n = count_shards(local_dir)
    logger.info("Safetensor shards in %s: %d / 4", local_dir, n)
    if n == 4:
        logger.info("Download complete.")
    else:
        logger.warning(
            "Incomplete (%d/4 shards in root). Re-run this script to resume.", n
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
