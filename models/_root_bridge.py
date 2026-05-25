"""桥接根目录 models.py，避免 models/ 包遮蔽原模块。"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def load_root_models_module():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "vqgraph_models_root", root / "models.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod
