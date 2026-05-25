"""离散图表征 Pipeline 子包；通过 __getattr__ 桥接根目录 models.py。"""

from models._root_bridge import load_root_models_module

__all__ = [
    "Model",
    "GCN",
    "SAGE",
    "CodebookTrainer",
    "StructuralCodebookTrainer",
    "CodebookArtifacts",
    "TFIDFComputer",
    "TFIDFStatistics",
    "SemanticVectorQuantize",
    "LocalSubgraph",
    "SubgraphExtractor",
]


def __getattr__(name: str):
    root = load_root_models_module()
    if name in ("Model", "GCN", "SAGE"):
        return getattr(root, name)
    if name in (
        "CodebookTrainer",
        "StructuralCodebookTrainer",
        "CodebookArtifacts",
        "TFIDFComputer",
        "TFIDFStatistics",
    ):
        from models.codebook_trainer import (
            CodebookArtifacts,
            CodebookTrainer,
            TFIDFComputer,
            TFIDFStatistics,
        )
        return {
            "CodebookTrainer": CodebookTrainer,
            "StructuralCodebookTrainer": CodebookTrainer,
            "CodebookArtifacts": CodebookArtifacts,
            "TFIDFComputer": TFIDFComputer,
            "TFIDFStatistics": TFIDFStatistics,
        }[name]
    if name == "SemanticVectorQuantize":
        from models.semantic_vq import SemanticVectorQuantize
        return SemanticVectorQuantize
    if name in ("LocalSubgraph", "SubgraphExtractor"):
        from models.subgraph_extraction import LocalSubgraph, SubgraphExtractor
        return {
            "LocalSubgraph": LocalSubgraph,
            "SubgraphExtractor": SubgraphExtractor,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
