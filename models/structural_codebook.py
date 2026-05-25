"""模块1 接口 re-export（实现见 codebook_trainer.py）。"""

from models.codebook_trainer import (
    CodebookArtifacts,
    CodebookTrainer,
    TFIDFComputer,
    TFIDFStatistics,
)

# 向后兼容旧名称
StructuralCodebookTrainer = CodebookTrainer

__all__ = [
    "CodebookArtifacts",
    "CodebookTrainer",
    "StructuralCodebookTrainer",
    "TFIDFComputer",
    "TFIDFStatistics",
]
