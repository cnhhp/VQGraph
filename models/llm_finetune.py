"""
模块5：大模型微调 — 指令数据加载、QLoRA 验证、标准 LoRA 最终训练。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from config import Config, get_config

logger = logging.getLogger(__name__)


# 默认指令模板（与 prompt.md 一致）
DEFAULT_INSTRUCTION = (
    "You are given a subgraph of a citation network, centered at a target node.\n"
    "Format:\n"
    "- Each line: <level> <structure_token> [text_tokens]\n"
    "- level 0 = target node, level 1 = direct neighbor, level 2 = two-hop neighbor\n"
    "- Nodes under the same parent are sorted by structural importance "
    "(degree + PageRank + semantic similarity)"
)


@dataclass
class InstructionSample:
    """单条指令微调样本。"""

    instruction: str
    input: str
    output: str  # 类别标签或任务答案

    def to_dict(self) -> Dict[str, str]:
        return {
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output,
        }


class InstructionDatasetBuilder:
    """将 JSONL 样本构建为 HuggingFace Dataset。"""

    def __init__(self, cfg: Optional[Config] = None) -> None:
        self.cfg = cfg or get_config()
        self.instruction_template: str = DEFAULT_INSTRUCTION

    @staticmethod
    def load_jsonl(path: Path) -> List[InstructionSample]:
        import json

        samples: List[InstructionSample] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                samples.append(
                    InstructionSample(
                        instruction=obj["instruction"],
                        input=obj["input"],
                        output=obj["output"],
                    )
                )
        return samples

    @staticmethod
    def save_jsonl(samples: List[InstructionSample], path: Path) -> None:
        import json

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
        logger.info("Wrote %d samples to %s", len(samples), path)

    def build_prompt(self, sample: InstructionSample) -> str:
        """拼接 instruction + input 为模型输入。"""
        raise NotImplementedError

    def to_hf_dataset(
        self,
        samples: List[InstructionSample],
        tokenizer: Any,
    ) -> Any:
        raise NotImplementedError


class LLMFinetuner:
    """
    两阶段微调：QLoRA 快速验证 -> 标准 LoRA 全精度训练。
    """

    def __init__(self, cfg: Optional[Config] = None) -> None:
        self.cfg = cfg or get_config()
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def load_base_model(self, use_qlora: bool = True) -> None:
        """
        加载基座 LLM。

        QLoRA: load_in_4bit=True, bnb_4bit_use_double_quant=True, nf4
        标准 LoRA: bf16 全精度
        """
        raise NotImplementedError

    def apply_lora(self) -> None:
        """应用 LoRA 适配器（r, alpha, target_modules, dropout）。"""
        raise NotImplementedError

    def build_trainer(
        self,
        train_dataset: Any,
        eval_dataset: Any,
        output_dir: Path,
    ) -> Any:
        """构建 HuggingFace Trainer + DataCollatorForSeq2Seq。"""
        raise NotImplementedError

    def train(self, num_epochs: Optional[int] = None) -> Dict[str, float]:
        """启动训练，返回 metrics。"""
        raise NotImplementedError

    def evaluate_accuracy(self, eval_dataset: Any) -> float:
        """验证/测试集分类准确率。"""
        raise NotImplementedError

    def save_model(self, save_dir: Path) -> None:
        raise NotImplementedError

    def run_qlora_phase(
        self,
        train_path: Path,
        val_path: Path,
        output_dir: Path,
    ) -> float:
        """QLoRA 阶段：3 epoch，验证通过后保存。"""
        raise NotImplementedError

    def run_lora_phase(
        self,
        train_path: Path,
        val_path: Path,
        test_path: Path,
        output_dir: Path,
    ) -> float:
        """标准 LoRA 阶段：更多 epoch，测试集报告准确率。"""
        raise NotImplementedError

    def run_pipeline(
        self,
        train_path: Path,
        val_path: Path,
        test_path: Path,
        output_dir: Path,
        skip_qlora: bool = False,
    ) -> Dict[str, float]:
        """
        完整微调流水线。

        默认先 QLoRA 验证，再标准 LoRA；``skip_qlora=True`` 跳过第一阶段。
        """
        raise NotImplementedError
