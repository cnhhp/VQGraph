"""
模块5：大模型微调 — 指令数据加载、QLoRA 验证、标准 LoRA 最终训练。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

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

INSTRUCTION_PCODE_STRUCT_SUPPLEMENT = (
    "You are given a subgraph of a citation network, centered at a target node.\n"
    "Format:\n"
    "- Each line: <level> <structure_token> [text_tokens]\n"
    "- structure_token: <S_k|w1,w2,...> — k is the structural code index; "
    "words after | are top semantic tokens predicted from the codebook (P_code)\n"
    "- level 0 = target node, level 1 = direct neighbor, level 2 = two-hop neighbor\n"
    "- Nodes under the same parent are sorted by structural importance "
    "(degree + PageRank + semantic similarity)"
)

INSTRUCTION_PCODE_STRUCT_REPLACE = (
    "You are given a subgraph of a citation network, centered at a target node.\n"
    "Format:\n"
    "- Each line: <level> [struct: w1, w2, ...] [text_tokens]\n"
    "- [struct: ...] lists top semantic tokens from the structural codebook (P_code) "
    "for that node; [text_tokens] are content keywords from the paper text\n"
    "- level 0 = target node, level 1 = direct neighbor, level 2 = two-hop neighbor\n"
    "- Nodes under the same parent are sorted by structural importance "
    "(degree + PageRank + semantic similarity)"
)

INSTRUCTION_STRUCT_SUMMARY = (
    "You are given a subgraph of a citation network, centered at a target node.\n"
    "Format:\n"
    "- First line: [struct_summary: ...] — center structural code with P_code keywords "
    "and structural-code counts in the subgraph (one summary only)\n"
    "- Each following line: <level> <structure_token> [text_tokens]\n"
    "- level 0 = target node, level 1 = direct neighbor, level 2 = two-hop neighbor\n"
    "- Nodes under the same parent are sorted by structural importance "
    "(degree + PageRank + semantic similarity)"
)

INSTRUCTION_PROJECTED_VECTOR = (
    "You are given a subgraph of a citation network, centered at a target node.\n"
    "Format:\n"
    "- Each line: <level> [v1, v2, ..., v8] [text_tokens]\n"
    "- [v1..v8] is an 8-d PCA projection of the node's structural codebook vector\n"
    "- level 0 = target node, level 1 = direct neighbor, level 2 = two-hop neighbor\n"
    "- Nodes under the same parent are sorted by structural importance "
    "(degree + PageRank + semantic similarity)"
)


def instruction_for_struct_mode(mode: str) -> str:
    """按 struct_token_mode 返回匹配的 instruction 模板。"""
    if mode == "pcode_supplement":
        return INSTRUCTION_PCODE_STRUCT_SUPPLEMENT
    if mode == "pcode_replace":
        return INSTRUCTION_PCODE_STRUCT_REPLACE
    if mode == "struct_summary":
        return INSTRUCTION_STRUCT_SUMMARY
    if mode == "projected_vector":
        return INSTRUCTION_PROJECTED_VECTOR
    return DEFAULT_INSTRUCTION


class QLoRANotAvailableError(RuntimeError):
    """QLoRA 所需依赖或硬件不可用。"""


def normalize_prediction(text: str) -> str:
    """将模型输出规范为类名字符串（与 resolve_node_class_name 格式一致）。"""
    raw = str(text).strip()
    if not raw:
        return ""
    # 取首行，去掉常见前缀
    line = raw.splitlines()[0].strip()
    for prefix in ("Answer:", "Category:", "Class:", "Output:", "Response:"):
        if line.lower().startswith(prefix.lower()):
            line = line[len(prefix) :].strip()
    # 去掉引号与尾部标点
    line = line.strip("\"'`.,; ")
    # 统一空白/连字符为下划线，并做 Title_Case
    parts = re.split(r"[\s\-]+", line)
    return "_".join(p.capitalize() for p in parts if p)


def _bitsandbytes_available() -> bool:
    try:
        import bitsandbytes  # noqa: F401

        return torch.cuda.is_available()
    except ImportError:
        return False


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
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
        logger.info("Wrote %d samples to %s", len(samples), path)

    def _user_content(self, sample: InstructionSample) -> str:
        return f"{sample.instruction}\n\nSubgraph:\n{sample.input}"

    def build_prompt(self, sample: InstructionSample, tokenizer: Any) -> str:
        """拼接 instruction + input 为模型输入（不含 answer）。"""
        user_text = self._user_content(sample)
        if hasattr(tokenizer, "apply_chat_template") and getattr(
            tokenizer, "chat_template", None
        ):
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You classify citation-network subgraphs into topic categories. "
                        "Reply with the category name only."
                    ),
                },
                {"role": "user", "content": user_text},
            ]
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return (
            f"### Instruction:\n{sample.instruction}\n\n"
            f"### Input:\n{sample.input}\n\n"
            f"### Response:\n"
        )

    def build_full_text(self, sample: InstructionSample, tokenizer: Any) -> str:
        """训练用完整文本：prompt + answer。"""
        user_text = self._user_content(sample)
        if hasattr(tokenizer, "apply_chat_template") and getattr(
            tokenizer, "chat_template", None
        ):
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You classify citation-network subgraphs into topic categories. "
                        "Reply with the category name only."
                    ),
                },
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": sample.output},
            ]
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        return (
            f"### Instruction:\n{sample.instruction}\n\n"
            f"### Input:\n{sample.input}\n\n"
            f"### Response:\n{sample.output}"
        )

    def build_generation_prompt(self, sample: InstructionSample, tokenizer: Any) -> str:
        """评估用：仅含 instruction + input。"""
        return self.build_prompt(sample, tokenizer)

    def to_hf_dataset(
        self,
        samples: List[InstructionSample],
        tokenizer: Any,
        max_seq_length: Optional[int] = None,
    ) -> Any:
        """
        转为 HF Dataset，labels 对 prompt 部分置 -100（仅对 response 计算 CE loss）。
        """
        from datasets import Dataset

        max_len = max_seq_length or self.cfg.max_seq_length
        rows: List[Dict[str, Any]] = []

        for sample in samples:
            full_text = self.build_full_text(sample, tokenizer)
            prompt_text = self.build_prompt(sample, tokenizer)

            full_ids = tokenizer(
                full_text,
                truncation=False,
                add_special_tokens=False,
            )["input_ids"]
            prompt_ids = tokenizer(
                prompt_text,
                truncation=False,
                add_special_tokens=False,
            )["input_ids"]

            if len(full_ids) > max_len:
                full_ids = full_ids[:max_len]
                prompt_len = min(len(prompt_ids), max_len - 1)
            else:
                prompt_len = len(prompt_ids)

            labels = [-100] * len(full_ids)
            for i in range(prompt_len, len(full_ids)):
                labels[i] = full_ids[i]

            rows.append(
                {
                    "input_ids": full_ids,
                    "attention_mask": [1] * len(full_ids),
                    "labels": labels,
                }
            )

        return Dataset.from_list(rows)


class LLMFinetuner:
    """
    两阶段微调：QLoRA 快速验证 -> 标准 LoRA 全精度训练。
    """

    def __init__(
        self,
        cfg: Optional[Config] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.cfg = cfg or get_config()
        self.device = self._resolve_device(device)
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self._use_qlora = False
        self._raw_samples: Dict[str, List[InstructionSample]] = {}

    @staticmethod
    def _resolve_device(device: Optional[Union[str, torch.device]]) -> torch.device:
        if device is not None:
            if isinstance(device, torch.device):
                return device
            if str(device).lower() in ("cpu", "-1"):
                return torch.device("cpu")
            if str(device).startswith("cuda"):
                return torch.device(device)
            return torch.device(f"cuda:{device}")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _ensure_pad_token(self, tokenizer: Any) -> None:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

    def load_base_model(self, use_qlora: bool = True) -> None:
        """
        加载基座 LLM。

        QLoRA: load_in_4bit=True, bnb_4bit_use_double_quant=True, nf4
        标准 LoRA: bf16/fp16 全精度（CPU 用 fp32）
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._use_qlora = use_qlora
        model_name = self.cfg.base_model_name

        if use_qlora and not _bitsandbytes_available():
            raise QLoRANotAvailableError(
                "QLoRA requires CUDA + bitsandbytes. "
                "Install bitsandbytes and use a GPU, or run with --mode lora."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        self._ensure_pad_token(self.tokenizer)

        model_kwargs: Dict[str, Any] = {"trust_remote_code": True}

        if use_qlora:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model_kwargs["device_map"] = "auto"
        else:
            if self.device.type == "cuda":
                dtype = torch.bfloat16 if self.cfg.use_bf16 else torch.float16
                model_kwargs["torch_dtype"] = dtype
                model_kwargs["device_map"] = "auto"
            else:
                model_kwargs["torch_dtype"] = torch.float32
                model_kwargs["device_map"] = "cpu"

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        if not use_qlora and self.device.type == "cpu":
            self.model = self.model.to(self.device)

        logger.info(
            "Loaded base model %s (qlora=%s, device=%s)",
            model_name,
            use_qlora,
            self.device,
        )

    def apply_lora(self) -> None:
        """应用 LoRA 适配器（r, alpha, target_modules, dropout）。"""
        from peft import LoraConfig, TaskType, get_peft_model

        if self.model is None:
            raise RuntimeError("Call load_base_model() before apply_lora().")

        lora_config = LoraConfig(
            r=self.cfg.lora_r,
            lora_alpha=self.cfg.lora_alpha,
            target_modules=list(self.cfg.lora_target_modules),
            lora_dropout=self.cfg.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def _prepare_datasets(
        self,
        train_path: Path,
        val_path: Path,
        test_path: Optional[Path] = None,
        max_samples: Optional[int] = None,
    ) -> Tuple[Any, Any, Optional[Any]]:
        builder = InstructionDatasetBuilder(self.cfg)
        train_samples = builder.load_jsonl(train_path)
        val_samples = builder.load_jsonl(val_path)
        test_samples = builder.load_jsonl(test_path) if test_path else None

        if max_samples is not None:
            train_samples = train_samples[:max_samples]
            val_samples = val_samples[:max_samples]
            if test_samples is not None:
                test_samples = test_samples[:max_samples]

        self._raw_samples = {
            "train": train_samples,
            "val": val_samples,
        }
        if test_samples is not None:
            self._raw_samples["test"] = test_samples

        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded.")

        train_ds = builder.to_hf_dataset(train_samples, self.tokenizer)
        val_ds = builder.to_hf_dataset(val_samples, self.tokenizer)
        test_ds = (
            builder.to_hf_dataset(test_samples, self.tokenizer)
            if test_samples is not None
            else None
        )
        return train_ds, val_ds, test_ds

    def build_trainer(
        self,
        train_dataset: Any,
        eval_dataset: Any,
        output_dir: Path,
        num_epochs: Optional[int] = None,
    ) -> Any:
        """构建 HuggingFace Trainer + DataCollatorForLanguageModeling。"""
        from transformers import Trainer, TrainingArguments

        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model and tokenizer must be loaded first.")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        use_cuda = self.device.type == "cuda" and torch.cuda.is_available()
        use_bf16 = use_cuda and self.cfg.use_bf16

        epochs = num_epochs if num_epochs is not None else self.cfg.lora_epochs

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=self.cfg.finetune_batch_size,
            per_device_eval_batch_size=self.cfg.finetune_eval_batch_size,
            gradient_accumulation_steps=self.cfg.gradient_accumulation_steps,
            learning_rate=self.cfg.finetune_lr,
            warmup_ratio=self.cfg.warmup_ratio,
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="no",
            bf16=use_bf16,
            fp16=use_cuda and not use_bf16,
            gradient_checkpointing=use_cuda,
            report_to="none",
            seed=self.cfg.finetune_seed,
            remove_unused_columns=False,
        )

        if use_cuda and training_args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()

        def _collate(features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
            """按 batch 内最长序列 padding；labels 用 -100 填充。"""
            max_len = max(len(f["input_ids"]) for f in features)
            pad_multiple = 8 if use_cuda else None
            if pad_multiple and max_len % pad_multiple != 0:
                max_len = max_len + (pad_multiple - max_len % pad_multiple)

            pad_id = self.tokenizer.pad_token_id
            input_ids, attention_mask, labels = [], [], []
            for f in features:
                pad_len = max_len - len(f["input_ids"])
                input_ids.append(f["input_ids"] + [pad_id] * pad_len)
                attention_mask.append(f["attention_mask"] + [0] * pad_len)
                labels.append(f["labels"] + [-100] * pad_len)
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            }

        data_collator = _collate

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        return self.trainer

    def train(self, num_epochs: Optional[int] = None) -> Dict[str, float]:
        """启动训练，返回 metrics。"""
        if self.trainer is None:
            raise RuntimeError("Call build_trainer() before train().")

        if num_epochs is not None:
            self.trainer.args.num_train_epochs = num_epochs

        result = self.trainer.train()
        metrics = dict(result.metrics) if hasattr(result, "metrics") else {}
        logger.info("Training finished: %s", metrics)
        return metrics

    @torch.inference_mode()
    def evaluate_accuracy(
        self,
        split: str = "val",
        max_samples: Optional[int] = None,
        log_errors: int = 5,
    ) -> float:
        """验证/测试集分类准确率（生成 + 精确匹配）。"""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded.")

        samples = self._raw_samples.get(split)
        if not samples:
            raise ValueError(f"No samples for split '{split}'.")

        if max_samples is not None:
            samples = samples[:max_samples]

        self.model.eval()
        builder = InstructionDatasetBuilder(self.cfg)
        correct = 0
        total = len(samples)
        errors_logged = 0

        gen_device = next(self.model.parameters()).device

        for sample in samples:
            prompt = builder.build_generation_prompt(sample, self.tokenizer)
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.cfg.max_seq_length,
            )
            inputs = {k: v.to(gen_device) for k, v in inputs.items()}

            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            new_tokens = output_ids[0, inputs["input_ids"].shape[1] :]
            pred_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            pred = normalize_prediction(pred_text)
            gold = normalize_prediction(sample.output)

            if pred == gold:
                correct += 1
            elif errors_logged < log_errors:
                logger.info(
                    "Mismatch [%s]: pred=%r gold=%r raw=%r",
                    split,
                    pred,
                    gold,
                    pred_text[:80],
                )
                errors_logged += 1

        acc = correct / total if total else 0.0
        logger.info("Accuracy on %s: %.4f (%d/%d)", split, acc, correct, total)
        return acc

    def save_model(self, save_dir: Path, metrics: Optional[Dict[str, Any]] = None) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        if self.model is not None:
            self.model.save_pretrained(save_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(save_dir)
        if metrics is not None:
            with open(save_dir / "finetune_metrics.json", "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
        logger.info("Saved model to %s", save_dir)

    def _cleanup(self) -> None:
        """释放模型显存，便于下一阶段重新加载基座。"""
        self.trainer = None
        if self.model is not None:
            del self.model
            self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def run_qlora_phase(
        self,
        train_path: Path,
        val_path: Path,
        output_dir: Path,
        max_samples: Optional[int] = None,
        num_epochs: Optional[int] = None,
    ) -> float:
        """QLoRA 阶段：3 epoch，验证集准确率，通过后保存。"""
        output_dir = Path(output_dir)
        epochs = num_epochs or self.cfg.qlora_epochs

        self.load_base_model(use_qlora=True)
        self.apply_lora()
        train_ds, val_ds, _ = self._prepare_datasets(
            train_path, val_path, max_samples=max_samples
        )
        self.build_trainer(train_ds, val_ds, output_dir / "checkpoints", num_epochs=epochs)
        self.train(num_epochs=epochs)

        acc = self.evaluate_accuracy(split="val", max_samples=max_samples)
        threshold = self.cfg.qlora_val_acc_threshold
        metrics = {"qlora_val_acc": acc, "epochs": epochs}

        if acc >= threshold:
            self.save_model(output_dir, metrics=metrics)
            logger.info("QLoRA phase passed (acc=%.4f >= threshold=%.4f)", acc, threshold)
        else:
            logger.warning(
                "QLoRA val acc %.4f below threshold %.4f; checkpoint still saved.",
                acc,
                threshold,
            )
            self.save_model(output_dir, metrics=metrics)

        self._cleanup()
        return acc

    def run_lora_phase(
        self,
        train_path: Path,
        val_path: Path,
        test_path: Path,
        output_dir: Path,
        max_samples: Optional[int] = None,
        num_epochs: Optional[int] = None,
    ) -> float:
        """标准 LoRA 阶段：重新加载全精度基座，训练后在测试集报告准确率。"""
        output_dir = Path(output_dir)
        epochs = num_epochs or self.cfg.lora_epochs

        self.load_base_model(use_qlora=False)
        self.apply_lora()
        train_ds, val_ds, test_ds = self._prepare_datasets(
            train_path, val_path, test_path, max_samples=max_samples
        )
        self.build_trainer(train_ds, val_ds, output_dir / "checkpoints", num_epochs=epochs)
        self.train(num_epochs=epochs)

        # 验证集 + 测试集准确率
        val_acc = self.evaluate_accuracy(split="val", max_samples=max_samples)
        test_acc = self.evaluate_accuracy(split="test", max_samples=max_samples)
        metrics = {
            "lora_val_acc": val_acc,
            "lora_test_acc": test_acc,
            "epochs": epochs,
        }
        self.save_model(output_dir, metrics=metrics)
        self._cleanup()
        return test_acc

    def run_pipeline(
        self,
        train_path: Path,
        val_path: Path,
        test_path: Path,
        output_dir: Path,
        skip_qlora: bool = False,
        max_samples: Optional[int] = None,
        qlora_epochs: Optional[int] = None,
        lora_epochs: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        完整微调流水线。

        默认先 QLoRA 验证，再标准 LoRA；``skip_qlora=True`` 跳过第一阶段。
        """
        output_dir = Path(output_dir)
        metrics: Dict[str, float] = {}

        if not skip_qlora:
            try:
                qlora_acc = self.run_qlora_phase(
                    train_path,
                    val_path,
                    output_dir / "qlora",
                    max_samples=max_samples,
                    num_epochs=qlora_epochs,
                )
                metrics["qlora_val_acc"] = qlora_acc
            except QLoRANotAvailableError as exc:
                logger.warning("Skipping QLoRA phase: %s", exc)
                metrics["qlora_val_acc"] = float("nan")

        test_acc = self.run_lora_phase(
            train_path,
            val_path,
            test_path,
            output_dir / "lora",
            max_samples=max_samples,
            num_epochs=lora_epochs,
        )
        metrics["lora_test_acc"] = test_acc

        summary_path = output_dir / "finetune_metrics.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        logger.info("Pipeline metrics: %s", metrics)
        return metrics
