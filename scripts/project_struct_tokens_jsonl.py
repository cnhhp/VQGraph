"""
将 JSONL 行内 <S_k> 替换为 PCA 投影向量，text token 保持不变。

用法::
  PYTHONPATH=. python scripts/build_projected_codebook.py \\
    --codebook_dir ./outputs/experiments/e5b_pubmed_m4096/pubmed/GCN/seed_1 \\
    --output_dir ./outputs/experiments/e5b_pubmed_m4096/pubmed/GCN/seed_1/projected_k8

  PYTHONPATH=. python scripts/project_struct_tokens_jsonl.py \\
    --input_dir ./data/llm_finetune_json/llm_finetune_e5b_no_ltoken_pubmed \\
    --projected_codebook_dir ./outputs/experiments/e5b_pubmed_m4096/pubmed/GCN/seed_1/projected_k8 \\
    --output_dir ./data/llm_finetune_json/llm_finetune_e5b_projected_k8_pubmed
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from models.llm_finetune import INSTRUCTION_PROJECTED_VECTOR
from models.struct_projector import StructuralProjector

_LINE_RE = re.compile(
    r"^(\d+)\s+<S_(\d+)(?:\|[^>]+)?>\s+(\[[^\]]+\])\s*$"
)


def transform_line(line: str, projector: StructuralProjector) -> str:
    line = line.strip()
    if not line:
        return line
    if line.startswith("[struct_summary:"):
        return line
    m = _LINE_RE.match(line)
    if not m:
        raise ValueError(f"Line does not match expected format: {line[:120]!r}")
    level, code_idx, text_tokens = m.group(1), int(m.group(2)), m.group(3)
    vec = projector.format_vector(code_idx)
    return f"{level} {vec} {text_tokens}"


def transform_input(text: str, projector: StructuralProjector) -> str:
    return "\n".join(
        transform_line(ln, projector) for ln in text.splitlines() if ln.strip()
    )


def process_split(src: Path, dst: Path, projector: StructuralProjector) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(src, encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            obj = json.loads(line)
            obj["instruction"] = INSTRUCTION_PROJECTED_VECTOR
            obj["input"] = transform_input(obj["input"], projector)
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> None:
    p = argparse.ArgumentParser(
        description="Replace <S_k> with PCA projected vectors in JSONL"
    )
    p.add_argument("--input_dir", type=str, required=True)
    p.add_argument("--projected_codebook_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = p.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    proj_dir = Path(args.projected_codebook_dir)
    projector = StructuralProjector.load(proj_dir)

    counts = {}
    for split in args.splits:
        src = in_dir / f"{split}.jsonl"
        if not src.is_file():
            raise FileNotFoundError(src)
        counts[split] = process_split(src, out_dir / f"{split}.jsonl", projector)

    src_manifest = in_dir / "manifest.json"
    base_meta = {}
    if src_manifest.is_file():
        with open(src_manifest, encoding="utf-8") as f:
            base_meta = json.load(f)

    manifest = {
        "source": str(in_dir),
        "transform": "project_struct_tokens",
        "format": "<level> [v1..vk] [text_tokens] per line (PCA replaces <S_k>)",
        "projected_codebook_dir": str(proj_dir.resolve()),
        "projector": {
            "M": int(projector.projections.shape[0]),
            "k": int(projector.k),
            "d_in": int(projector.d_in),
            "method": projector.method,
            "decimals": projector.decimals,
        },
        "codebook_dir": base_meta.get("codebook_dir"),
        "dataset": base_meta.get("dataset"),
        "splits": {
            k: {"path": str(out_dir / f"{k}.jsonl"), "num_samples": v}
            for k, v in counts.items()
        },
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"Wrote {out_dir} -> {counts}")


if __name__ == "__main__":
    main()
