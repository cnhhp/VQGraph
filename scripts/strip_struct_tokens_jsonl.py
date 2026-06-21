"""Remove <S_k> (and <S_k|...>) from each input line; keep level + [text_tokens]."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_LINE_RE = re.compile(
    r"^(\d+)\s+<S_\d+(?:\|[^>]+)?>\s+(\[[^\]]+\])\s*$"
)

INSTRUCTION = (
    "You are given a subgraph of a citation network, centered at a target node.\n"
    "Format:\n"
    "- Each line: <level> [text_tokens]\n"
    "- level 0 = target node, level 1 = direct neighbor, level 2 = two-hop neighbor\n"
    "- Nodes under the same parent are sorted by structural importance "
    "(degree + PageRank + semantic similarity)"
)


def transform_line(line: str) -> str:
    line = line.strip()
    if not line:
        return line
    m = _LINE_RE.match(line)
    if not m:
        raise ValueError(f"Line does not match expected format: {line[:120]!r}")
    return f"{m.group(1)} {m.group(2)}"


def transform_input(text: str) -> str:
    return "\n".join(transform_line(ln) for ln in text.splitlines() if ln.strip())


def process_split(src: Path, dst: Path) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(src, encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            obj = json.loads(line)
            obj["instruction"] = INSTRUCTION
            obj["input"] = transform_input(obj["input"])
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> None:
    p = argparse.ArgumentParser(description="Strip <S_k> from JSONL input lines")
    p.add_argument("--input_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = p.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    counts = {}
    for split in args.splits:
        src = in_dir / f"{split}.jsonl"
        if not src.is_file():
            raise FileNotFoundError(src)
        counts[split] = process_split(src, out_dir / f"{split}.jsonl")

    manifest = {
        "source": str(in_dir),
        "transform": "strip_struct_tokens",
        "format": "<level> [text_tokens] per line (no <S_k>)",
        "splits": {k: {"path": str(out_dir / f"{k}.jsonl"), "num_samples": v} for k, v in counts.items()},
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"Wrote {out_dir} -> {counts}")


if __name__ == "__main__":
    main()
