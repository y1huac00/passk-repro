#!/usr/bin/env python3
"""Preprocess MATH-500 with the SimpleRL prompt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROMPT = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{question}
Please reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>
<|im_start|>assistant
"""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_record(row: dict[str, Any], index: int) -> dict[str, Any]:
    problem = row["problem"].strip()
    return {
        "id": f"math500_{index:04d}",
        "source": "MATH500",
        "split": "test",
        "unique_id": row.get("unique_id"),
        "subject": row.get("subject"),
        "level": row.get("level"),
        "problem": problem,
        "prompt_template": "SimpleRL",
        "prompt": PROMPT.format(question=problem),
        "answer": row["answer"].strip(),
        "solution": row.get("solution", "").strip(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/raw/MATH-500/test.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/math500_simplerl.jsonl"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    if args.limit is not None:
        rows = rows[: args.limit]
    write_jsonl(args.output, [make_record(row, i) for i, row in enumerate(rows, start=1)])
    print(f"Wrote {len(rows)} records to {args.output}")


if __name__ == "__main__":
    main()
