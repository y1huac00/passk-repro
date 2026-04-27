#!/usr/bin/env python3
"""Generate repeated samples for processed prompts with vLLM."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def existing_keys(path: Path, model: str, model_type: str | None) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    keys = set()
    for row in read_jsonl(path):
        if row.get("model") == model and row.get("model_type") == model_type:
            keys.add((row["problem_id"], int(row["sample_id"])))
    return keys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/processed/math500_simplerl.jsonl"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-type")
    parser.add_argument("--samples-per-problem", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--tensor-parallel-size", type=int)
    parser.add_argument("--dtype")
    parser.add_argument("--gpu-memory-utilization", type=float)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output = args.output or Path("outputs/generations") / f"{safe_name(args.model)}.jsonl"
    problems = read_jsonl(args.input)[args.offset :]
    if args.limit is not None:
        problems = problems[: args.limit]

    done = set() if args.overwrite else existing_keys(output, args.model, args.model_type)
    pending = []
    for problem in problems:
        sample_ids = [
            i for i in range(args.samples_per_problem) if (problem["id"], i) not in done
        ]
        if sample_ids:
            pending.append((problem, sample_ids))

    total = sum(len(sample_ids) for _, sample_ids in pending)
    print(f"Loaded {len(problems)} problems from {args.input}")
    print(f"Pending samples: {total}")
    print(f"Output path: {output}")
    if args.dry_run:
        return
    if args.overwrite:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("", encoding="utf-8")

    os.environ.setdefault("VLLM_USE_V1", "0")
    from vllm import LLM, SamplingParams

    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "trust_remote_code": args.trust_remote_code,
    }
    for name in (
        "tensor_parallel_size",
        "dtype",
        "gpu_memory_utilization",
        "max_model_len",
    ):
        value = getattr(args, name)
        if value is not None:
            llm_kwargs[name] = value

    llm = LLM(**llm_kwargs)
    by_n: dict[int, list[tuple[dict[str, Any], list[int]]]] = defaultdict(list)
    for item in pending:
        by_n[len(item[1])].append(item)

    written = 0
    for n, items in sorted(by_n.items()):
        sampling_kwargs: dict[str, Any] = {
            "n": n,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_new_tokens,
        }
        if args.seed is not None:
            sampling_kwargs["seed"] = args.seed
        sampling_params = SamplingParams(**sampling_kwargs)

        for start in range(0, len(items), args.batch_size):
            batch = items[start : start + args.batch_size]
            outputs = llm.generate([problem["prompt"] for problem, _ in batch], sampling_params)
            rows = []
            for (problem, sample_ids), request_output in zip(batch, outputs):
                for sample_id, completion in zip(sample_ids, request_output.outputs):
                    token_ids = getattr(completion, "token_ids", None)
                    rows.append(
                        {
                            "problem_id": problem["id"],
                            "dataset": problem.get("source"),
                            "model": args.model,
                            "model_type": args.model_type,
                            "sample_id": sample_id,
                            "temperature": args.temperature,
                            "top_p": args.top_p,
                            "max_new_tokens": args.max_new_tokens,
                            "prompt_template": problem.get("prompt_template"),
                            "response": completion.text,
                            "num_output_tokens": len(token_ids) if token_ids else None,
                            "finish_reason": getattr(completion, "finish_reason", None),
                        }
                    )
            append_jsonl(output, rows)
            written += len(rows)
            print(f"Wrote {written}/{total} samples")


if __name__ == "__main__":
    main()
