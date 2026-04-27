#!/usr/bin/env python3
"""Generate repeated samples with Hugging Face Transformers."""

from __future__ import annotations

import argparse
import json
import re
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


def torch_dtype(torch: Any, name: str | None) -> Any:
    if name is None or name == "auto":
        return "auto"
    aliases = {
        "fp16": "float16",
        "float16": "float16",
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
        "fp32": "float32",
        "float32": "float32",
    }
    return getattr(torch, aliases[name])


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
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device")
    parser.add_argument("--device-map")
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
        for sample_id in range(args.samples_per_problem):
            if (problem["id"], sample_id) not in done:
                pending.append((problem, sample_id))

    print(f"Loaded {len(problems)} problems from {args.input}")
    print(f"Pending samples: {len(pending)}")
    print(f"Output path: {output}")
    if args.dry_run:
        return
    if args.overwrite:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("", encoding="utf-8")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.seed is not None:
        torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
        "torch_dtype": torch_dtype(torch, args.dtype),
    }
    if args.device_map:
        model_kwargs["device_map"] = args.device_map

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    if not args.device_map:
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
    model.eval()

    written = 0
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        prompts = [problem["prompt"] for problem, _ in batch]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=args.max_model_len is not None,
            max_length=args.max_model_len,
        )
        if not args.device_map:
            inputs = {key: value.to(model.device) for key, value in inputs.items()}

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                do_sample=args.temperature > 0,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        rows = []
        for (problem, sample_id), ids in zip(batch, output_ids):
            completion_ids = ids[prompt_len:]
            completion_token_ids = completion_ids.tolist()
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
                    "response": tokenizer.decode(completion_ids, skip_special_tokens=True),
                    "num_output_tokens": len(completion_ids),
                    "finish_reason": "eos"
                    if tokenizer.eos_token_id is not None
                    and tokenizer.eos_token_id in completion_token_ids
                    else "length",
                }
            )
        append_jsonl(output, rows)
        written += len(rows)
        print(f"Wrote {written}/{len(pending)} samples")


if __name__ == "__main__":
    main()
