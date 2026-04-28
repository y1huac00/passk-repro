#!/usr/bin/env python3
"""Run generation, verification, and evaluation in one directory."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STOP_STRINGS = ["<|im_end|>", "<|endoftext|>", "\nHuman:", "\nAssistant:"]


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def run(command: list[str], dry_run: bool) -> None:
    print("$ " + " ".join(command))
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", type=Path, default=Path("data/processed/math500_simplerl.jsonl"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-name")
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
    parser.add_argument("--ks", default="1,2,4,8,16,32,64,128")
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--skip-evaluate", action="store_true")
    parser.add_argument("--generate-dry-run", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir or Path("outputs/experiments") / (
        args.run_name or safe_name(args.model)
    )
    paths = {
        "generations": output_dir / "generations.jsonl",
        "verified": output_dir / "verified.jsonl",
        "metrics": output_dir / "metrics.json",
        "config": output_dir / "experiment_config.json",
    }
    print(f"Experiment directory: {output_dir}")

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        with paths["config"].open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "model": args.model,
                    "backend": "vllm",
                    "model_type": args.model_type,
                    "processed": str(args.processed),
                    "samples_per_problem": args.samples_per_problem,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_new_tokens": args.max_new_tokens,
                    "stop": DEFAULT_STOP_STRINGS,
                    "limit": args.limit,
                    "offset": args.offset,
                    "ks": args.ks,
                    "paths": {key: str(value) for key, value in paths.items()},
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
            f.write("\n")

    generate_cmd = [
        sys.executable,
        "scripts/generate_vllm.py",
        "--input",
        str(args.processed),
        "--output",
        str(paths["generations"]),
        "--model",
        args.model,
        "--samples-per-problem",
        str(args.samples_per_problem),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--batch-size",
        str(args.batch_size),
        "--offset",
        str(args.offset),
    ]
    for flag, value in (
        ("--model-type", args.model_type),
        ("--limit", args.limit),
        ("--seed", args.seed),
        ("--dtype", args.dtype),
        ("--max-model-len", args.max_model_len),
    ):
        if value is not None:
            generate_cmd.extend([flag, str(value)])
    for flag, value in (
        ("--tensor-parallel-size", args.tensor_parallel_size),
        ("--gpu-memory-utilization", args.gpu_memory_utilization),
    ):
        if value is not None:
            generate_cmd.extend([flag, str(value)])
    for flag, enabled in (
        ("--trust-remote-code", args.trust_remote_code),
        ("--overwrite", args.overwrite),
        ("--dry-run", args.generate_dry_run),
    ):
        if enabled:
            generate_cmd.append(flag)

    verify_cmd = [
        sys.executable,
        "scripts/verify.py",
        "--predictions",
        str(paths["generations"]),
        "--problems",
        str(args.processed),
        "--output",
        str(paths["verified"]),
    ]
    if args.force_extract:
        verify_cmd.append("--force-extract")

    evaluate_cmd = [
        sys.executable,
        "scripts/evaluate.py",
        "--input",
        str(paths["verified"]),
        "--output",
        str(paths["metrics"]),
        "--ks",
        args.ks,
    ]

    if not args.skip_generate:
        run(generate_cmd, args.dry_run)
    if not args.skip_verify and not args.generate_dry_run:
        run(verify_cmd, args.dry_run)
    if not args.skip_evaluate and not args.generate_dry_run:
        run(evaluate_cmd, args.dry_run)


if __name__ == "__main__":
    main()
