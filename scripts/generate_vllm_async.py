#!/usr/bin/env python3
"""Generate repeated samples with vLLM's async engine.

This submits each sample as its own request so vLLM can continuously batch
active requests instead of waiting for a fixed offline batch to finish.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_STOP_STRINGS = ["<|im_end|>", "<|endoftext|>", "Human:", "Assistant:"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_experiment_config(args: argparse.Namespace, output: Path) -> None:
    output_dir = output.parent
    paths = {
        "generations": output,
        "verified": output_dir / "verified.jsonl",
        "metrics": output_dir / "metrics.json",
        "config": output_dir / "experiment_config.json",
    }
    config = {
        "model": args.model,
        "backend": "vllm_async",
        "model_type": args.model_type,
        "processed": str(args.input),
        "samples_per_problem": args.samples_per_problem,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "stop": DEFAULT_STOP_STRINGS,
        "limit": args.limit,
        "offset": args.offset,
        "seed": args.seed,
        "max_concurrent_requests": args.max_concurrent_requests,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "write_every": args.write_every,
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": args.dtype,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "trust_remote_code": args.trust_remote_code,
        "paths": {key: str(value) for key, value in paths.items()},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with paths["config"].open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
        f.write("\n")


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


def build_engine(args: argparse.Namespace) -> Any:
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine

    engine_kwargs: dict[str, Any] = {
        "model": args.model,
        "trust_remote_code": args.trust_remote_code,
    }
    for name in (
        "tensor_parallel_size",
        "dtype",
        "gpu_memory_utilization",
        "max_model_len",
        "max_num_seqs",
        "max_num_batched_tokens",
    ):
        value = getattr(args, name)
        if value is not None:
            engine_kwargs[name] = value

    return AsyncLLMEngine.from_engine_args(
        AsyncEngineArgs(**engine_kwargs),
        start_engine_loop=True,
    )


def build_sampling_params(args: argparse.Namespace, request_index: int) -> Any:
    from vllm import SamplingParams

    sampling_kwargs: dict[str, Any] = {
        "n": 1,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_new_tokens,
        "stop": DEFAULT_STOP_STRINGS,
    }
    if args.seed is not None:
        sampling_kwargs["seed"] = args.seed + request_index
    return SamplingParams(**sampling_kwargs)


def make_requests(
    problems: list[dict[str, Any]],
    done: set[tuple[str, int]],
    samples_per_problem: int,
) -> list[tuple[dict[str, Any], int]]:
    requests = []
    for problem in problems:
        for sample_id in range(samples_per_problem):
            if (problem["id"], sample_id) not in done:
                requests.append((problem, sample_id))
    return requests


async def run_one_request(
    *,
    engine: Any,
    args: argparse.Namespace,
    problem: dict[str, Any],
    sample_id: int,
    request_index: int,
) -> dict[str, Any]:
    request_id = f"{problem['id']}-{sample_id}-{request_index}"
    final_output = None
    generator = engine.generate(
        problem["prompt"],
        build_sampling_params(args, request_index),
        request_id,
    )
    async for request_output in generator:
        final_output = request_output

    if final_output is None:
        raise RuntimeError(f"Request produced no output: {request_id}")

    completion = final_output.outputs[0]
    token_ids = getattr(completion, "token_ids", None)
    return {
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


async def run_async(args: argparse.Namespace) -> None:
    output = args.output or Path("outputs/generations") / f"{safe_name(args.model)}.jsonl"
    problems = read_jsonl(args.input)[args.offset :]
    if args.limit is not None:
        problems = problems[: args.limit]

    done = set() if args.overwrite else existing_keys(output, args.model, args.model_type)
    requests = make_requests(problems, done, args.samples_per_problem)

    print(f"Loaded {len(problems)} problems from {args.input}")
    print(f"Pending samples: {len(requests)}")
    print(f"Output path: {output}")
    print(f"Max concurrent requests: {args.max_concurrent_requests}")
    if args.dry_run:
        return
    write_experiment_config(args, output)
    if args.overwrite:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("", encoding="utf-8")

    engine = build_engine(args)
    semaphore = asyncio.Semaphore(args.max_concurrent_requests)

    async def run_limited(
        item: tuple[int, tuple[dict[str, Any], int]],
    ) -> dict[str, Any]:
        request_index, (problem, sample_id) = item
        async with semaphore:
            return await run_one_request(
                engine=engine,
                args=args,
                problem=problem,
                sample_id=sample_id,
                request_index=request_index,
            )

    written = 0
    buffer: list[dict[str, Any]] = []
    tasks = [asyncio.create_task(run_limited(item)) for item in enumerate(requests)]
    for task in asyncio.as_completed(tasks):
        buffer.append(await task)
        if len(buffer) >= args.write_every:
            append_jsonl(output, buffer)
            written += len(buffer)
            buffer.clear()
            print(f"Wrote {written}/{len(requests)} samples")

    if buffer:
        append_jsonl(output, buffer)
        written += len(buffer)
        print(f"Wrote {written}/{len(requests)} samples")


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
    parser.add_argument("--max-concurrent-requests", type=int, default=128)
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument("--max-num-batched-tokens", type=int)
    parser.add_argument("--write-every", type=int, default=16)
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

    if args.max_concurrent_requests < 1:
        raise ValueError("--max-concurrent-requests must be >= 1")
    if args.max_num_seqs is not None and args.max_num_seqs < 1:
        raise ValueError("--max-num-seqs must be >= 1")
    if args.max_num_batched_tokens is not None and args.max_num_batched_tokens < 1:
        raise ValueError("--max-num-batched-tokens must be >= 1")
    if args.write_every < 1:
        raise ValueError("--write-every must be >= 1")

    asyncio.run(run_async(args))


if __name__ == "__main__":
    main()
