#!/usr/bin/env python3
"""Small vLLM diagnostics independent of the experiment pipeline."""

from __future__ import annotations

import argparse
import os
import platform
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="What is 2 + 2? Answer briefly.")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--n", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--tensor-parallel-size", type=int)
    parser.add_argument("--dtype")
    parser.add_argument("--gpu-memory-utilization", type=float)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--use-v1", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("VLLM_USE_V1", "1" if args.use_v1 else "0")
    print(f"python={platform.python_version()}")
    print(f"VLLM_USE_V1={os.environ.get('VLLM_USE_V1')}")

    import vllm
    from vllm import LLM, SamplingParams

    print(f"vllm={getattr(vllm, '__version__', 'unknown')}")

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

    print(f"loading model={args.model}")
    llm = LLM(**llm_kwargs)

    sampling_kwargs: dict[str, Any] = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }
    if args.seed is not None:
        sampling_kwargs["seed"] = args.seed

    print("\n[1] single prompt")
    outputs = llm.generate([args.prompt], SamplingParams(**sampling_kwargs))
    for output in outputs[0].outputs:
        print(output.text.strip())

    print("\n[2] batch prompts")
    prompts = [f"{args.prompt}\nCase {i}:" for i in range(args.batch_size)]
    outputs = llm.generate(prompts, SamplingParams(**sampling_kwargs))
    for i, output in enumerate(outputs):
        print(f"batch[{i}]={output.outputs[0].text.strip()}")

    print("\n[3] n-best samples")
    sampling_kwargs["n"] = args.n
    outputs = llm.generate([args.prompt], SamplingParams(**sampling_kwargs))
    for i, output in enumerate(outputs[0].outputs):
        token_ids = getattr(output, "token_ids", None)
        print(f"sample[{i}] finish={output.finish_reason} tokens={len(token_ids) if token_ids else None}")
        print(output.text.strip())


if __name__ == "__main__":
    main()
