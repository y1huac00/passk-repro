#!/usr/bin/env python3
"""Compute pass@k metrics from verified generation JSONL."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from math import prod
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def pass_at_k(n: int, c: int, k: int) -> float | None:
    if k > n:
        return None
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - prod(1.0 - k / i for i in range(n - c + 1, n + 1))


def model_key(row: dict[str, Any]) -> str:
    return f"{row.get('model', 'unknown')}::{row.get('model_type') or 'unknown'}"


def evaluate(rows: list[dict[str, Any]], ks: list[int]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(model_key(row), row["problem_id"])].append(row)

    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (key, problem_id), samples in sorted(grouped.items()):
        first = samples[0]
        correct = sum(bool(sample.get("is_correct")) for sample in samples)
        by_model[key].append(
            {
                "problem_id": problem_id,
                "model": first.get("model"),
                "model_type": first.get("model_type"),
                "dataset": first.get("dataset"),
                "num_samples": len(samples),
                "num_correct": correct,
                "solved": correct > 0,
            }
        )

    models: dict[str, Any] = {}
    for key, problems in sorted(by_model.items()):
        first = problems[0]
        pass_values = {}
        for k in ks:
            values = [
                value
                for value in (
                    pass_at_k(p["num_samples"], p["num_correct"], k) for p in problems
                )
                if value is not None
            ]
            if values:
                pass_values[str(k)] = sum(values) / len(values)

        models[key] = {
            "model": first.get("model"),
            "model_type": first.get("model_type"),
            "dataset": first.get("dataset"),
            "num_problems": len(problems),
            "sample_counts": sorted({p["num_samples"] for p in problems}),
            "solved_problems": sum(p["solved"] for p in problems),
            "pass_at_k": pass_values,
        }

    return {
        "ks": ks,
        "num_rows": len(rows),
        "models": models,
        "problem_stats": [p for problems in by_model.values() for p in problems],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ks", default="1,2,4,8,16,32,64,128")
    args = parser.parse_args()

    ks = [int(k) for k in args.ks.split(",") if k]
    metrics = evaluate(read_jsonl(args.input), ks)
    write_json(args.output, metrics)
    print(f"Wrote metrics to {args.output}")


if __name__ == "__main__":
    main()
