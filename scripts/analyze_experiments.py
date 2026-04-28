#!/usr/bin/env python3
"""Analyze pass@k experiment outputs and draw simple comparison plots."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def stat(values: list[int | float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    xs = sorted(values)
    return {
        "n": len(xs),
        "mean": mean(xs),
        "median": median(xs),
        "min": xs[0],
        "max": xs[-1],
    }


def experiment_label(path: Path, config: dict[str, Any]) -> str:
    model_type = config.get("model_type")
    if model_type:
        return str(model_type)
    return path.name


def summarize_generations(rows: list[dict[str, Any]]) -> dict[str, Any]:
    token_counts = [
        row["num_output_tokens"]
        for row in rows
        if isinstance(row.get("num_output_tokens"), int)
    ]
    char_counts = [len(row.get("response", "")) for row in rows]
    finish_reasons = Counter(str(row.get("finish_reason")) for row in rows)
    return {
        "num_rows": len(rows),
        "finish_reason": dict(finish_reasons),
        "output_tokens": stat(token_counts),
        "response_chars": stat(char_counts),
    }


def summarize_verified(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_problem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_problem[row["problem_id"]].append(row)

    correct_counts = []
    sample_counts = []
    for samples in by_problem.values():
        sample_counts.append(len(samples))
        correct_counts.append(sum(bool(row.get("is_correct")) for row in samples))

    return {
        "num_rows": len(rows),
        "num_problems": len(by_problem),
        "sample_counts": sorted(set(sample_counts)),
        "solved_problems": sum(count > 0 for count in correct_counts),
        "correct_counts": stat(correct_counts),
    }


def load_experiment(path: Path) -> dict[str, Any]:
    config = read_json(path / "experiment_config.json")
    metrics = read_json(path / "metrics.json")
    generations = read_jsonl(path / "generations.jsonl")
    verified = read_jsonl(path / "verified.jsonl")
    return {
        "path": str(path),
        "label": experiment_label(path, config),
        "config": config,
        "metrics": metrics,
        "generations": summarize_generations(generations),
        "verified": summarize_verified(verified),
        "generation_rows": generations,
        "verified_rows": verified,
    }


def write_summary(experiments: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    public = []
    for exp in experiments:
        public.append(
            {
                "path": exp["path"],
                "label": exp["label"],
                "config": exp["config"],
                "metrics": exp["metrics"],
                "generations": exp["generations"],
                "verified": exp["verified"],
            }
        )

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(public, f, ensure_ascii=False, indent=2)
        f.write("\n")

    lines = []
    for exp in public:
        lines.append(f"[{exp['label']}] {exp['path']}")
        for model in exp["metrics"].get("models", {}).values():
            pairs = ", ".join(
                f"pass@{k}={v:.4f}" for k, v in model.get("pass_at_k", {}).items()
            )
            lines.append(f"  {pairs}")
        lines.append(f"  generations={exp['generations']['num_rows']}")
        lines.append(f"  verified={exp['verified']['num_rows']}")
        lines.append("")
    (output_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")


def plot_pass_at_k(experiments: list[dict[str, Any]], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4.5))
    for exp in experiments:
        for model in exp["metrics"].get("models", {}).values():
            values = model.get("pass_at_k", {})
            ks = sorted(int(k) for k in values)
            ys = [values[str(k)] for k in ks]
            plt.plot(ks, ys, marker="o", label=exp["label"])
    plt.xlabel("k")
    plt.ylabel("pass@k")
    plt.xscale("log", base=2)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "pass_at_k.png", dpi=200)
    plt.close()


def plot_correct_count_hist(experiments: list[dict[str, Any]], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4.5))
    for exp in experiments:
        counts = Counter()
        for row in exp["verified_rows"]:
            counts[row["problem_id"]] += int(bool(row.get("is_correct")))
        plt.hist(list(counts.values()), bins=30, alpha=0.45, label=exp["label"])
    plt.xlabel("correct samples per problem")
    plt.ylabel("problem count")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "correct_count_hist.png", dpi=200)
    plt.close()


def plot_output_tokens_hist(experiments: list[dict[str, Any]], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4.5))
    for exp in experiments:
        values = [
            row["num_output_tokens"]
            for row in exp["generation_rows"]
            if isinstance(row.get("num_output_tokens"), int)
        ]
        plt.hist(values, bins=50, alpha=0.45, label=exp["label"])
    plt.xlabel("output tokens")
    plt.ylabel("sample count")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "output_tokens_hist.png", dpi=200)
    plt.close()


def plot_tokens_by_correctness(experiments: list[dict[str, Any]], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(experiments), 1, figsize=(7, 3.6 * len(experiments)))
    if len(experiments) == 1:
        axes = [axes]

    for ax, exp in zip(axes, experiments):
        correct = [
            row["num_output_tokens"]
            for row in exp["verified_rows"]
            if row.get("is_correct") and isinstance(row.get("num_output_tokens"), int)
        ]
        wrong = [
            row["num_output_tokens"]
            for row in exp["verified_rows"]
            if not row.get("is_correct") and isinstance(row.get("num_output_tokens"), int)
        ]
        ax.hist([wrong, correct], bins=50, stacked=True, label=["wrong", "correct"])
        ax.set_title(exp["label"])
        ax.set_xlabel("output tokens")
        ax.set_ylabel("sample count")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "tokens_by_correctness.png", dpi=200)
    plt.close()


def plot_finish_reason(experiments: list[dict[str, Any]], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [exp["label"] for exp in experiments]
    reasons = sorted(
        {
            reason
            for exp in experiments
            for reason in exp["generations"]["finish_reason"].keys()
        }
    )
    x = list(range(len(labels)))
    width = 0.8 / max(1, len(reasons))

    plt.figure(figsize=(7, 4.5))
    for i, reason in enumerate(reasons):
        ys = [
            exp["generations"]["finish_reason"].get(reason, 0)
            for exp in experiments
        ]
        offsets = [value + (i - len(reasons) / 2) * width + width / 2 for value in x]
        plt.bar(offsets, ys, width=width, label=reason)
    plt.xticks(x, labels)
    plt.ylabel("sample count")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "finish_reason.png", dpi=200)
    plt.close()


def plot_per_problem_delta(experiments: list[dict[str, Any]], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    if len(experiments) != 2:
        return

    counts = []
    for exp in experiments:
        counter = Counter()
        for row in exp["verified_rows"]:
            counter[row["problem_id"]] += int(bool(row.get("is_correct")))
        counts.append(counter)

    common = sorted(set(counts[0]) & set(counts[1]))
    deltas = [counts[1][problem_id] - counts[0][problem_id] for problem_id in common]

    plt.figure(figsize=(7, 4.5))
    plt.hist(deltas, bins=31, alpha=0.8)
    plt.xlabel(f"correct count delta: {experiments[1]['label']} - {experiments[0]['label']}")
    plt.ylabel("problem count")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "per_problem_delta.png", dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    experiments = [load_experiment(path) for path in args.experiments]
    write_summary(experiments, args.output_dir)
    plot_pass_at_k(experiments, args.output_dir)
    plot_correct_count_hist(experiments, args.output_dir)
    plot_output_tokens_hist(experiments, args.output_dir)
    plot_tokens_by_correctness(experiments, args.output_dir)
    plot_finish_reason(experiments, args.output_dir)
    plot_per_problem_delta(experiments, args.output_dir)
    print(f"Wrote analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
