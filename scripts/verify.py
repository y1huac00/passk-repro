#!/usr/bin/env python3
"""Extract final answers and verify them with math-verify."""

from __future__ import annotations

import argparse
import json
import re
import signal
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from math_verify import (
    ExprExtractionConfig,
    LatexExtractionConfig,
    StringExtractionConfig,
    parse,
    verify,
)


VERIFY_TIMEOUT_SECONDS = 3.0


class VerifyTimeoutError(Exception):
    pass


@contextmanager
def time_limit(seconds: float):
    def handler(signum, frame):
        raise VerifyTimeoutError()

    old_handler = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def last_boxed(text: str) -> str | None:
    answers: list[str] = []
    for command in ("\\boxed", "\\fbox"):
        start = 0
        while True:
            pos = text.find(command, start)
            if pos == -1:
                break
            left = text.find("{", pos + len(command))
            if left == -1:
                break
            depth = 0
            for right in range(left, len(text)):
                if text[right] == "{":
                    depth += 1
                elif text[right] == "}":
                    depth -= 1
                    if depth == 0:
                        answers.append(text[left + 1 : right].strip())
                        start = right + 1
                        break
            else:
                break
    return answers[-1] if answers else None


def extract_answer(response: str, dataset: str | None = None) -> tuple[str, str]:
    boxed = last_boxed(response)
    if boxed:
        return boxed, "boxed"

    match = re.findall(r"(?is)(?:final answer|answer)\s*(?:is|:)?\s*(.+)$", response)
    for value in reversed(match):
        lines = [line.strip(" .") for line in value.strip().splitlines() if line.strip(" .")]
        if lines:
            return lines[0], "final_answer"

    if dataset and dataset.upper().startswith("AIME"):
        ints = re.findall(r"(?<![\d.])-?\d+(?![\d.])", response)
        if ints:
            return ints[-1], "integer_fallback"

    lines = [line.strip() for line in response.splitlines() if line.strip()]
    if lines:
        line = lines[-1].strip(" .")
        math_spans = re.findall(r"\$([^$]+)\$", line)
        if math_spans:
            return math_spans[-1].strip(), "last_line"
        expressions = re.findall(
            r"(\\frac\s*\{[^{}]+\}\s*\{[^{}]+\}|-?\d+\s*/\s*-?\d+|-?\d+(?:\.\d+)?)",
            line,
        )
        return (expressions[-1] if expressions else line), "last_line"
    return "", "failed"


def strip_wrappers(answer: str) -> str:
    answer = answer.strip().strip("$").strip()
    answer = re.sub(r"^\\\(|\\\)$", "", answer).strip()
    answer = re.sub(r"^\\\[|\\\]$", "", answer).strip()
    return last_boxed(answer) or answer


def text_value(answer: str) -> str | None:
    text = re.sub(r"\\(?:text|mathrm)\s*\{([^{}]*)\}", r"\1", strip_wrappers(answer))
    text = re.sub(r"\s+", " ", text).strip(" .")
    if re.fullmatch(r"[A-Za-z][A-Za-z .'-]*", text):
        return text
    return None


def parse_candidates(answer: str, configs: list[Any]) -> list[Any]:
    text = text_value(answer)
    base = strip_wrappers(answer)
    candidates = [text, answer, f"${base}$"] if text else [f"${base}$", answer]
    for candidate in candidates:
        if candidate is None:
            continue
        value = parse(candidate, extraction_config=configs)
        if value:
            return [value]
    return []


def safe_repr(value: Any) -> str:
    try:
        return repr(value)
    except Exception as error:
        return f"<repr_error:{type(error).__name__}>"


def verify_answer(prediction: str, gold: str) -> dict[str, Any]:
    try:
        with time_limit(VERIFY_TIMEOUT_SECONDS):
            strings = sorted({value for value in (text_value(prediction), text_value(gold)) if value})
            configs: list[Any] = [LatexExtractionConfig(), ExprExtractionConfig()]
            if strings:
                configs.append(StringExtractionConfig(strings=tuple(strings)))

            parsed_predictions = parse_candidates(prediction, configs)
            parsed_golds = parse_candidates(gold, configs)
            for parsed_gold in parsed_golds:
                for parsed_prediction in parsed_predictions:
                    if verify(parsed_gold, parsed_prediction):
                        return {
                            "is_correct": True,
                            "verification_method": "math_verify",
                            "parsed_prediction": safe_repr(parsed_prediction),
                            "parsed_gold": safe_repr(parsed_gold),
                        }
            return {
                "is_correct": False,
                "verification_method": "math_verify_no_match",
                "parsed_prediction": safe_repr(parsed_predictions),
                "parsed_gold": safe_repr(parsed_golds),
            }
    except VerifyTimeoutError:
        return {
            "is_correct": False,
            "verification_method": "math_verify_timeout",
            "parsed_prediction": "",
            "parsed_gold": "",
        }
    except Exception as error:
        return {
            "is_correct": False,
            "verification_method": f"math_verify_error:{type(error).__name__}",
            "parsed_prediction": "",
            "parsed_gold": "",
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--problems", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force-extract", action="store_true")
    args = parser.parse_args()

    problems = {row["id"]: row for row in read_jsonl(args.problems)}
    verified = []
    for row in read_jsonl(args.predictions):
        problem_id = row.get("problem_id") or row.get("id")
        problem = problems[problem_id]
        if row.get("extracted_answer") and not args.force_extract:
            answer, method = row["extracted_answer"], row.get("extraction_method", "provided")
        else:
            answer, method = extract_answer(row.get("response", ""), row.get("dataset"))

        result = verify_answer(answer, problem["answer"])
        verified.append(
            {
                **row,
                **result,
                "problem_id": problem_id,
                "gold_answer": problem["answer"],
                "extracted_answer": answer,
                "extraction_method": method,
            }
        )

    write_jsonl(args.output, verified)
    print(f"Wrote {len(verified)} verified rows to {args.output}")


if __name__ == "__main__":
    main()
