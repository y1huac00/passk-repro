#!/usr/bin/env python3
"""Build a static HTML viewer for manual inspection of experiment examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_problems(path: Path) -> dict[str, dict[str, Any]]:
    return {row["id"]: row for row in read_jsonl(path)}


def make_record(label: str, row: dict[str, Any], problem: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment": label,
        "problem_id": row.get("problem_id"),
        "sample_id": row.get("sample_id"),
        "problem": problem.get("problem", ""),
        "gold_answer": row.get("gold_answer") or problem.get("answer", ""),
        "extracted_answer": row.get("extracted_answer", ""),
        "extraction_method": row.get("extraction_method", ""),
        "is_correct": row.get("is_correct"),
        "finish_reason": row.get("finish_reason"),
        "num_output_tokens": row.get("num_output_tokens"),
        "response": row.get("response", ""),
    }


def load_experiment(
    path: Path,
    problems: dict[str, dict[str, Any]],
    verified_name: str,
    generations_name: str,
    limit: int | None,
) -> dict[str, Any]:
    config_path = path / "experiment_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    label = str(config.get("model_type") or path.name)
    rows = read_jsonl(path / verified_name) or read_jsonl(path / generations_name)
    if limit is not None:
        rows = rows[:limit]
    records = [
        make_record(label, row, problems.get(row.get("problem_id"), {}))
        for row in rows
    ]
    return {"path": str(path), "label": label, "config": config, "records": records}


def write_html(experiments: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(experiments, ensure_ascii=False).replace("</", "<\\/")
    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Experiment Example Viewer</title>
<script>
window.MathJax = {{
  tex: {{
    inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
    displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
    processEscapes: true
  }}
}};
</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
<style>
body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #1f2933; background: #f6f7f9; }}
header {{ padding: 14px 18px; background: #20242c; color: white; display: flex; gap: 12px; align-items: center; }}
header input, header select {{ padding: 7px 9px; border-radius: 4px; border: 1px solid #9aa4b2; min-width: 150px; }}
main {{ display: grid; grid-template-columns: 360px 1fr; height: calc(100vh - 58px); }}
#list {{ overflow: auto; border-right: 1px solid #d6dbe1; background: white; }}
.item {{ padding: 10px 12px; border-bottom: 1px solid #eef1f4; cursor: pointer; }}
.item:hover, .item.active {{ background: #e9f1ff; }}
.item .meta {{ font-size: 12px; color: #607080; margin-top: 4px; }}
#detail {{ overflow: auto; padding: 18px; }}
.problem {{ background: white; border: 1px solid #d6dbe1; border-radius: 6px; padding: 14px; margin-bottom: 14px; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 14px; }}
.card {{ background: white; border: 1px solid #d6dbe1; border-radius: 6px; padding: 14px; }}
.card.correct {{ border-left: 5px solid #16834b; }}
.card.wrong {{ border-left: 5px solid #b42318; }}
.kv {{ font-size: 13px; color: #485766; margin: 4px 0; }}
.mathblock {{ white-space: pre-wrap; overflow-wrap: anywhere; background: #f8fafc; border: 1px solid #e1e6eb; border-radius: 4px; padding: 10px; line-height: 1.45; }}
.empty {{ color: #667789; padding: 20px; }}
</style>
</head>
<body>
<header>
  <strong>Example Viewer</strong>
  <input id="search" placeholder="search problem / answer / text">
  <select id="correct">
    <option value="all">all</option>
    <option value="wrong">has wrong</option>
    <option value="correct">has correct</option>
  </select>
  <select id="finish"><option value="all">all finish reasons</option></select>
  <span id="count"></span>
</header>
<main>
  <div id="list"></div>
  <div id="detail"><div class="empty">Select an example.</div></div>
</main>
<script>
const experiments = {payload};
const records = experiments.flatMap(exp => exp.records);
const groups = new Map();
for (const r of records) {{
  const key = `${{r.problem_id}}::${{r.sample_id}}`;
  if (!groups.has(key)) groups.set(key, []);
  groups.get(key).push(r);
}}
let selectedKey = null;

const search = document.getElementById("search");
const correct = document.getElementById("correct");
const finish = document.getElementById("finish");
const list = document.getElementById("list");
const detail = document.getElementById("detail");
const count = document.getElementById("count");

for (const reason of [...new Set(records.map(r => String(r.finish_reason)))].sort()) {{
  const option = document.createElement("option");
  option.value = reason;
  option.textContent = reason;
  finish.appendChild(option);
}}

function textOf(group) {{
  return group.map(r => [
    r.problem_id, r.sample_id, r.problem, r.gold_answer, r.extracted_answer, r.response
  ].join("\\n")).join("\\n");
}}

function passes(group) {{
  const q = search.value.toLowerCase();
  if (q && !textOf(group).toLowerCase().includes(q)) return false;
  if (correct.value === "wrong" && !group.some(r => r.is_correct === false)) return false;
  if (correct.value === "correct" && !group.some(r => r.is_correct === true)) return false;
  if (finish.value !== "all" && !group.some(r => String(r.finish_reason) === finish.value)) return false;
  return true;
}}

function renderList() {{
  list.innerHTML = "";
  const visible = [...groups.entries()].filter(([, group]) => passes(group));
  count.textContent = `${{visible.length}} examples`;
  for (const [key, group] of visible) {{
    const first = group[0];
    const item = document.createElement("div");
    item.className = "item" + (key === selectedKey ? " active" : "");
    item.onclick = () => {{ selectedKey = key; renderList(); renderDetail(group); }};
    const verdicts = group.map(r => `${{r.experiment}}:${{r.is_correct ? "correct" : "wrong"}}`).join(" ");
    item.innerHTML = `<strong>${{first.problem_id}}</strong> sample ${{first.sample_id}}<div class="meta">${{verdicts}}</div>`;
    list.appendChild(item);
  }}
  if (visible.length === 0) list.innerHTML = '<div class="empty">No examples match.</div>';
}}

function addKV(parent, key, value) {{
  const div = document.createElement("div");
  div.className = "kv";
  const label = document.createElement("strong");
  label.textContent = `${{key}}: `;
  const span = document.createElement("span");
  span.textContent = mathValue(value);
  div.appendChild(label);
  div.appendChild(span);
  parent.appendChild(div);
}}

function mathValue(value) {{
  const text = String(value ?? "");
  if (text.includes("\\\\") && !text.includes("$")) return `\\\\(${{text}}\\\\)`;
  return text;
}}

function typesetDetail() {{
  if (window.MathJax && window.MathJax.typesetPromise) {{
    window.MathJax.typesetPromise([detail]);
  }}
}}

function renderDetail(group) {{
  const first = group[0];
  detail.innerHTML = "";
  const problem = document.createElement("div");
  problem.className = "problem";
  problem.innerHTML = `<h3>${{first.problem_id}} sample ${{first.sample_id}}</h3>`;
  addKV(problem, "gold", first.gold_answer);
  const p = document.createElement("div");
  p.className = "mathblock";
  p.textContent = first.problem;
  problem.appendChild(p);
  detail.appendChild(problem);

  const cards = document.createElement("div");
  cards.className = "cards";
  for (const r of group) {{
    const card = document.createElement("div");
    card.className = "card " + (r.is_correct ? "correct" : "wrong");
    card.innerHTML = `<h3>${{r.experiment}}</h3>`;
    addKV(card, "correct", r.is_correct);
    addKV(card, "extracted", r.extracted_answer);
    addKV(card, "method", r.extraction_method);
    addKV(card, "finish", r.finish_reason);
    addKV(card, "tokens", r.num_output_tokens);
    const response = document.createElement("div");
    response.className = "mathblock";
    response.textContent = r.response;
    card.appendChild(response);
    cards.appendChild(card);
  }}
  detail.appendChild(cards);
  typesetDetail();
}}

search.oninput = renderList;
correct.onchange = renderList;
finish.onchange = renderList;
renderList();
</script>
</body>
</html>
"""
    output.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problems", type=Path, default=Path("data/processed/math500_simplerl.jsonl"))
    parser.add_argument("--experiments", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verified-name", default="verified.jsonl")
    parser.add_argument("--generations-name", default="generations.jsonl")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    problems = load_problems(args.problems)
    experiments = [
        load_experiment(path, problems, args.verified_name, args.generations_name, args.limit)
        for path in args.experiments
    ]
    write_html(experiments, args.output)
    print(f"Wrote viewer to {args.output}")


if __name__ == "__main__":
    main()
