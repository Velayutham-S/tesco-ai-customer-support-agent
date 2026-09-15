"""
Phase 8 - evaluate the complete agent on the golden set (mock generation).

Runs every golden customer message through the full agent workflow (mock mode, so no API key is
needed) and reports aggregate metrics. Intent accuracy is measured against the golden set's
PROVISIONAL labels, so it reflects agreement with a rule-based labeller, NOT factual correctness
of the generated replies (no human review was performed here).

Outputs:
  reports/phase8_agent_evaluation.json
  reports/phase8_agent_evaluation.md

Usage:
  python scripts/evaluate_agent.py --mock
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import config
from src.agent import build_agent
from src.retrieval.retriever import DEFAULT_INDEX_DIR


def _stdout_utf8_safe() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # pragma: no cover
        pass


def evaluate(golden: pd.DataFrame, agent) -> dict:
    total = len(golden)
    correct = 0
    escalation_count = 0
    success_count = 0
    retrieved_total = 0
    mismatches: Counter = Counter()
    per_intent_total: Counter = Counter()
    per_intent_correct: Counter = Counter()

    for _, row in golden.iterrows():
        expected = str(row["intent"])
        result = agent.handle(str(row["customer_text"]))
        predicted = result.intent
        per_intent_total[expected] += 1
        if predicted == expected:
            correct += 1
            per_intent_correct[expected] += 1
        else:
            mismatches[f"{expected} -> {predicted}"] += 1
        if result.needs_human_review:
            escalation_count += 1
        if result.generation_success and not result.used_fallback:
            success_count += 1
        retrieved_total += result.retrieved_count

    incorrect = total - correct
    return {
        "total_examples": total,
        "correct_intents": correct,
        "incorrect_intents": incorrect,
        "intent_accuracy": round(correct / total, 4) if total else 0.0,
        "escalation_recommended_count": escalation_count,
        "escalation_rate": round(escalation_count / total, 4) if total else 0.0,
        "successful_response_count": success_count,
        "failed_response_count": total - success_count,
        "response_success_rate": round(success_count / total, 4) if total else 0.0,
        "average_retrieved_examples": round(retrieved_total / total, 3) if total else 0.0,
        "top_intent_mismatches": mismatches.most_common(10),
        "per_intent_accuracy": {
            k: round(per_intent_correct[k] / per_intent_total[k], 3)
            for k in sorted(per_intent_total)
        },
    }


def write_markdown(metrics: dict, golden_path: str, reports_dir: Path) -> Path:
    m = metrics
    L: list[str] = []
    a = L.append
    a("# Phase 8 - Agent Evaluation (mock mode)")
    a("")
    a(f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by `scripts/evaluate_agent.py`._")
    a("")
    a("## Objective")
    a("Measure the complete agent's behaviour on the golden set end-to-end in mock generation mode: "
      "how often its intent prediction agrees with the golden label, how often it recommends human "
      "review, and whether it returns a response and retrieves context. This is an **agreement / "
      "smoke** evaluation, not a measure of factual reply correctness.")
    a("## Dataset")
    a(f"- `{golden_path}` - {m['total_examples']} examples.")
    a("- Labels are **PROVISIONAL** (rule-based, not human-verified).")
    a("## Method")
    a("- Each golden customer message is run through the full agent (`build_agent(mock=True)`): "
      "validate -> classify intent -> escalation recommendation -> retrieve -> mock generation.")
    a("- Intent accuracy = predicted intent == golden (provisional) intent.")
    a("- Mock generation is deterministic and needs no API key.")
    a("## Metrics")
    a("")
    a("| Metric | Value |")
    a("| --- | ---: |")
    a(f"| Total examples | {m['total_examples']} |")
    a(f"| Intent accuracy (vs provisional labels) | {m['intent_accuracy']} |")
    a(f"| Correct intents | {m['correct_intents']} |")
    a(f"| Incorrect intents | {m['incorrect_intents']} |")
    a(f"| Escalation recommended | {m['escalation_recommended_count']} ({m['escalation_rate']}) |")
    a(f"| Successful responses | {m['successful_response_count']} |")
    a(f"| Failed responses | {m['failed_response_count']} |")
    a(f"| Response success rate | {m['response_success_rate']} |")
    a(f"| Average retrieved examples | {m['average_retrieved_examples']} |")
    a("")
    a("### Per-intent accuracy (vs provisional labels)")
    a("")
    a("| Intent | Accuracy |")
    a("| --- | ---: |")
    for k, v in m["per_intent_accuracy"].items():
        a(f"| `{k}` | {v} |")
    a("")
    a("## Error analysis")
    a("Most common intent disagreements (`golden -> predicted`):")
    a("")
    if m["top_intent_mismatches"]:
        for pair, count in m["top_intent_mismatches"]:
            a(f"- `{pair}` x{count}")
    else:
        a("- (none)")
    a("")
    a("These disagreements are expected and honest, not necessarily agent errors:")
    a("- The **agent** re-ranks overlapping intents with an agent-specific priority (e.g. a duplicate "
      "charge -> `refund_or_payment` even if it mentions an online order), whereas the golden "
      "**provisional** labels used the taxonomy's own priority order.")
    a("- The golden set was created **before** the `product_availability` keyword fix, so some "
      "availability-style messages were provisionally labelled `other_or_unclear`.")
    a("## Known limitations")
    a("- Provisional labels are not human ground truth; accuracy here is *agreement*, not correctness.")
    a("- Mock mode does not call the LLM, so reply quality is not assessed (responses are placeholders).")
    a("- Retrieval is lexical TF-IDF; escalation is a recommendation only.")
    a("## Reproducibility")
    a("```bash")
    a("python scripts/evaluate_agent.py --mock")
    a("```")
    a("")
    path = reports_dir / "phase8_agent_evaluation.md"
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path


def main() -> int:
    _stdout_utf8_safe()
    parser = argparse.ArgumentParser(description="Evaluate the complete agent on the golden set (mock).")
    parser.add_argument("--mock", action="store_true", help="use mock generation (default; kept for clarity)")
    parser.add_argument("--golden", default=str(Path(config.PROCESSED_DIR) / "tesco_golden_set.parquet"))
    parser.add_argument("--reports-dir", default=str(config.REPORTS_DIR))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    args = parser.parse_args()

    golden_path = Path(args.golden)
    if not golden_path.exists():
        alt = golden_path.with_suffix(".csv")
        golden_path = alt if alt.exists() else golden_path
    if not golden_path.exists():
        print(f"ERROR: golden set not found at {args.golden}", file=sys.stderr)
        return 2

    golden = pd.read_parquet(golden_path) if golden_path.suffix == ".parquet" else pd.read_csv(golden_path)
    # Evaluation always uses mock generation so it runs without an API key.
    agent = build_agent(mock=True, top_k=args.top_k, index_dir=args.index_dir)
    print(f"Evaluating {len(golden)} golden examples (mock generation) ...", flush=True)

    metrics = evaluate(golden, agent)
    metrics["golden_path"] = str(golden_path).replace("\\", "/")
    metrics["labels"] = "provisional"
    metrics["generation_mode"] = "mock"
    metrics["generated_at"] = datetime.now().isoformat(timespec="seconds")

    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "phase8_agent_evaluation.json"
    json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    md_path = write_markdown(metrics, metrics["golden_path"], reports_dir)

    print("\n===== AGENT EVALUATION (mock; provisional labels) =====", flush=True)
    print(f"examples          : {metrics['total_examples']}")
    print(f"intent_accuracy   : {metrics['intent_accuracy']}")
    print(f"escalation_recomm : {metrics['escalation_recommended_count']} ({metrics['escalation_rate']})")
    print(f"response_success  : {metrics['response_success_rate']}")
    print(f"avg_retrieved     : {metrics['average_retrieved_examples']}")
    print(f"top mismatches    : {metrics['top_intent_mismatches'][:5]}")
    print(f"\nreports:\n  {json_path}\n  {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
