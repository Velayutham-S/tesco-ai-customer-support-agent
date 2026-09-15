"""
Phase 8 - qualitative demo of the complete Tesco support agent.

Runs a small, fixed set of representative customer messages through the full workflow and prints
intent, confidence, escalation recommendation, retrieval count, and the generated reply.

Mock mode by default (no API key needed). Use --live to call Groq (needs GROQ_API_KEY).

Usage:
  python scripts/demo_agent.py --mock
  python scripts/demo_agent.py --live
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import ConfigError
from src.agent import build_agent
from src.retrieval.retriever import DEFAULT_INDEX_DIR

DEMO_MESSAGES = [
    "My delivery has not arrived",
    "Is bread available?",
    "I was charged twice for my order",
    "My Clubcard points are missing",
    "I cannot log into my account",
    "I am very disappointed with the service",
    "I want to speak to a human",
]


def _stdout_utf8_safe() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # pragma: no cover
        pass


def main() -> int:
    _stdout_utf8_safe()
    parser = argparse.ArgumentParser(description="Qualitative demo of the Tesco support agent.")
    parser.add_argument("--live", action="store_true", help="use real Groq (needs GROQ_API_KEY)")
    parser.add_argument("--mock", action="store_true", help="offline mock generation (default)")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    args = parser.parse_args()

    mock = not args.live  # mock is the default; --live overrides
    print(f"Tesco support agent demo ({'live' if args.live else 'mock'} mode) - "
          "qualitative only; replies are not verified Tesco policy.\n")

    try:
        agent = build_agent(mock=mock, top_k=args.top_k, index_dir=args.index_dir)
    except ConfigError as exc:
        print("Configuration error:", exc, file=sys.stderr)
        print("Tip: omit --live to run in mock mode.", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print("Retrieval index error:", exc, file=sys.stderr)
        return 2

    for message in DEMO_MESSAGES:
        result = agent.handle(message)
        print("=" * 72)
        print(f"customer message   : {message}")
        print(f"intent             : {result.intent}  (confidence {result.intent_confidence})")
        print(f"needs_human_review : {result.needs_human_review}  -> {result.escalation_reason}")
        print(f"retrieved_examples : {result.retrieved_count}")
        print(f"response           : {result.generated_response}")
    print("=" * 72)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
