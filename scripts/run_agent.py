"""
CLI - run the full Tesco support agent on a customer message (Phase 7, +interactive in Phase 8).

Examples:
  python scripts/run_agent.py --message "My delivery has not arrived"
  python scripts/run_agent.py --message "My delivery has not arrived" --mock
  python scripts/run_agent.py --message "My delivery has not arrived" --mock --json
  python scripts/run_agent.py --message "My delivery has not arrived" --top-k 5 --mock
  python scripts/run_agent.py --interactive --mock
  python scripts/run_agent.py --message "Where is the nearest Tesco?" --deterministic-only

Never prints the API key or the full prompt. Use --mock to run without a Groq key.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import ConfigError
from src.generation import InputValidationError
from src.agent import build_agent
from src.retrieval.retriever import DEFAULT_INDEX_DIR


def _stdout_utf8_safe() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # pragma: no cover
        pass


def _print_human(result) -> None:
    print("=== Customer-facing response ===")
    print(result.generated_response)
    print()
    print("=== Agent diagnostics ===")
    print(f"intent             : {result.intent}")
    print(f"intent_confidence  : {result.intent_confidence}")
    print(f"intent_method      : {result.intent_method}")
    print(f"intent_evidence    : {result.intent_evidence}")
    print(f"retrieved_count    : {result.retrieved_count}")
    print(f"top_similarity     : {result.top_similarity}")
    print(f"generation_success : {result.generation_success}")
    print(f"used_fallback      : {result.used_fallback}")
    print(f"error_type         : {result.error_type}")
    print(f"needs_human_review : {result.needs_human_review}")
    print(f"escalation_reason  : {result.escalation_reason}")
    print(f"mode               : {result.mode}")
    print(f"model_name         : {result.model_name}")


def _run_once(agent, message: str, as_json: bool) -> int:
    try:
        result = agent.handle(message)
    except InputValidationError as exc:
        print("Invalid message:", exc, file=sys.stderr)
        return 2
    if as_json:
        print(result.to_json())
    else:
        _print_human(result)
    return 0


def main() -> int:
    _stdout_utf8_safe()
    parser = argparse.ArgumentParser(description="Run the Tesco support agent.")
    parser.add_argument("--message", help="incoming customer message")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--mock", action="store_true", help="offline mock generation (no API key)")
    parser.add_argument("--deterministic-only", action="store_true",
                        help="deterministic intent classification only (no Groq for intent)")
    parser.add_argument("--json", action="store_true", help="print only the AgentResult as JSON")
    parser.add_argument("--interactive", action="store_true", help="prompt for messages in a loop")
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    args = parser.parse_args()

    if not args.interactive and not args.message:
        print("Error: provide --message \"...\" or use --interactive.", file=sys.stderr)
        return 2

    try:
        agent = build_agent(mock=args.mock, deterministic_only=args.deterministic_only,
                            top_k=args.top_k, index_dir=args.index_dir)
    except ConfigError as exc:
        print("Configuration error:", exc, file=sys.stderr)
        print("Tip: run with --mock, or set GROQ_API_KEY in a .env file (see .env.example).",
              file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print("Retrieval index error:", exc, file=sys.stderr)
        return 2

    if args.interactive:
        print(f"Interactive mode ({'mock' if args.mock else 'live'}). "
              "Type a customer message, or 'quit'/'exit' (or empty) to stop.")
        while True:
            try:
                message = input("\ncustomer> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if message.lower() in ("", "quit", "exit"):
                break
            _run_once(agent, message, args.json)
        return 0

    return _run_once(agent, args.message, args.json)


if __name__ == "__main__":
    raise SystemExit(main())
