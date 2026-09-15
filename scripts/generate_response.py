"""
Phase 6 CLI - generate a grounded Tesco support reply for one customer message.

Examples:
  python scripts/generate_response.py --message "My delivery has not arrived" --top-k 5
  python scripts/generate_response.py --message "My delivery has not arrived" --mock

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
from src.generation import build_generator, InputValidationError
from src.retrieval.retriever import DEFAULT_INDEX_DIR


def _stdout_utf8_safe() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # pragma: no cover
        pass


def main() -> int:
    _stdout_utf8_safe()
    parser = argparse.ArgumentParser(description="Generate a Tesco support reply (Phase 6).")
    parser.add_argument("--message", required=True, help="incoming customer message")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--mock", action="store_true", help="use the offline mock LLM (no API key)")
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    args = parser.parse_args()

    try:
        generator = build_generator(mock=args.mock, top_k=args.top_k, index_dir=args.index_dir)
    except ConfigError as exc:
        print("Configuration error:", exc, file=sys.stderr)
        print("Tip: run with --mock, or set GROQ_API_KEY in a .env file (see .env.example).",
              file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print("Retrieval index error:", exc, file=sys.stderr)
        return 2

    try:
        result = generator.generate(args.message)
    except InputValidationError as exc:
        print("Invalid message:", exc, file=sys.stderr)
        return 2

    print("=== Customer-facing response ===")
    print(result.response_text)
    print()
    print("=== Diagnostics ===")
    print(f"mode            : {'mock' if args.mock else 'live'}")
    print(f"success         : {result.success}")
    print(f"used_fallback   : {result.used_fallback}")
    print(f"error_type      : {result.error_type}")
    print(f"retrieved_count : {result.retrieved_count}")
    print(f"top_similarity  : {result.top_similarity}")
    print(f"model_name      : {result.model_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
