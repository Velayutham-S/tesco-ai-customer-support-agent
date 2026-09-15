"""
Phase 6 - qualitative generation smoke test (NOT a formal benchmark).

Runs a small fixed set of Tesco customer messages (one per common intent) through the retriever +
prompt builder + LLM client, and prints the generated reply plus minimal diagnostics.

Mock mode is the default (no API key needed). Use --live to call Groq (requires GROQ_API_KEY).
Never prints the API key or the full prompt.

Usage:
  python scripts/evaluate_generation.py            # mock
  python scripts/evaluate_generation.py --live      # real Groq (needs .env key)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import ConfigError
from src.generation import build_generator
from src.retrieval.retriever import DEFAULT_INDEX_DIR

# One representative message per intent (fixed, deterministic).
SAMPLE_MESSAGES = [
    ("delivery_issue", "My Tesco delivery never arrived and I waited in all afternoon."),
    ("missing_or_incorrect_item", "Half my order was missing when the shopping was delivered today."),
    ("refund_or_payment", "I was charged twice for my grocery order, how do I get a refund?"),
    ("clubcard_or_loyalty", "My Clubcard points from last week haven't shown up yet."),
    ("product_availability", "Do you know when the gluten free bread will be back in stock?"),
    ("store_experience", "The queue at my local Tesco Express was huge and only one till was open."),
    ("unclear", "Tesco why is it always like this honestly"),
]


def _stdout_utf8_safe() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # pragma: no cover
        pass


def main() -> int:
    _stdout_utf8_safe()
    parser = argparse.ArgumentParser(description="Qualitative Phase 6 generation smoke test.")
    parser.add_argument("--live", action="store_true", help="call the real Groq API (needs key)")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    args = parser.parse_args()

    mode = "live" if args.live else "mock"
    print(f"Generation smoke test ({mode} mode) - qualitative only, NOT a benchmark.\n")

    try:
        generator = build_generator(mock=not args.live, top_k=args.top_k, index_dir=args.index_dir)
    except ConfigError as exc:
        print("Configuration error:", exc, file=sys.stderr)
        print("Tip: omit --live to run in mock mode.", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print("Retrieval index error:", exc, file=sys.stderr)
        return 2

    for intent, message in SAMPLE_MESSAGES:
        result = generator.generate(message)
        print("=" * 70)
        print(f"[{intent}] customer: {message}")
        print(f"reply: {result.response_text}")
        print(f"  success={result.success} used_fallback={result.used_fallback} "
              f"error_type={result.error_type} retrieved={result.retrieved_count} "
              f"top_sim={result.top_similarity} model={result.model_name}")
    print("=" * 70)
    print("Done. (Replies are model/mock output and are not verified Tesco policy.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
