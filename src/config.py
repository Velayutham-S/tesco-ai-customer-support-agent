"""Central configuration for the Hiver AI support-agent project.

Single source of truth for the selected brand and key paths. The selected brand is read from
`reports/selected_brand_summary.json` (produced by `scripts/analyze_brand_candidates.py`) so the
brand name is never hard-coded or duplicated across the codebase. Later phases should import from
here rather than repeating literals.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

# Repo root = parent of this file's directory (src/)
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CONFIG_DIR)

# Load a local .env (if present) so GROQ_* variables are available. The real .env is
# git-ignored; only .env.example (placeholders) is committed.
try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(REPO_ROOT, ".env"))
except Exception:  # python-dotenv is optional; shell env vars still work
    pass

# --- paths ---
DATASET_PATH = os.path.join(REPO_ROOT, "dataset", "twcs", "twcs.csv")
SAMPLE_PATH = os.path.join(REPO_ROOT, "dataset", "sample.csv")
REPORTS_DIR = os.path.join(REPO_ROOT, "reports")
DATA_DIR = os.path.join(REPO_ROOT, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
SELECTED_BRAND_SUMMARY = os.path.join(REPORTS_DIR, "selected_brand_summary.json")

# --- reproducibility ---
RANDOM_SEED = 42

# --- Groq LLM (Phase 6) ---
# NOTE: the assignment suggested "llama-3.3-70b-versatile", but that model now returns
# 404 model_not_found on Groq (verified via models.list on 2026-09-15) - it has been
# decommissioned. We default to an available, fast general model instead. Override anytime
# with the GROQ_MODEL environment variable. Other available chat options seen on the account
# include: openai/gpt-oss-120b, groq/compound, qwen/qwen3.8-27b.
GROQ_MODEL_DEFAULT = "openai/gpt-oss-20b"


class ConfigError(RuntimeError):
    """Raised for missing/invalid configuration. Never contains secret values."""


@dataclass(frozen=True)
class GroqSettings:
    """Groq API settings resolved from the environment (the API key is never logged)."""

    api_key: Optional[str]
    model: str

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key)

    def require_api_key(self) -> str:
        """Return the key for live mode, or raise a clear (secret-free) error if missing."""
        if not self.api_key:
            raise ConfigError(
                "GROQ_API_KEY is not set. Create a .env in the project root with "
                "GROQ_API_KEY=your_key (see .env.example), or run with --mock."
            )
        return self.api_key


def get_groq_settings() -> GroqSettings:
    """Single place that reads GROQ_* environment variables."""
    return GroqSettings(
        api_key=os.environ.get("GROQ_API_KEY"),
        model=os.environ.get("GROQ_MODEL") or GROQ_MODEL_DEFAULT,
    )


def load_brand_summary() -> dict:
    """Load the Phase 2 brand-selection summary, or raise a clear, actionable error."""
    if not os.path.exists(SELECTED_BRAND_SUMMARY):
        raise FileNotFoundError(
            f"{SELECTED_BRAND_SUMMARY} not found. Run Phase 2 first:\n"
            "  python scripts/analyze_brand_candidates.py "
            "--dataset dataset/twcs/twcs.csv --output-dir reports"
        )
    with open(SELECTED_BRAND_SUMMARY, encoding="utf-8") as fh:
        return json.load(fh)


# The selected brand is resolved at import time from the Phase 2 summary (single source of truth).
try:
    BRAND_SUMMARY = load_brand_summary()
    SELECTED_BRAND = BRAND_SUMMARY.get("selected_brand")
except FileNotFoundError:
    BRAND_SUMMARY = {}
    SELECTED_BRAND = None


if __name__ == "__main__":
    print("SELECTED_BRAND :", SELECTED_BRAND)
    print("DATASET_PATH   :", DATASET_PATH)
    print("REPORTS_DIR    :", REPORTS_DIR)
    print("PROCESSED_DIR  :", PROCESSED_DIR)
    print("RANDOM_SEED    :", RANDOM_SEED)
    _s = get_groq_settings()
    print("GROQ_MODEL     :", _s.model)
    print("GROQ_API_KEY   :", "configured" if _s.has_api_key else "missing")
