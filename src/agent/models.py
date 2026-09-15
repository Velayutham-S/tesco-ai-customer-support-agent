"""Structured, JSON-serializable models and thresholds for the Phase 7 agent."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

# Heuristic confidence thresholds (NOT calibrated probabilities). Centralized here.
HIGH_CONFIDENCE = 0.80
MEDIUM_CONFIDENCE = 0.55


def confidence_band(confidence: float) -> str:
    """Map a heuristic confidence to a coarse band."""
    if confidence >= HIGH_CONFIDENCE:
        return "high"
    if confidence >= MEDIUM_CONFIDENCE:
        return "medium"
    return "low"


@dataclass
class IntentPrediction:
    """Result of intent classification."""

    intent: str
    confidence: float
    method: str            # "deterministic" | "groq" | "groq_fallback"
    evidence: str          # short, observable evidence only (no hidden reasoning)


@dataclass
class EscalationDecision:
    """Human-review recommendation (recommendation only - no action is taken)."""

    needs_human_review: bool
    reason: str            # short, observable reason


@dataclass
class AgentResult:
    """Full agent output: a customer-facing response plus transparent diagnostics.

    JSON-serializable via `to_dict()` / `to_json()`. Contains no API keys, tweet IDs,
    raw dataset rows, prompt contents, or hidden reasoning.
    """

    customer_message: str
    intent: str
    intent_confidence: float
    intent_method: str
    intent_evidence: str
    retrieved_count: int
    top_similarity: Optional[float]
    generated_response: str
    generation_success: bool
    used_fallback: bool
    error_type: Optional[str]
    needs_human_review: bool
    escalation_reason: str
    model_name: Optional[str]
    mode: str
    workflow_steps: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        import json
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
