"""Hybrid intent classifier for the Phase 7 agent.

Deterministic first (reusing the Phase 4 taxonomy's keyword rules and the 11 official labels),
with an OPTIONAL Groq fallback for low-confidence messages. The label set is imported from
`src.intent_taxonomy` - this module never defines a second, conflicting list of labels.

For overlapping messages the agent re-ranks the taxonomy's matched intents with an
agent-specific priority (documented below) so the *primary customer problem* wins, e.g. a
duplicate charge outranks the fact that it was an online order. This does not modify the
Phase 4 taxonomy; it is an agent-layer decision.
"""

from __future__ import annotations

import json
import re
from typing import Optional

try:
    from src.intent_taxonomy import PRIORITY_ORDER, OTHER_INTENT, intent_matches, _COMPILED
    from src.text_utils import clean_for_heuristic
    from src.agent.models import IntentPrediction, MEDIUM_CONFIDENCE
    from src.generation.llm_client import LLMClient
except ImportError:  # pragma: no cover
    from intent_taxonomy import PRIORITY_ORDER, OTHER_INTENT, intent_matches, _COMPILED  # type: ignore
    from text_utils import clean_for_heuristic  # type: ignore
    from agent.models import IntentPrediction, MEDIUM_CONFIDENCE  # type: ignore
    from generation.llm_client import LLMClient  # type: ignore

# Agent-specific priority for resolving overlaps (uses the SAME 11 labels as the taxonomy).
# Rationale: a specific loyalty signal (clubcard) is unambiguous; a missing item is the primary
# problem even if delivered; a money dispute (refund/charge) outranks the ordering/delivery channel.
AGENT_PRIORITY = [
    "clubcard_or_loyalty",
    "missing_or_incorrect_item",
    "refund_or_payment",
    "account_or_technical_issue",
    "delivery_issue",
    "online_order_issue",
    "product_availability",
    "store_experience",
    "complaint",
    "general_information",
    OTHER_INTENT,
]
assert set(AGENT_PRIORITY) == set(PRIORITY_ORDER), "AGENT_PRIORITY must match the taxonomy labels"

# A few high-signal phrases per intent that justify a HIGH confidence when present.
# Keys are validated against the official labels; this is a confidence hint, not a new label set.
STRONG_PHRASES: dict[str, list[str]] = {
    "clubcard_or_loyalty": ["clubcard", "club card", "loyalty points"],
    "missing_or_incorrect_item": ["missing item", "items missing", "item missing", "wrong item",
                                  "incorrect item", "missing half"],
    "refund_or_payment": ["charged twice", "double charged", "duplicate charge", "want a refund",
                          "overcharged", "money back"],
    "delivery_issue": ["never arrived", "hasn't arrived", "has not arrived", "delivery slot",
                       "not been delivered", "didn't arrive"],
    "online_order_issue": ["online order", "checkout", "my basket", "order confirmation"],
    "product_availability": ["out of stock", "in stock", "back in stock", "sold out"],
    "account_or_technical_issue": ["can't log in", "cant log in", "reset my password",
                                   "locked out", "website not working", "app not working"],
    "store_experience": ["self checkout", "at the till", "in store", "tesco express"],
    "complaint": ["formal complaint", "poor service", "terrible service", "unacceptable"],
    "general_information": ["opening hours", "nearest store", "how do i", "do you offer"],
}
assert set(STRONG_PHRASES).issubset(set(PRIORITY_ORDER)), "STRONG_PHRASES keys must be valid labels"

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


class IntentClassifier:
    """Deterministic classifier with an optional Groq fallback for low-confidence cases."""

    def __init__(self, llm_client: Optional[LLMClient] = None, use_groq: bool = False) -> None:
        self.llm_client = llm_client
        self.use_groq = use_groq and llm_client is not None

    # ------------------------------------------------------------------ public
    def classify(self, message: str) -> IntentPrediction:
        det = self._deterministic(message)
        if det.confidence >= MEDIUM_CONFIDENCE or not self.use_groq:
            return det
        # low-confidence deterministic result -> try Groq, else keep the safe deterministic result
        groq_pred = self._groq_classify(message)
        return groq_pred if groq_pred is not None else det

    # --------------------------------------------------------- deterministic
    def _deterministic(self, message: str) -> IntentPrediction:
        clean = clean_for_heuristic(message)
        matched = intent_matches(message)  # taxonomy keyword matches (excludes fallback)
        if not matched:
            return IntentPrediction(OTHER_INTENT, 0.30, "deterministic", "no keyword rule matched")

        primary = next(name for name in AGENT_PRIORITY if name in matched)
        n_hits = len(_COMPILED[primary].findall(clean)) or 1
        competing = max(len(matched) - 1, 0)
        strong = any(p in clean for p in STRONG_PHRASES.get(primary, []))

        conf = 0.72 + 0.08 * min(n_hits - 1, 2)      # 1 hit -> 0.72, up to +0.16
        if strong:
            conf = max(conf, 0.88)
        conf -= 0.06 * min(competing, 3)             # competing intents reduce confidence
        conf = round(min(max(conf, 0.35), 0.98), 2)

        evidence = f"matched taxonomy rule for '{primary}'"
        if strong:
            evidence += "; strong phrase match"
        if competing:
            others = [m for m in matched if m != primary]
            evidence += f"; also matched: {', '.join(others)}"
        return IntentPrediction(primary, conf, "deterministic", evidence)

    # ---------------------------------------------------------------- groq
    def _groq_classify(self, message: str) -> Optional[IntentPrediction]:
        """Ask Groq to pick one label. Returns None on ANY failure (caller falls back)."""
        labels = ", ".join(PRIORITY_ORDER)
        system = (
            "You are an intent classifier for Tesco customer support. Classify the customer "
            f"message into exactly one intent from this list: {labels}. "
            'Respond with ONLY a JSON object of the form '
            '{"intent": "<one_label>", "confidence": <0..1>, "evidence": "<short reason>"}. '
            "Do not add any text before or after the JSON. Do not perform any other task."
        )
        try:
            raw = self.llm_client.generate(system, message)  # type: ignore[union-attr]
            match = _JSON_OBJ_RE.search(raw or "")
            if not match:
                return None
            data = json.loads(match.group(0))
            intent = data.get("intent")
            confidence = data.get("confidence")
            evidence = str(data.get("evidence", ""))[:200]
            if intent not in PRIORITY_ORDER:
                return None
            confidence = float(confidence)
            if not (0.0 <= confidence <= 1.0):
                return None
            return IntentPrediction(intent, round(confidence, 2), "groq",
                                    evidence or "classified by Groq")
        except Exception:
            # Any failure (network/LLMError/JSON/type/label) -> safe fallback handled by caller.
            return None
