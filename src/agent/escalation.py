"""Transparent, recommendation-only human-review policy for the Phase 7 agent.

This module ONLY recommends whether a message should be seen by a human. It never creates a
ticket, contacts staff, processes refunds, or takes any external action. Rules are explainable
keyword/intent checks evaluated in a fixed priority order; the reason is short and observable.
"""

from __future__ import annotations

try:
    from src.text_utils import clean_for_heuristic
    from src.agent.models import EscalationDecision, IntentPrediction, MEDIUM_CONFIDENCE
except ImportError:  # pragma: no cover
    from text_utils import clean_for_heuristic  # type: ignore
    from agent.models import EscalationDecision, IntentPrediction, MEDIUM_CONFIDENCE  # type: ignore

HUMAN_REQUEST = [
    "speak to a human", "speak to someone", "talk to someone", "talk to a person",
    "real person", "real human", "human agent", "speak to an advisor",
    "speak to a representative", "want a person", "speak to a manager",
]
LEGAL_SAFETY = [
    "legal action", "take legal", "solicitor", "lawyer", "ombudsman",
    "trading standards", "sue you", "take you to court", "court",
]
UNRESOLVED = [
    "still not", "still missing", "still waiting", "still haven't", "still no",
    "nobody has helped", "no one has helped", "no-one has helped", "been ignored",
    "second time", "third time", "again and again", "weeks ago", "keep contacting",
]
PAYMENT_TERMS = [
    "charged twice", "double charged", "duplicate charge", "overcharged",
    "refund", "money back", "charged me twice", "took the money",
]
ACCOUNT_TERMS = [
    "can't log in", "cant log in", "can't login", "cant login", "locked out",
    "account access", "reset my password", "hacked", "can't access my account",
]
# Intents that typically need live order/account data the agent cannot see.
ORDER_LIVE_INTENTS = {"delivery_issue", "missing_or_incorrect_item", "online_order_issue"}


def _contains_any(text: str, phrases: list[str]) -> bool:
    return any(p in text for p in phrases)


def decide_escalation(message: str, intent: IntentPrediction) -> EscalationDecision:
    """Return a human-review recommendation with a short, observable reason.

    Rules are checked in priority order; the first match wins.
    """
    clean = clean_for_heuristic(message)

    if _contains_any(clean, HUMAN_REQUEST):
        return EscalationDecision(True, "Customer explicitly requested human support.")
    if _contains_any(clean, LEGAL_SAFETY):
        return EscalationDecision(True, "Legal or safety concern; recommend human review.")
    if intent.intent == "refund_or_payment" or _contains_any(clean, PAYMENT_TERMS):
        return EscalationDecision(
            True, "Payment or refund issue requires account-level verification; the agent cannot process refunds.")
    if intent.intent == "account_or_technical_issue" and _contains_any(clean, ACCOUNT_TERMS):
        return EscalationDecision(True, "Account access issue requires identity verification.")
    if _contains_any(clean, UNRESOLVED):
        return EscalationDecision(True, "Repeated or unresolved issue detected.")
    if intent.intent in ORDER_LIVE_INTENTS:
        return EscalationDecision(True, "Request requires live order information.")
    if intent.intent == "other_or_unclear" or intent.confidence < MEDIUM_CONFIDENCE:
        return EscalationDecision(True, "Low-confidence or unclear intent; recommend human review.")
    return EscalationDecision(False, "No escalation signals detected; suitable for an automated draft reply.")
