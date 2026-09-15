"""Tesco support agent workflow (Phase 7).

Orchestrates: validate -> classify intent -> escalation recommendation -> retrieve + generate
(delegated to the Phase 6 response generator, which wraps the Phase 5 retriever) -> AgentResult.

The agent reuses existing components and never re-implements retrieval or generation, and never
generates a second, independent response.
"""

from __future__ import annotations

from typing import Optional

try:
    from src.retrieval.retriever import DEFAULT_INDEX_DIR
    from src.generation import build_generator, validate_message
    from src.agent.models import AgentResult
    from src.agent.intent_classifier import IntentClassifier
    from src.agent.escalation import decide_escalation
except ImportError:  # pragma: no cover
    from retrieval.retriever import DEFAULT_INDEX_DIR  # type: ignore
    from generation import build_generator, validate_message  # type: ignore
    from agent.models import AgentResult  # type: ignore
    from agent.intent_classifier import IntentClassifier  # type: ignore
    from agent.escalation import decide_escalation  # type: ignore


class TescoSupportAgent:
    """Runs the full support workflow and returns a structured AgentResult."""

    def __init__(self, response_generator, intent_classifier: IntentClassifier,
                 mode: str = "live", model_name: Optional[str] = None) -> None:
        self.response_generator = response_generator
        self.intent_classifier = intent_classifier
        self.mode = mode
        self.model_name = model_name

    def handle(self, message: object) -> AgentResult:
        """Execute the workflow. Raises InputValidationError on invalid input."""
        steps: list[str] = []

        clean = validate_message(message)  # reuse Phase 6 validation; raises on bad input
        steps.append("validate_input")

        intent = self.intent_classifier.classify(clean)
        steps.append("classify_intent")

        escalation = decide_escalation(clean, intent)
        steps.append("escalation_policy")

        # Phase 6 generator performs retrieval (via the Phase 5 retriever) AND generation.
        generation = self.response_generator.generate(clean)
        steps.append("retrieve_and_generate")

        return AgentResult(
            customer_message=clean,
            intent=intent.intent,
            intent_confidence=intent.confidence,
            intent_method=intent.method,
            intent_evidence=intent.evidence,
            retrieved_count=generation.retrieved_count,
            top_similarity=generation.top_similarity,
            generated_response=generation.response_text,
            generation_success=generation.success,
            used_fallback=generation.used_fallback,
            error_type=generation.error_type,
            needs_human_review=escalation.needs_human_review,
            escalation_reason=escalation.reason,
            model_name=self.model_name,
            mode=self.mode,
            workflow_steps=steps,
        )


def build_agent(mock: bool = False, deterministic_only: bool = False,
                top_k: int = 5, index_dir=DEFAULT_INDEX_DIR) -> TescoSupportAgent:
    """Factory that wires the real retriever + generator + intent classifier.

    - mock=True: generation uses the offline mock client and no Groq call is made anywhere.
    - deterministic_only=True: intent classification never calls Groq (generation still follows mode).
    - Live generation requires GROQ_API_KEY (raises ConfigError if missing) - use mock otherwise.
    """
    response_generator = build_generator(mock=mock, top_k=top_k, index_dir=index_dir)
    use_groq_intent = (not mock) and (not deterministic_only)
    intent_llm = response_generator.llm_client if use_groq_intent else None
    classifier = IntentClassifier(llm_client=intent_llm, use_groq=use_groq_intent)
    mode = "mock" if mock else "live"
    model_name = getattr(response_generator.llm_client, "model_name", None)
    return TescoSupportAgent(response_generator, classifier, mode=mode, model_name=model_name)
