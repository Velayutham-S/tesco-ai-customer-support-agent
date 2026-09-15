"""Response generation orchestration for Phase 6.

Ties together: input validation -> Phase 5 retrieval -> grounded prompt -> LLM client -> structured
result. Retrieval is delegated to the existing retriever (never re-implemented). All system failures
degrade to a safe, customer-facing fallback; invalid input raises InputValidationError.

Logging note (privacy): failures are logged with an error *type* only. The full prompt is never
logged, and the customer message is only logged at DEBUG level.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

try:
    from src.config import get_groq_settings
    from src.retrieval.retriever import TescoRetriever, DEFAULT_INDEX_DIR
    from src.generation.llm_client import (
        GroqLLMClient, LLMClient, LLMError, MockLLMClient,
    )
    from src.generation.prompt_builder import build_prompt
except ImportError:  # pragma: no cover
    from config import get_groq_settings  # type: ignore
    from retrieval.retriever import TescoRetriever, DEFAULT_INDEX_DIR  # type: ignore
    from generation.llm_client import GroqLLMClient, LLMClient, LLMError, MockLLMClient  # type: ignore
    from generation.prompt_builder import build_prompt  # type: ignore

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 5
MAX_MESSAGE_LENGTH = 2000

FALLBACK_TEXT = (
    "Sorry, I'm unable to provide a complete answer right now. Please try again later or contact "
    "Tesco customer support through the official support channel."
)


class InputValidationError(ValueError):
    """Raised when the incoming customer message is invalid."""


@dataclass
class GenerationResult:
    """Structured result usable from the CLI, tests, and future agent/UI layers."""

    response_text: str
    success: bool
    used_fallback: bool
    error_type: Optional[str] = None
    retrieved_count: int = 0
    top_similarity: Optional[float] = None
    model_name: Optional[str] = None


def validate_message(message: object) -> str:
    """Validate + normalize the customer message, or raise InputValidationError."""
    if message is None:
        raise InputValidationError("customer message is required (got None)")
    if not isinstance(message, str):
        raise InputValidationError("customer message must be a string")
    stripped = message.strip()
    if not stripped:
        raise InputValidationError("customer message must not be empty or whitespace-only")
    if len(stripped) > MAX_MESSAGE_LENGTH:
        raise InputValidationError(
            f"customer message is too long (>{MAX_MESSAGE_LENGTH} characters)")
    return stripped


class ResponseGenerator:
    """Generate a grounded support reply. Retriever and LLM client are injected (test-friendly)."""

    def __init__(self, retriever, llm_client: LLMClient, top_k: int = DEFAULT_TOP_K) -> None:
        self.retriever = retriever
        self.llm_client = llm_client
        self.top_k = top_k

    def _retrieve(self, message: str) -> list[dict]:
        """Retrieve via the Phase 5 retriever; on failure degrade to a no-context path."""
        try:
            return self.retriever.retrieve(message, top_k=self.top_k)
        except Exception as exc:
            logger.warning("retrieval failed (%s); continuing with no historical context",
                           type(exc).__name__)
            return []

    def _fallback(self, error_type: str, examples: list[dict]) -> GenerationResult:
        return GenerationResult(
            response_text=FALLBACK_TEXT,
            success=False,
            used_fallback=True,
            error_type=error_type,
            retrieved_count=len(examples),
            top_similarity=(examples[0]["retrieval_score"] if examples else None),
            model_name=getattr(self.llm_client, "model_name", None),
        )

    def generate(self, message: object) -> GenerationResult:
        """Full pipeline. Raises InputValidationError for bad input; otherwise returns a result."""
        clean_message = validate_message(message)  # raises on invalid input
        logger.debug("generating response for message: %s", clean_message)

        examples = self._retrieve(clean_message)

        try:
            system_prompt, user_prompt = build_prompt(clean_message, examples, max_examples=self.top_k)
        except Exception as exc:
            logger.warning("prompt build failed (%s)", type(exc).__name__)
            return self._fallback("prompt_error", examples)

        try:
            text = self.llm_client.generate(system_prompt, user_prompt)
        except LLMError as exc:
            logger.warning("LLM generation failed (%s)", type(exc).__name__)
            return self._fallback("llm_error", examples)
        except Exception as exc:  # defensive: any unexpected client error
            logger.warning("unexpected LLM error (%s)", type(exc).__name__)
            return self._fallback("llm_error", examples)

        if not text or not text.strip():
            return self._fallback("empty_response", examples)

        return GenerationResult(
            response_text=text.strip(),
            success=True,
            used_fallback=False,
            error_type=None,
            retrieved_count=len(examples),
            top_similarity=(examples[0]["retrieval_score"] if examples else None),
            model_name=getattr(self.llm_client, "model_name", None),
        )


def build_generator(mock: bool = False, top_k: int = DEFAULT_TOP_K,
                    index_dir=DEFAULT_INDEX_DIR) -> ResponseGenerator:
    """Factory: load the real retriever and choose the mock or live Groq client.

    Live mode reads GROQ settings and will raise ConfigError if the API key is missing.
    Mock mode requires no API key and never contacts Groq.
    """
    retriever = TescoRetriever.load(index_dir)
    if mock:
        client: LLMClient = MockLLMClient()
    else:
        client = GroqLLMClient(get_groq_settings())
    return ResponseGenerator(retriever, client, top_k=top_k)
